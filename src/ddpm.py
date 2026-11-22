"""Denoising Diffusion Probabilistic Model (Ho et al. 2020).

Forward (fixed, no learning):  q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I)
    -> x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,  eps ~ N(0, I)
    linear beta schedule 1e-4 .. 0.02, T = 1000, alpha_t = 1 - beta_t, abar_t = prod alpha
Training (simple loss): L = || eps - eps_theta(x_t, t) ||^2,  t ~ Uniform{1..T}
    -> plain supervised regression, no adversary, so the loss curve is smooth.
Sampling (reverse, T network calls):
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - abar_t) * eps_theta(x_t, t)) + sigma_t z,
    sigma_t^2 = beta_t, z ~ N(0, I) (z = 0 at the last step)
Images live in [-1, 1].
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEVICE, fid_score, get_loaders, latency_ms, set_seed

T = 1000
EPOCHS = 30


# ---------- small U-Net noise predictor eps_theta(x_t, t) ----------
# ponytail: no self-attention layers; fine at 28x28, add at 14x14 if FID needs it.

def time_embedding(t, dim):
    """Sinusoidal embedding of the timestep, same idea as Transformer positions."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]  # inject timestep
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, ch=64, t_dim=256):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.conv_in = nn.Conv2d(1, ch, 3, padding=1)
        self.rb1 = ResBlock(ch, ch, t_dim)                  # 28x28
        self.down1 = nn.Conv2d(ch, ch, 3, 2, 1)             # -> 14x14
        self.rb2 = ResBlock(ch, 2 * ch, t_dim)
        self.down2 = nn.Conv2d(2 * ch, 2 * ch, 3, 2, 1)     # -> 7x7
        self.mid = ResBlock(2 * ch, 2 * ch, t_dim)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(2 * ch, 2 * ch, 3, padding=1))
        self.rb3 = ResBlock(4 * ch, 2 * ch, t_dim)          # 14x14, with skip from rb2
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(2 * ch, 2 * ch, 3, padding=1))
        self.rb4 = ResBlock(3 * ch, ch, t_dim)              # 28x28, with skip from rb1
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x, t):
        te = self.t_mlp(time_embedding(t, self.t_dim))
        h1 = self.rb1(self.conv_in(x), te)
        h2 = self.rb2(self.down1(h1), te)
        m = self.mid(self.down2(h2), te)
        u = self.rb3(torch.cat([self.up2(m), h2], 1), te)
        u = self.rb4(torch.cat([self.up1(u), h1], 1), te)
        return self.out(u)


# ---------- diffusion process ----------

class Diffusion:
    def __init__(self, model):
        self.model = model
        self.beta = torch.linspace(1e-4, 0.02, T, device=DEVICE)
        self.alpha = 1 - self.beta
        self.abar = torch.cumprod(self.alpha, 0)

    def loss(self, x0):
        t = torch.randint(0, T, (len(x0),), device=DEVICE)
        eps = torch.randn_like(x0)
        ab = self.abar[t][:, None, None, None]
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return F.mse_loss(self.model(x_t, t), eps)

    @torch.no_grad()
    def sample(self, n, keep_every=None):
        """x_T ~ N(0, I) -> x_0 in [0, 1]. keep_every=k also returns every k-th x_t (for Figure 5)."""
        x = torch.randn(n, 1, 28, 28, device=DEVICE)
        traj = []
        for t in reversed(range(T)):
            tt = torch.full((n,), t, device=DEVICE, dtype=torch.long)
            eps = self.model(x, tt)
            x = (x - self.beta[t] / (1 - self.abar[t]).sqrt() * eps) / self.alpha[t].sqrt()
            if t > 0:
                x = x + self.beta[t].sqrt() * torch.randn_like(x)
            if keep_every and (t % keep_every == 0 or t == T - 1):
                traj.append(((x.clamp(-1, 1) + 1) / 2).cpu())
        out = (x.clamp(-1, 1) + 1) / 2
        return (out, traj) if keep_every else out


def main():
    set_seed()
    train, val, test = get_loaders()
    model = UNet().to(DEVICE)
    diff = Diffusion(model)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    print(f"U-Net params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    hist = {"train_loss": [], "val_loss": []}
    times = []
    for ep in range(EPOCHS):
        model.train()
        t0 = time.perf_counter()
        tot, n = 0.0, 0
        for x, _ in train:
            x = x.to(DEVICE) * 2 - 1
            loss = diff.loss(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        times.append(time.perf_counter() - t0)

        model.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for x, _ in val:
                x = x.to(DEVICE) * 2 - 1
                vt += diff.loss(x).item() * len(x)
                vn += len(x)
        hist["train_loss"].append(tot / n)
        hist["val_loss"].append(vt / vn)
        print(f"[ddpm] epoch {ep + 1:02d} train {tot / n:.4f} val {vt / vn:.4f} ({times[-1]:.1f}s)")

    model.eval()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/ddpm.pt")

    metrics = {
        "latency_ms": latency_ms(diff.sample, warmup=1, runs=3),  # 1000 steps per batch, so fewer runs
        "epoch_time_s": sum(times) / len(times),
        "latent_dim": "1x28x28",
        "sampling_steps": T,
    }
    print(json.dumps(metrics, indent=2))
    metrics["fid"] = fid_score(diff.sample, test)
    print("fid", metrics["fid"])
    with open("results/ddpm.json", "w") as f:
        json.dump({"metrics": metrics, "history": hist}, f, indent=2)


if __name__ == "__main__":
    main()
