"""Vanilla GAN (fully connected) and DCGAN (convolutional). Same training loop, different nets.

Minimax objective:
    min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))],  z ~ N(0, I) in R^100
In practice:
    L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)
    L_G = BCE(D(G(z)), 1)     (non-saturating trick: stronger gradients early, when D easily wins)
Images are scaled to [-1, 1] to match the generator's Tanh output.

Usage: python src/gan.py            (vanilla GAN)
       python src/gan.py --dcgan    (DCGAN)
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEVICE, fid_score, get_loaders, latency_ms, set_seed

Z_DIM = 100


# ---------- Vanilla GAN: plain MLPs, no normalisation ----------

class MLPGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ---------- DCGAN (Radford et al. 2016 guidelines) ----------
# strided (transposed) convs instead of pooling, BatchNorm in both nets,
# ReLU in G / LeakyReLU in D, Tanh output, no fully connected hidden layers.

class DCGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, 256, 7, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(),  # 7x7
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),    # 14x14
            nn.ConvTranspose2d(128, 1, 4, 2, 1), nn.Tanh(),                                       # 28x28
        )

    def forward(self, z):
        return self.net(z.view(-1, Z_DIM, 1, 1))


class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),                               # 14x14
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # 7x7
            nn.Conv2d(128, 1, 7, 1, 0),                                                 # 1x1
        )

    def forward(self, x):
        return self.net(x).view(-1)


def dcgan_init(m):
    # DCGAN paper: all weights from N(0, 0.02)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def build(dcgan):
    if dcgan:
        G, D = DCGenerator(), DCDiscriminator()
        G.apply(dcgan_init)
        D.apply(dcgan_init)
    else:
        G, D = MLPGenerator(), MLPDiscriminator()
    return G.to(DEVICE), D.to(DEVICE)


def make_sampler(G):
    """z ~ N(0, I) -> image in [0, 1]."""
    @torch.no_grad()
    def sample(n):
        return (G(torch.randn(n, Z_DIM, device=DEVICE)) + 1) / 2
    return sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dcgan", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()
    name = "dcgan" if args.dcgan else "vanilla_gan"

    set_seed()
    train, _, test = get_loaders()
    G, D = build(args.dcgan)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = F.binary_cross_entropy_with_logits

    # per-epoch means plus raw per-iteration losses (every 10 steps) to show oscillation for Q3
    hist = {k: [] for k in ("d_loss", "g_loss", "d_real", "d_fake", "iter_d", "iter_g")}
    times, step = [], 0
    for ep in range(args.epochs):
        G.train(); D.train()
        t0 = time.perf_counter()
        sums = torch.zeros(4)
        for x, _ in train:
            x = x.to(DEVICE) * 2 - 1
            b = len(x)
            ones, zeros = torch.ones(b, device=DEVICE), torch.zeros(b, device=DEVICE)

            # D step: push D(x) -> 1, D(G(z)) -> 0
            fake = G(torch.randn(b, Z_DIM, device=DEVICE))
            d_real, d_fake = D(x), D(fake.detach())
            loss_d = bce(d_real, ones) + bce(d_fake, zeros)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # G step: push D(G(z)) -> 1
            loss_g = bce(D(fake), ones)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            sums += torch.tensor([loss_d.item(), loss_g.item(),
                                  torch.sigmoid(d_real).mean().item(), torch.sigmoid(d_fake).mean().item()])
            if step % 10 == 0:
                hist["iter_d"].append(loss_d.item())
                hist["iter_g"].append(loss_g.item())
            step += 1
        times.append(time.perf_counter() - t0)
        means = (sums / len(train)).tolist()
        for k, v in zip(("d_loss", "g_loss", "d_real", "d_fake"), means):
            hist[k].append(v)
        print(f"[{name}] epoch {ep + 1:02d} D {means[0]:.3f} G {means[1]:.3f} "
              f"D(x) {means[2]:.2f} D(G(z)) {means[3]:.2f} ({times[-1]:.1f}s)")

    G.eval()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.save(G.state_dict(), f"checkpoints/{name}_G.pt")
    torch.save(D.state_dict(), f"checkpoints/{name}_D.pt")

    sample = make_sampler(G)
    metrics = {
        "fid": fid_score(sample, test),
        "latency_ms": latency_ms(sample),
        "epoch_time_s": sum(times) / len(times),
        "latent_dim": Z_DIM,
    }
    print(json.dumps(metrics, indent=2))
    with open(f"results/{name}.json", "w") as f:
        json.dump({"metrics": metrics, "history": hist}, f, indent=2)


if __name__ == "__main__":
    main()
