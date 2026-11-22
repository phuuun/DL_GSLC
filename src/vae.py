"""Variational Autoencoder.

Encoder q(z|x) = N(mu(x), diag(sigma(x)^2)), prior p(z) = N(0, I), latent 64.
Architecture: 784 -> 256 -> (mu, logvar) in R^64 -> 256 -> 784 (same size as the AE).
Loss (negative ELBO, per image):
    L = BCE(x, x_hat) summed over pixels + KL(q(z|x) || N(0, I))
    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
Reparameterisation: z = mu + sigma * eps, eps ~ N(0, I), so gradients flow through mu, sigma.
"""
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEVICE, eval_reconstruction, fid_score, get_loaders, latency_ms, set_seed

LATENT = 64
HIDDEN = 256
EPOCHS = 30


class VAE(nn.Module):
    def __init__(self, latent=LATENT, hidden=HIDDEN):
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.dec(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n):
        return self.decode(torch.randn(n, self.latent, device=DEVICE))


def loss_fn(x, x_hat, mu, logvar):
    bce = F.binary_cross_entropy(x_hat, x, reduction="sum") / len(x)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
    return bce + kl, bce, kl


def run_epoch(model, loader, opt=None):
    model.train(opt is not None)
    tot = torch.zeros(3)
    n = 0
    with torch.set_grad_enabled(opt is not None):
        for x, _ in loader:
            x = x.to(DEVICE)
            loss, bce, kl = loss_fn(x, *model(x))
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += torch.tensor([loss.item(), bce.item(), kl.item()]) * len(x)
            n += len(x)
    return (tot / n).tolist()


def main():
    set_seed()
    train, val, test = get_loaders()
    model = VAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    hist = {"train_loss": [], "val_loss": [], "train_bce": [], "train_kl": []}
    times = []
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        tr, bce, kl = run_epoch(model, train, opt)
        times.append(time.perf_counter() - t0)
        va = run_epoch(model, val)[0]
        for k, v in zip(hist, (tr, va, bce, kl)):
            hist[k].append(v)
        print(f"[vae] epoch {ep + 1:02d} train {tr:.2f} (bce {bce:.2f} kl {kl:.2f}) val {va:.2f} ({times[-1]:.1f}s)")

    model.eval()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/vae.pt")
    # reconstruction uses the posterior mean (no sampling noise) so it's comparable to the AEs
    metrics = eval_reconstruction(lambda x: model.decode(model.encode(x)[0]), test)
    metrics["fid"] = fid_score(model.sample, test)
    metrics["latency_ms"] = latency_ms(model.sample)
    metrics["epoch_time_s"] = sum(times) / len(times)
    metrics["latent_dim"] = LATENT
    print(json.dumps(metrics, indent=2))
    with open("results/vae.json", "w") as f:
        json.dump({"metrics": metrics, "history": hist}, f, indent=2)


if __name__ == "__main__":
    main()
