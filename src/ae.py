"""Vanilla Autoencoder and Sparse Autoencoder (same MLP, sparse adds an L1 penalty).

Architecture: 784 -> 256 -> 64 (z) -> 256 -> 784, mirrors the MLP classifier encoder.
Vanilla loss: L = ||x - x_hat||^2
Sparse loss:  L = ||x - x_hat||^2 + lambda * ||h||_1, h = 256-d hidden encoder activations

Usage: python src/ae.py            (vanilla)
       python src/ae.py --sparse   (sparse)
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEVICE, eval_reconstruction, get_loaders, latency_ms, set_seed

LATENT = 64
HIDDEN = 256


class AE(nn.Module):
    def __init__(self, latent=LATENT, hidden=HIDDEN):
        super().__init__()
        self.enc_h = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU())
        self.enc_z = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, 784), nn.Sigmoid(),
        )

    def encode(self, x, return_h=False):
        h = self.enc_h(x)
        z = self.enc_z(h)
        return (z, h) if return_h else z

    def decode(self, z):
        return self.dec(z).view(-1, 1, 28, 28)

    def forward(self, x):
        z, h = self.encode(x, return_h=True)
        return self.decode(z), h


def run_epoch(model, loader, lam, opt=None):
    model.train(opt is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for x, _ in loader:
            x = x.to(DEVICE)
            x_hat, h = model(x)
            loss = F.mse_loss(x_hat, x) + lam * h.abs().sum(1).mean()
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sparse", action="store_true")
    p.add_argument("--lam", type=float, default=1e-4, help="L1 sparsity weight")
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    name = "sparse_ae" if args.sparse else "vanilla_ae"
    lam = args.lam if args.sparse else 0.0

    set_seed()
    train, val, test = get_loaders()
    model = AE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    hist = {"train_loss": [], "val_loss": []}
    times = []
    for ep in range(args.epochs):
        t0 = time.perf_counter()
        tr = run_epoch(model, train, lam, opt)
        times.append(time.perf_counter() - t0)
        va = run_epoch(model, val, lam)
        hist["train_loss"].append(tr)
        hist["val_loss"].append(va)
        print(f"[{name}] epoch {ep + 1:02d} train {tr:.5f} val {va:.5f} ({times[-1]:.1f}s)")

    model.eval()
    metrics = eval_reconstruction(lambda x: model(x)[0], test)

    # sparsity stats on the 256-d hidden layer (for Q2)
    with torch.no_grad():
        hs = torch.cat([model.encode(x.to(DEVICE), return_h=True)[1] for x, _ in test])
    metrics["hidden_zero_frac"] = (hs == 0).float().mean().item()
    metrics["hidden_mean_act"] = hs.mean().item()

    x64 = next(iter(test))[0][:64].to(DEVICE)
    metrics["latency_ms"] = latency_ms(lambda b: model(x64[:b])[0])
    metrics["epoch_time_s"] = sum(times) / len(times)
    metrics["latent_dim"] = LATENT
    metrics["lambda"] = lam
    print(json.dumps(metrics, indent=2))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{name}.pt")
    with open(f"results/{name}.json", "w") as f:
        json.dump({"metrics": metrics, "history": hist}, f, indent=2)


if __name__ == "__main__":
    main()
