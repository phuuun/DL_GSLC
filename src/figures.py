"""Step 7: the five report figures + the Section IV comparison table.

Everything comes from checkpoints/ and results/*.json, nothing is retrained.
Usage: python src/figures.py   ->  figures/fig1..fig5.png, results/model_comparison.csv
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from ae import AE
from common import DEVICE, get_loaders, set_seed
from ddpm import Diffusion, UNet
from gan import build, make_sampler
from mlp_probe import MLPClassifier, make_decoder, transpose_from_class, transpose_from_latent
from vae import VAE

CLASSES = ["t-shirt", "trouser", "pullover", "dress", "coat",
           "sandal", "shirt", "sneaker", "bag", "ankle boot"]
CMAP = "gray_r"          # dark garment on white paper; use "gray" for the raw look
INK, MUTED = "#111111", "#9a9a9a"

plt.rcParams.update({
    "font.size": 8, "font.family": "sans-serif",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 8, "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 4,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "lines.linewidth": 1.0, "legend.frameon": False, "legend.fontsize": 7,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def load(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


def strip(imgs, nrow=None):
    """[N,1,28,28] in [0,1] -> one 2-D array; padding is 0 which is white under gray_r."""
    return make_grid(imgs.detach().cpu().clamp(0, 1), nrow=nrow or len(imgs), padding=2, pad_value=0)[0].numpy()


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def image_rows(rows, labels, path, col_labels=None, width=7.0):
    fig, axes = plt.subplots(len(rows), 1, figsize=(width, 0.72 * len(rows)))
    for i, (ax, imgs, lab) in enumerate(zip(axes, rows, labels)):
        ax.imshow(strip(imgs), cmap=CMAP, vmin=0, vmax=1)
        bare(ax)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", color=INK, labelpad=6)
        if i == 0 and col_labels:
            ax.set_xticks([30 * j + 16 for j in range(len(col_labels))], col_labels, color=MUTED)
            ax.xaxis.tick_top(); ax.tick_params(length=0)
    fig.subplots_adjust(hspace=0.08)
    fig.savefig(path); plt.close(fig)


@torch.no_grad()
def one_per_class(loader):
    """First test image of each class, ordered 0..9."""
    picked = {}
    for x, y in loader:
        for xi, yi in zip(x, y.tolist()):
            picked.setdefault(yi, xi)
        if len(picked) == 10:
            break
    assert sorted(picked) == list(range(10))
    return torch.stack([picked[c] for c in range(10)]).to(DEVICE)


def relstd_last10(v):
    v = np.array(v[-10:])
    return float(v.std() / abs(v.mean()))


def main():
    set_seed()
    os.makedirs("figures", exist_ok=True)
    _, _, test = get_loaders()
    x = one_per_class(test)
    R = {n: json.load(open(f"results/{n}.json")) for n in
         ("vanilla_ae", "sparse_ae", "vae", "vanilla_gan", "dcgan", "ddpm", "mlp_probe")}

    ae = load(AE(), "checkpoints/vanilla_ae.pt")
    sae = load(AE(), "checkpoints/sparse_ae.pt")
    vae = load(VAE(), "checkpoints/vae.pt")
    clf = load(MLPClassifier(), "checkpoints/mlp_classifier.pt")
    dec = load(make_decoder(), "checkpoints/mlp_frozen_decoder.pt")
    stats = torch.load("checkpoints/mlp_latent_stats.pt", map_location=DEVICE)
    G_gan, _ = build(False); load(G_gan, "checkpoints/vanilla_gan_G.pt")
    G_dc, _ = build(True); load(G_dc, "checkpoints/dcgan_G.pt")
    diff = Diffusion(load(UNet(), "checkpoints/ddpm.pt"))

    with torch.no_grad():
        # ---- fig 1: reconstruction ----
        image_rows([x, ae(x)[0], sae(x)[0], vae.decode(vae.encode(x)[0])],
                   ["original", "vanilla ae", "sparse ae", "vae (μ)"],
                   "figures/fig1_reconstruction.png", col_labels=CLASSES)

        # ---- fig 2: 16 samples per generative model ----
        ddpm_out, traj = diff.sample(16, keep_every=100)   # ~1000 U-Net calls, the slow bit
        samples = [("vae", vae.sample(16)), ("vanilla gan", make_sampler(G_gan)(16)),
                   ("dcgan", make_sampler(G_dc)(16)), ("ddpm", ddpm_out)]
        keys = ["vae", "vanilla_gan", "dcgan", "ddpm"]
        fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.1))
        for ax, (name, s), k in zip(axes, samples, keys):
            ax.imshow(strip(s, nrow=4), cmap=CMAP, vmin=0, vmax=1)
            bare(ax)
            ax.set_title(f"{name}   fid {R[k]['metrics']['fid']:.1f}", color=INK)
        fig.subplots_adjust(wspace=0.08)
        fig.savefig("figures/fig2_samples.png"); plt.close(fig)

        # ---- fig 3: MLP classifier is not a generator ----
        z_mlp = clf.features(x)
        prior = dec(stats["mu"] + stats["sigma"] * torch.randn(10, 64, device=DEVICE))
        image_rows([x, transpose_from_class(clf), transpose_from_latent(clf, z_mlp),
                    ae(x)[0], dec(z_mlp), prior],
                   ["original", "Wᵀ from y_c", "Wᵀ from z_mlp", "vanilla ae (z_ae)",
                    "frozen dec (z_mlp)", "prior sample"],
                   "figures/fig3_mlp_probe.png", col_labels=CLASSES)

        # ---- fig 4: z(α) = (1-α) z_A + α z_B, α in {0, .2, ..., 1} as the brief asks ----
        a = torch.linspace(0, 1, 6, device=DEVICE)[:, None]
        pair = x[[1, 7]]                                    # encoders: trouser -> sneaker
        rows = []
        for enc, decode in ((ae.encode, ae.decode), (sae.encode, sae.decode),
                            (lambda t: vae.encode(t)[0], vae.decode)):
            zA, zB = enc(pair)
            rows.append(decode((1 - a) * zA + a * zB))
        for G in (G_gan, G_dc):                             # no encoder: two random prior draws
            zA, zB = torch.randn(2, 100, device=DEVICE)
            rows.append((G((1 - a) * zA + a * zB) + 1) / 2)
        image_rows(rows, ["vanilla ae", "sparse ae", "vae (μ)", "vanilla gan", "dcgan"],
                   "figures/fig4_interpolation.png", width=4.2,
                   col_labels=[f"α={v:.1f}" for v in a.squeeze(1).tolist()])

    # ---- fig 5: training curves (top) + DDPM reverse trajectory (bottom) ----
    fig = plt.figure(figsize=(7.5, 4.4))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.05], hspace=0.06)
    axes = top.subplots(1, 6)
    curves = [
        ("vanilla ae", "vanilla_ae", ("train_loss", "val_loss")),
        ("sparse ae", "sparse_ae", ("train_loss", "val_loss")),
        ("vae", "vae", ("train_loss", "val_loss")),
        ("vanilla gan", "vanilla_gan", ("d_loss", "g_loss")),
        ("dcgan", "dcgan", ("d_loss", "g_loss")),
        ("ddpm", "ddpm", ("train_loss", "val_loss")),
    ]
    for ax, (title, k, (a_key, b_key)) in zip(axes, curves):
        h = R[k]["history"]
        ep = np.arange(1, len(h[a_key]) + 1)
        ax.plot(ep, h[a_key], color=INK, label=a_key.split("_")[0])
        ax.plot(ep, h[b_key], color=MUTED, label=b_key.split("_")[0])
        ax.set_title(title, color=INK)
        ax.set_xlabel("epoch", color=MUTED)
        ax.legend(loc="upper right", handlelength=1)
    top.subplots_adjust(wspace=0.55)

    ax = bottom.subplots()
    frames = torch.stack(traj, 1)[:3]                       # [3 samples, 11 steps, 1, 28, 28]
    ax.imshow(strip(frames.flatten(0, 1), nrow=frames.shape[1]), cmap=CMAP, vmin=0, vmax=1)
    bare(ax)
    steps = [999] + list(range(900, -1, -100))
    ax.set_xticks([30 * j + 16 for j in range(len(steps))], [f"t={s}" for s in steps], color=MUTED)
    ax.tick_params(length=0)
    ax.set_title("ddpm reverse process", color=INK)
    fig.savefig("figures/fig5_training.png"); plt.close(fig)

    # ---- Section IV table ----
    names = {"vanilla_ae": "Vanilla AE", "sparse_ae": "Sparse AE", "vae": "VAE",
             "vanilla_gan": "Vanilla GAN", "dcgan": "DCGAN", "ddpm": "DDPM"}
    stab_key = {"vanilla_gan": "g_loss", "dcgan": "g_loss"}
    table = []
    for k, label in names.items():
        m, h = R[k]["metrics"], R[k]["history"]
        sk = stab_key.get(k, "val_loss")
        table.append({
            "model": label, "latent": m["latent_dim"],
            "epoch_time_s": round(m["epoch_time_s"], 2), "latency_ms_b64": round(m["latency_ms"], 2),
            "mse": round(m["mse"], 5) if "mse" in m else "N/A",
            "psnr_db": round(m["psnr"], 2) if "psnr" in m else "N/A",
            "ssim": round(m["ssim"], 3) if "ssim" in m else "N/A",
            "fid": round(m["fid"], 1) if "fid" in m else "N/A",
            "last10_relstd": f"{relstd_last10(h[sk]) * 100:.1f}% ({sk})",
            "final_D(G(z))": round(h["d_fake"][-1], 2) if "d_fake" in h else "",
        })
    with open("results/model_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table[0].keys())
        w.writeheader(); w.writerows(table)

    cols = list(table[0])
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in table:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    for p in ("fig1_reconstruction", "fig2_samples", "fig3_mlp_probe", "fig4_interpolation", "fig5_training"):
        assert os.path.getsize(f"figures/{p}.png") > 0
    print("figures/ done")


if __name__ == "__main__":
    main()
