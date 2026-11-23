# Generative models on Fashion-MNIST

Deep Learning D6421 · Sesi 07 take-home.
Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN and DDPM trained under one protocol,
plus a probe showing why an MLP classifier can't act as a generator.

## Protocol

- Fashion-MNIST train + test merged (70k), split 80/10/10 → 56k / 7k / 7k, seed 42
- batch 128, pixels in [0, 1] (GAN/DDPM rescale to [−1, 1] internally)
- all metrics on the 7k test split
- FID: 7k generated vs 7k test images, Inception-v3 at 299×299
- latency: one batch of 64, 3 warmup + 10 runs (DDPM 1 + 3)
- hardware: RTX 3050 Laptop, 4 GB

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/common.py           # sanity check: split sizes, metric behaviour
python src/ae.py               # vanilla AE
python src/ae.py --sparse      # sparse AE, λ = 1e-4
python src/vae.py
python src/gan.py              # vanilla GAN
python src/gan.py --dcgan
python src/ddpm.py             # slowest: ~30 min training + ~45 min FID sampling
python src/mlp_probe.py        # needs vanilla_ae results first
python src/figures.py          # needs every checkpoint above
```

The dataset downloads into `data/` on first run.

## Layout

```
src/common.py      seed, data split, MSE/PSNR/SSIM, FID, latency
src/ae.py          vanilla + sparse AE
src/vae.py         VAE
src/gan.py         vanilla GAN + DCGAN
src/ddpm.py        DDPM, small U-Net, T = 1000
src/mlp_probe.py   MLP classifier, stages 1–5
src/figures.py     figures 1–5 + comparison table
results/           metrics + per-epoch history (json), model_comparison.csv
figures/           report figures
logs/              training output
checkpoints/       weights (gitignored)
```

## Results

| model | latent | s / epoch | latency (ms) | MSE | PSNR | SSIM | FID |
|---|---|---|---|---|---|---|---|
| Vanilla AE | 64 | 2.2 | 0.13 | 0.00648 | 23.00 | 0.867 | – |
| Sparse AE | 64 | 2.2 | 0.13 | 0.01053 | 20.76 | 0.798 | – |
| VAE | 64 | 2.2 | 0.09 | 0.01322 | 19.62 | 0.766 | 86.7 |
| Vanilla GAN | 100 | 2.5 | 0.17 | – | – | – | 121.1 |
| DCGAN | 100 | 10.3 | 2.98 | – | – | – | 19.3 |
| DDPM | 1×28×28 | 63.9 | 24 855 | – | – | – | 17.1 |

AEs have no prior to sample from, so no FID.

MLP classifier probe: test accuracy 89.2%.
Decoding frozen z_MLP gives SSIM 0.713, against 0.867 for the vanilla AE.
Sampling from a Gaussian fitted to z_MLP gives FID 146.0.
