"""Section 2: proving an MLP classifier's weights / latent features can't generate images.

Classifier: 784 -> 256 -> 64 (z_MLP) -> 10, ReLU, Cross-Entropy. Models P(y|x), not P(x).

Stage 1  train classifier to >= 88% test accuracy, keep W1 (256x784), W2 (64x256), W3 (10x64)
Stage 2  top-down transpose from one-hot class y_c:
             z = ReLU(W3^T y_c), h1 = ReLU(W2^T z), x = Sigmoid(W1^T h1)
Stage 3  transpose from a real test image's latent:
             x = Sigmoid(W1^T ReLU(W2^T z_MLP))
Stage 4  freeze classifier, train separate decoder 64 -> 256 -> 784 (MSE) on z_MLP,
         compare MSE/PSNR/SSIM with the Vanilla AE (same latent size, same decoder shape)
Stage 5  fit N(mu_z, diag(sigma_z^2)) to test-set z_MLP, sample, decode with the Stage 4 decoder
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEVICE, eval_reconstruction, fid_score, get_loaders, mse_psnr_ssim, set_seed

EPOCHS = 30


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def features(self, x):
        """z_MLP in R^64: the layer right before the final classifier."""
        return F.relu(self.fc2(F.relu(self.fc1(x.flatten(1)))))

    def forward(self, x):
        return self.fc3(self.features(x))


def make_decoder():
    """Same shape as the Vanilla AE decoder, so the only difference is where z comes from."""
    return nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(),
        nn.Linear(256, 784), nn.Sigmoid(),
        nn.Unflatten(1, (1, 28, 28)),
    )


# ---------- Stage 2 & 3: transpose projections (no training, biases ignored as in the brief) ----------

def transpose_from_class(clf):
    """One image per class, 10 x 1 x 28 x 28."""
    W1, W2, W3 = clf.fc1.weight, clf.fc2.weight, clf.fc3.weight
    y = torch.eye(10, device=W1.device)               # rows = one-hot y_c
    z = F.relu(y @ W3)                                # (W3^T y_c)^T
    h1 = F.relu(z @ W2)
    return torch.sigmoid(h1 @ W1).view(-1, 1, 28, 28)


def transpose_from_latent(clf, z):
    W1, W2 = clf.fc1.weight, clf.fc2.weight
    return torch.sigmoid(F.relu(z @ W2) @ W1).view(-1, 1, 28, 28)


def ortho_error(W):
    """||W^T W - I||_F / ||I||_F. 0 would mean W^T is a perfect inverse."""
    WtW = W.T @ W
    I = torch.eye(WtW.shape[0], device=W.device)
    return (torch.linalg.norm(WtW - I) / torch.linalg.norm(I)).item()


@torch.no_grad()
def accuracy(clf, loader):
    correct = n = 0
    for x, y in loader:
        correct += (clf(x.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
        n += len(y)
    return correct / n


def main():
    set_seed()
    train, val, test = get_loaders()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    out = {"history": {"clf_train_loss": [], "clf_val_acc": [], "dec_train_loss": [], "dec_val_loss": []}}
    hist = out["history"]

    # ----- Stage 1 -----
    clf = MLPClassifier().to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        clf.train()
        tot, n = 0.0, 0
        for x, y in train:
            loss = F.cross_entropy(clf(x.to(DEVICE)), y.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        clf.eval()
        hist["clf_train_loss"].append(tot / n)
        hist["clf_val_acc"].append(accuracy(clf, val))
        print(f"[clf] epoch {ep + 1:02d} loss {tot / n:.4f} val acc {hist['clf_val_acc'][-1]:.4f}")
    test_acc = accuracy(clf, test)
    print(f"[clf] TEST ACC {test_acc:.4f}")
    assert test_acc >= 0.88, "brief requires >= 88% test accuracy"
    torch.save(clf.state_dict(), "checkpoints/mlp_classifier.pt")
    for p in clf.parameters():
        p.requires_grad = False   # frozen from here on

    # ----- Stage 2 & 3: numbers behind the pictures -----
    with torch.no_grad():
        W = {k: getattr(clf, k).weight for k in ("fc1", "fc2", "fc3")}
        z_test = torch.cat([clf.features(x.to(DEVICE)) for x, _ in test])
        stage3 = eval_reconstruction(lambda x: transpose_from_latent(clf, clf.features(x)), test)
    out["stage2_3"] = {
        "test_acc": test_acc,
        # rows of W1^T W1 span at most 256 of 784 dims, so 528 pixel directions are simply gone
        "rank_W1": torch.linalg.matrix_rank(W["fc1"]).item(),
        "ortho_error": {k: ortho_error(w) for k, w in W.items()},
        "z_mlp_zero_frac": (z_test == 0).float().mean().item(),   # ReLU killed these, irreversibly
        "stage3_transpose_recon": stage3,
    }
    print(json.dumps(out["stage2_3"], indent=2))

    # ----- Stage 4: frozen latent decoder -----
    dec = make_decoder().to(DEVICE)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        dec.train()
        tot, n = 0.0, 0
        for x, _ in train:
            x = x.to(DEVICE)
            loss = F.mse_loss(dec(clf.features(x)), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        dec.eval()
        with torch.no_grad():
            vt = sum(F.mse_loss(dec(clf.features(x.to(DEVICE))), x.to(DEVICE)).item() * len(x) for x, _ in val)
        hist["dec_train_loss"].append(tot / n)
        hist["dec_val_loss"].append(vt / len(val.dataset))
        print(f"[frozen-dec] epoch {ep + 1:02d} train {tot / n:.5f} val {hist['dec_val_loss'][-1]:.5f}")
    dec.eval()
    torch.save(dec.state_dict(), "checkpoints/mlp_frozen_decoder.pt")
    out["stage4_frozen_decoder"] = eval_reconstruction(lambda x: dec(clf.features(x)), test)
    with open("results/vanilla_ae.json") as f:
        ae = json.load(f)["metrics"]
    out["stage4_vanilla_ae"] = {k: ae[k] for k in ("mse", "psnr", "ssim")}
    print("frozen z_MLP decoder:", out["stage4_frozen_decoder"])
    print("vanilla AE z_AE     :", out["stage4_vanilla_ae"])

    # ----- Stage 5: prior sampling in z_MLP space -----
    mu, sigma = z_test.mean(0), z_test.std(0)
    torch.save({"mu": mu.cpu(), "sigma": sigma.cpu()}, "checkpoints/mlp_latent_stats.pt")

    @torch.no_grad()
    def sample(n):
        return dec(mu + sigma * torch.randn(n, 64, device=DEVICE))

    with torch.no_grad():
        s = sample(1000)
        # sampled z goes negative where real z_MLP (post-ReLU) never does: off-manifold by construction
        neg_frac = ((mu + sigma * torch.randn(1000, 64, device=DEVICE)) < 0).float().mean().item()
        # does the classifier itself recognise the samples? low confidence = not a coherent object
        conf_fake = F.softmax(clf(s), 1).max(1).values.mean().item()
        conf_real = torch.cat([F.softmax(clf(x.to(DEVICE)), 1).max(1).values for x, _ in test]).mean().item()
    out["stage5_prior_sampling"] = {
        "fid": fid_score(sample, test),
        "sampled_z_negative_frac": neg_frac,
        "clf_confidence_on_samples": conf_fake,
        "clf_confidence_on_real": conf_real,
    }
    print(json.dumps(out["stage5_prior_sampling"], indent=2))

    with open("results/mlp_probe.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
