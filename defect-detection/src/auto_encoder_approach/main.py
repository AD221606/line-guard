import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur

from dataset import MVTecDataset
from model import ConvAE


@torch.no_grad()
def anomaly_maps(model, x):
    """Per-pixel squared reconstruction error, blurred to suppress edge noise."""
    err = ((x - model(x)) ** 2).mean(dim=1, keepdim=True)  # B,1,H,W
    return gaussian_blur(err, kernel_size=11, sigma=4.0)


def train(args, device):
    ds = MVTecDataset(args.root, args.category, "train", args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = ConvAE(args.latent_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, *_ in dl:
            x = x.to(device)
            loss = F.mse_loss(model(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train mse {total / len(ds):.6f}")

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.ckpt)
    return model


def evaluate(args, model, device):
    ds = MVTecDataset(args.root, args.category, "test", args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model.eval()

    img_scores, labels, types = [], [], []
    pix_scores, pix_gt = [], []
    for x, mask, label, dtype in dl:
        amap = anomaly_maps(model, x.to(device)).cpu()
        img_scores += amap.flatten(1).max(dim=1).values.tolist()  # image score = max pixel error
        labels += label.tolist()
        types += list(dtype)
        pix_scores.append(amap.flatten().numpy())
        pix_gt.append(mask.flatten().numpy())

    img_scores, labels = np.array(img_scores), np.array(labels)
    print(f"\nImage AUROC: {roc_auc_score(labels, img_scores):.4f}")
    print(f"Pixel AUROC: {roc_auc_score(np.concatenate(pix_gt), np.concatenate(pix_scores)):.4f}")

    # per-defect-type breakdown (each defect type vs. all good images)
    good = img_scores[np.array(types) == "good"]
    for t in sorted(set(types) - {"good"}):
        d = img_scores[np.array(types) == t]
        auc = roc_auc_score([0] * len(good) + [1] * len(d), np.concatenate([good, d]))
        print(f"  {t:20s} n={len(d):3d}  AUROC {auc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="../../dataset/mvtec_anomaly_detection")
    p.add_argument("--category", default="bottle")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--latent_channels", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ckpt", default="checkpoints/ae_bottle.pt")
    p.add_argument("--eval_only", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvAE(args.latent_channels).to(device)
    if args.eval_only:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    else:
        model = train(args, device)
    evaluate(args, model, device)