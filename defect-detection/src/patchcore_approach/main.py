import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import MVTecDataset
from model import PatchCore


def evaluate(args, pc, device):
    ds = MVTecDataset(args.root, args.category, "test", args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    img_scores, labels, types, pix_scores, pix_gt = [], [], [], [], []
    n_imgs, elapsed = 0, 0.0
    for x, mask, label, dtype in dl:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        amap = pc.anomaly_maps(x).cpu()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0
        n_imgs += x.size(0)

        img_scores += amap.flatten(1).max(dim=1).values.tolist()
        labels += label.tolist()
        types += list(dtype)
        pix_scores.append(amap.flatten().numpy())
        pix_gt.append(mask.flatten().numpy())

    img_scores, labels = np.array(img_scores), np.array(labels)
    print(f"\nImage AUROC: {roc_auc_score(labels, img_scores):.4f}")
    print(f"Pixel AUROC: {roc_auc_score(np.concatenate(pix_gt), np.concatenate(pix_scores)):.4f}")
    print(f"Inference:   {1000 * elapsed / n_imgs:.1f} ms/image")

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
    p.add_argument("--coreset_ratio", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ckpt", default="checkpoints/patchcore_bottle.pt")
    p.add_argument("--eval_only", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    pc = PatchCore(device, coreset_ratio=args.coreset_ratio)

    if args.eval_only:
        pc.load(args.ckpt)
    else:
        train_ds = MVTecDataset(args.root, args.category, "train", args.img_size)
        pc.fit(DataLoader(train_ds, batch_size=args.batch_size, shuffle=False))
        Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
        pc.save(args.ckpt)

    evaluate(args, pc, device)