from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    """MVTec AD loader. train = good only; test = good + defect types with masks."""

    def __init__(self, root, category, split="train", img_size=256):
        self.root = Path(root) / category
        self.img_tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
        self.mask_tf = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])
        self.img_size = img_size
        self.samples = []  # (img_path, mask_path or None, label, defect_type)

        if split == "train":
            for p in sorted((self.root / "train" / "good").glob("*.png")):
                self.samples.append((p, None, 0, "good"))
        else:
            for d in sorted((self.root / "test").iterdir()):
                if not d.is_dir():
                    continue
                for p in sorted(d.glob("*.png")):
                    if d.name == "good":
                        self.samples.append((p, None, 0, "good"))
                    else:
                        m = self.root / "ground_truth" / d.name / f"{p.stem}_mask.png"
                        self.samples.append((p, m, 1, d.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, mask_path, label, dtype = self.samples[i]
        img = self.img_tf(Image.open(img_path).convert("RGB"))
        if mask_path is not None:
            mask = (self.mask_tf(Image.open(mask_path).convert("L")) > 0).float()
        else:
            mask = torch.zeros(1, self.img_size, self.img_size)
        return img, mask, label, dtype