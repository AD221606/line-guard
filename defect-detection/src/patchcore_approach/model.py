import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.transforms.functional import gaussian_blur


class PatchCore:
    def __init__(self, device, coreset_ratio=0.1, target_dim=1024):
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.target_dim = target_dim
        self.net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.bank = None  # (M, target_dim)

    @torch.no_grad()
    def embed(self, x):
        """x: B,3,H,W in [0,1] -> patch features (B*h*w, target_dim), plus (B, h, w)."""
        n = self.net
        x = (x.to(self.device) - self.mean) / self.std
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        l2 = n.layer2(n.layer1(x))          # B,512,H/8,W/8
        l3 = n.layer3(l2)                   # B,1024,H/16,W/16

        # local neighbourhood aggregation: each patch sees its 3x3 neighbours
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.avg_pool2d(l3, 3, 1, 1)
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)

        f = torch.cat([l2, l3], dim=1)      # B,1536,h,w
        B, C, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(-1, C)
        f = F.adaptive_avg_pool1d(f.unsqueeze(1), self.target_dim).squeeze(1)
        return f, (B, h, w)

    @torch.no_grad()
    def fit(self, loader):
        feats = [self.embed(x)[0].cpu() for x, *_ in loader]
        feats = torch.cat(feats).to(self.device)
        print(f"patch features: {tuple(feats.shape)}")
        self.bank = self._greedy_coreset(feats)
        print(f"memory bank:    {tuple(self.bank.shape)}")

    @torch.no_grad()
    def _greedy_coreset(self, feats, proj_dim=128, seed=0):
        """Greedy k-center: repeatedly add the patch farthest from the chosen set."""
        n = feats.size(0)
        k = max(1, int(n * self.coreset_ratio))
        g = torch.Generator().manual_seed(seed)
        proj = torch.randn(feats.size(1), proj_dim, generator=g).to(feats.device)
        z = feats @ proj                    # random projection keeps this fast
        idx = [int(torch.randint(n, (1,), generator=g))]
        min_d = ((z - z[idx[0]]) ** 2).sum(1)
        for _ in range(k - 1):
            i = int(min_d.argmax())
            idx.append(i)
            min_d = torch.minimum(min_d, ((z - z[i]) ** 2).sum(1))
        return feats[idx]

    @torch.no_grad()
    def anomaly_maps(self, x):
        """Returns B,1,H,W map: distance of each patch to its nearest normal patch."""
        f, (B, h, w) = self.embed(x)
        d = torch.cat([torch.cdist(c, self.bank).min(dim=1).values for c in f.split(4096)])
        amap = d.view(B, 1, h, w)
        amap = F.interpolate(amap, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return gaussian_blur(amap, kernel_size=33, sigma=4.0)

    def save(self, path):
        torch.save(self.bank.cpu(), path)

    def load(self, path):
        self.bank = torch.load(path, map_location=self.device)