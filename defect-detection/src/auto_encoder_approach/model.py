import torch.nn as nn


def down(i, o):  # halves H, W
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1, bias=False),
                         nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=True))


def up(i, o):  # doubles H, W
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ConvAE(nn.Module):
    """256x256x3 -> 8x8xlatent_channels -> 256x256x3."""

    def __init__(self, latent_channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            down(3, 32),      # 128
            down(32, 64),     # 64
            down(64, 128),    # 32
            down(128, 256),   # 16
            down(256, 256),   # 8
            nn.Conv2d(256, latent_channels, 1),  # bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 1), nn.ReLU(inplace=True),
            up(256, 256),     # 16
            up(256, 128),     # 32
            up(128, 64),      # 64
            up(64, 32),       # 128
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # 256
            nn.Sigmoid(),     # inputs are in [0, 1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))