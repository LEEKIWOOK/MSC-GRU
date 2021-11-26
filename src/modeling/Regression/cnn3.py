import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.utils import Flattening


class DoubleConv(nn.Module):
    """반복되는 conv - BN - ReLU 구조 모듈화"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias = False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1, bias = False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[2] - x1.size()[2]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.flattening = Flattening()
        self.predictor = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
        )
        # self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.flattening(x)
        x = self.predictor(x)
        x = x.squeeze()
        return x


class UNet(nn.Module):
    def __init__(self, bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear

        self.inc = DoubleConv(4, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        # self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        # self.down4 = Down(128, 256 // factor)
        self.up1 = Up(64, 32 // factor, bilinear)
        self.up2 = Up(32, 16 // factor, bilinear)
        # self.up3 = Up(64, 32 // factor, bilinear)
        # self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(528, 1)

    def forward(self, x):
        x = F.one_hot(x).to(torch.float)
        x = x.transpose(1, 2)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        out = self.outc(x)
        return out
