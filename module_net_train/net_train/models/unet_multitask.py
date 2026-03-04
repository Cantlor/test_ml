from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None



def _norm_layer(norm: str, channels: int) -> nn.Module:
    if norm == "batchnorm":
        return nn.BatchNorm2d(channels)
    if norm == "instancenorm":
        return nn.InstanceNorm2d(channels, affine=True)
    return nn.Identity()


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm_layer(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm_layer(norm, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str, dropout: float) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, norm=norm, dropout=dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, norm=norm, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)

        # safe alignment for odd spatial sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


@dataclass
class UNetMultiTaskConfig:
    in_channels: int
    base_channels: int = 16
    depth: int = 4
    norm: str = "batchnorm"
    dropout: float = 0.0


class UNetMultiTask(nn.Module):
    """Baseline U-Net with two segmentation heads: extent and boundary."""

    def __init__(self, cfg: UNetMultiTaskConfig) -> None:
        super().__init__()

        if cfg.depth < 2:
            raise ValueError("depth must be >= 2")

        self.cfg = cfg
        chs = [cfg.base_channels * (2 ** i) for i in range(cfg.depth)]

        self.stem = ConvBlock(cfg.in_channels, chs[0], norm=cfg.norm, dropout=cfg.dropout)

        self.downs = nn.ModuleList()
        for i in range(1, cfg.depth):
            self.downs.append(DownBlock(chs[i - 1], chs[i], norm=cfg.norm, dropout=cfg.dropout))

        self.ups = nn.ModuleList()
        for i in range(cfg.depth - 1, 0, -1):
            self.ups.append(UpBlock(chs[i], chs[i - 1], chs[i - 1], norm=cfg.norm, dropout=cfg.dropout))

        self.extent_head = nn.Conv2d(chs[0], 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x):
        skips = []

        x = self.stem(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            x = up(x, skip)

        extent_logits = self.extent_head(x)
        boundary_logits = self.boundary_head(x)

        return {
            "extent_logits": extent_logits,
            "boundary_logits": boundary_logits,
        }



def build_unet_multitask_from_cfg(model_cfg: Dict) -> UNetMultiTask:
    cfg = UNetMultiTaskConfig(
        in_channels=int(model_cfg.get("in_channels", 8)),
        base_channels=int(model_cfg.get("base_channels", 16)),
        depth=int(model_cfg.get("depth", 4)),
        norm=str(model_cfg.get("norm", "batchnorm")).lower(),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    return UNetMultiTask(cfg)
