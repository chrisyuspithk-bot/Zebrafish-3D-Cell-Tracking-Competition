"""3D UNet for cell center heatmap prediction.

Lightweight UNet that predicts a per-voxel likelihood of being a cell center.
Designed for small 3D+t volumes with anisotropic resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


class ConvBlock(nn.Module):
    """Double convolution block with GroupNorm and residual connection."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        g1 = min(groups, in_ch) if in_ch >= groups else 1
        g2 = min(groups, out_ch) if out_ch >= groups else 1
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.relu(self.norm1(x)))
        x = self.conv2(F.relu(self.norm2(x)))
        return x + residual


class UNet3D(nn.Module):
    """Lightweight 3D UNet for center detection heatmaps.

    Encoder-decoder with skip connections. Uses strided convolutions
    for downsampling and transposed convolutions for upsampling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        # Encoder
        self.encoders = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(ConvBlock(prev, f))
            prev = f

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        bottleneck_ch = features[-1] * 2
        self.bottleneck = ConvBlock(features[-1], bottleneck_ch)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(f * 2, f, 2, stride=2, bias=False))
            self.decoders.append(ConvBlock(f * 2, f))

        self.final = nn.Conv3d(features[0], out_channels, 1)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips), strict=True):
            x = upconv(x)
            x = self._pad_to_match(x, skip)
            x = decoder(torch.cat([skip, x], dim=1))
            x = self.dropout(x)

        return self.final(x)

    @staticmethod
    def _pad_to_match(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dz = target.shape[2] - x.shape[2]
        dy = target.shape[3] - x.shape[3]
        dx = target.shape[4] - x.shape[4]
        if dz == 0 and dy == 0 and dx == 0:
            return x
        return F.pad(x, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2))


class UNetDetector:
    """Wrapper around the UNet model for inference and peak extraction."""

    def __init__(
        self,
        model_path: str | None = None,
        features: list[int] | None = None,
        threshold: float = 0.3,
        peak_min_distance: int = 4,
        device: str = "cpu",
    ) -> None:
        self.threshold = threshold
        self.peak_min_distance = peak_min_distance
        self.device = torch.device(device)

        self.model = UNet3D(
            in_channels=1,
            out_channels=1,
            features=features or [32, 64, 128, 256],
        ).to(self.device)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    @torch.no_grad()
    def predict_heatmap(self, volume: np.ndarray) -> np.ndarray:
        """Predict cell center heatmap from a 3D volume.

        Args:
            volume: (Z, Y, X) or (1, Z, Y, X) numpy array.

        Returns:
            Heatmap of same spatial shape.
        """
        if volume.ndim == 3:
            volume = volume[np.newaxis, np.newaxis]
        elif volume.ndim == 4:
            volume = volume[np.newaxis] if volume.shape[0] != 1 else volume[np.newaxis]

        tensor = torch.as_tensor(volume, dtype=torch.float32, device=self.device)
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)

        output = self.model(tensor)
        heatmap = output.squeeze().cpu().numpy()
        return heatmap

    def extract_centroids(self, heatmap: np.ndarray) -> np.ndarray:
        """Extract cell centroids from heatmap using peak finding.

        Returns:
            Array of shape (N, 3) with (z, y, x) centroids.
        """
        binary = heatmap > self.threshold

        # Label connected components and find their centroids
        labeled, num_features = ndimage.label(binary)
        if num_features == 0:
            return np.empty((0, 3), dtype=np.float32)

        centroids = ndimage.center_of_mass(heatmap, labeled, range(1, num_features + 1))
        centroids = np.array(centroids, dtype=np.float32)

        # Subpixel refinement: local peak around each centroid
        refined = []
        for c in centroids:
            zc, yc, xc = int(round(c[0])), int(round(c[1])), int(round(c[2]))
            r = 3
            z_lo = max(0, zc - r)
            z_hi = min(heatmap.shape[0], zc + r + 1)
            y_lo = max(0, yc - r)
            y_hi = min(heatmap.shape[1], yc + r + 1)
            x_lo = max(0, xc - r)
            x_hi = min(heatmap.shape[2], xc + r + 1)

            patch = heatmap[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
            peak = np.unravel_index(patch.argmax(), patch.shape)
            refined.append([z_lo + peak[0], y_lo + peak[1], x_lo + peak[2]])

        return np.array(refined, dtype=np.float32)

    def detect(self, volume: np.ndarray) -> np.ndarray:
        """Full detection pipeline: heatmap -> centroids."""
        heatmap = self.predict_heatmap(volume)
        return self.extract_centroids(heatmap)
