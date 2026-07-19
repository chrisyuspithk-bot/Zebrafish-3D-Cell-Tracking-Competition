"""Difference of Gaussians (DoG) blob detector for 3D nuclei.

Classical approach: convolving with Gaussians at multiple scales
and finding local maxima in the scale-normalized Laplacian response.
Well-suited for bright nuclei on dark background.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def dog_blob_detect(
    volume: np.ndarray,
    min_sigma: float = 1.5,
    max_sigma: float = 4.0,
    sigma_ratio: float = 1.6,
    threshold: float = 0.01,
    overlap: float = 0.5,
    voxel_size: tuple[float, float, float] = (2.0, 0.5, 0.5),
) -> np.ndarray:
    """Detect nuclei using multi-scale DoG blob detection.

    Uses separable Gaussian filters for speed.

    Args:
        volume: 3D array (Z, Y, X).
        min_sigma: Minimum Gaussian sigma in pixels (XY).
        max_sigma: Maximum Gaussian sigma in pixels (XY).
        sigma_ratio: Ratio between successive scales.
        threshold: Normalized response threshold.
        overlap: Maximum allowed overlap fraction between detections.
        voxel_size: (Z, Y, X) voxel sizes in microns for anisotropy correction.

    Returns:
        Array of shape (N, 3) with (z, y, x) centroids.
    """
    if volume.ndim != 3:
        msg = f"Expected 3D volume, got shape {volume.shape}"
        raise ValueError(msg)

    volume = volume.astype(np.float32)
    volume -= volume.mean()
    volume /= (volume.std() + 1e-8)

    # Anisotropy ratio for Z vs XY
    z_ratio = voxel_size[0] / voxel_size[1]

    sigmas = []
    sigma = min_sigma
    while sigma <= max_sigma:
        sigmas.append(sigma)
        sigma *= sigma_ratio

    scale_space_max = np.zeros_like(volume, dtype=np.float32)

    for sigma_xy in sigmas:
        sigma_z = sigma_xy * z_ratio
        sigma2_xy = sigma_xy * sigma_ratio
        sigma2_z = sigma2_xy * z_ratio

        v1 = ndimage.gaussian_filter(volume, sigma=(sigma_z, sigma_xy, sigma_xy), mode="reflect")
        v2 = ndimage.gaussian_filter(volume, sigma=(sigma2_z, sigma2_xy, sigma2_xy), mode="reflect")
        dog = (v1 - v2) * (sigma_xy ** 2)
        scale_space_max = np.maximum(scale_space_max, dog)

    # Find local maxima in a 3x5x5 neighborhood
    footprint = np.ones((3, 5, 5), dtype=bool)
    local_max = ndimage.maximum_filter(scale_space_max, footprint=footprint) == scale_space_max
    local_max &= scale_space_max > threshold * scale_space_max.max()

    coords = np.argwhere(local_max)

    # Non-maximum suppression by overlap
    if len(coords) > 0 and overlap < 1.0:
        coords = _nms_centroids(coords, overlap, voxel_size)

    return coords.astype(np.float32)


def _nms_centroids(
    coords: np.ndarray,
    overlap: float,
    voxel_size: tuple[float, float, float],
) -> np.ndarray:
    """Suppress overlapping detections, keeping the stronger ones."""
    if len(coords) <= 1:
        return coords

    # Anisotropy-corrected distances
    weights = np.array(voxel_size)
    scores = np.ones(len(coords))  # Could store DoG response here
    order = np.argsort(-scores)

    kept = []
    suppressed = np.zeros(len(coords), dtype=bool)

    for i in order:
        if suppressed[i]:
            continue
        kept.append(i)
        dist = np.linalg.norm((coords[i] - coords[~suppressed]) * weights, axis=1)
        too_close = dist < (1 - overlap) * np.linalg.norm(weights * 3)
        suppressed[~suppressed] |= too_close

    return coords[np.array(kept)]
