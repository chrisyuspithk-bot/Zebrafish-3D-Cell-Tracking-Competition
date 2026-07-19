"""
3D cell detection in fluorescence microscopy volumes.

Strategy:
1. Background subtraction (percentile-based)
2. 3D Difference-of-Gaussian (DoG) blob enhancement
3. Local peak detection with non-maximum suppression
4. Intensity-based filtering of weak detections

Handles dense cell populations by using multi-scale detection
and adaptive thresholding.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage.feature import peak_local_max


def detect_cells_3d(
    volume: np.ndarray,
    sigma: float = 3.0,
    threshold_rel: float = 0.02,
    min_distance: int = 5,
    sigma_ratio: float = 1.6,
) -> np.ndarray:
    """
    Detect cells in a 3D volume (Z, Y, X).

    Smooths the volume with a 3D Gaussian, then finds local maxima
    above an adaptive threshold based on intensity percentiles.

    Args:
        volume: 3D array (Z, Y, X), typically uint16
        sigma: Gaussian sigma for smoothing
        threshold_rel: Threshold as fraction of (peak - background) intensity
        min_distance: Minimum separation between detections (voxels)
        sigma_ratio: Unused (kept for API compatibility)

    Returns:
        Array of shape (N, 3) with (z, y, x) integer voxel coordinates
    """
    vol = volume.astype(np.float32)

    smoothed = gaussian_filter(vol, sigma=sigma)

    # Adaptive absolute threshold
    bg = np.percentile(smoothed, 50)
    peak = np.percentile(smoothed, 99.9)
    abs_thresh = bg + threshold_rel * (peak - bg)

    if abs_thresh <= bg or peak <= bg:
        return np.zeros((0, 3), dtype=np.int32)

    coords = peak_local_max(
        smoothed,
        min_distance=min_distance,
        threshold_abs=abs_thresh,
        exclude_border=2,
    )

    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    return coords.astype(np.int32)


def detect_cells_multiscale(
    volume: np.ndarray,
    sigmas: tuple = (2.0, 3.0, 4.0, 5.0, 6.0),
    threshold_rel: float = 0.02,
    min_distance: int = 5,
) -> np.ndarray:
    """
    Multi-scale detection: detects at multiple scales and merges.
    This handles cells of varying sizes more robustly.
    """
    all_coords = []
    for sigma in sigmas:
        coords = detect_cells_3d(
            volume, sigma=sigma, threshold_rel=threshold_rel,
            min_distance=min_distance,
        )
        if len(coords) > 0:
            all_coords.append(coords)

    if not all_coords:
        return np.zeros((0, 3), dtype=np.int32)

    merged = np.vstack(all_coords)
    if len(merged) > 1:
        merged = _deduplicate(merged, min_distance)
    return merged.astype(np.int32)


def _deduplicate(coords: np.ndarray, min_dist: float) -> np.ndarray:
    """Greedy deduplication: remove coords within min_dist of a kept coord."""
    keep = np.ones(len(coords), dtype=bool)
    for i in range(len(coords)):
        if not keep[i]:
            continue
        dist = np.sqrt(np.sum((coords[i+1:] - coords[i]) ** 2, axis=1))
        too_close = dist < min_dist
        keep[i+1:][too_close] = False
    return coords[keep]
