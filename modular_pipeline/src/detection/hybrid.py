"""Hybrid detection fusing DoG and UNet-based detections.

Combines classical blob detection with learned heatmaps via IoU-based
non-maximum suppression (NMS) to produce a consensus set of detections.
"""

from __future__ import annotations

import numpy as np

from .dog_detector import dog_blob_detect
from .unet_detector import UNetDetector


def _iou_distance(c1: np.ndarray, c2: np.ndarray, radius: float = 5.0) -> float:
    """Approximate 3D IoU distance between two centroids."""
    dist = np.linalg.norm(c1 - c2)
    if dist > 2 * radius:
        return 1.0  # No overlap
    # Approximate as sphere intersection
    d = max(dist, 1e-8)
    overlap_vol = (np.pi * (2 * radius - d) ** 2 * (4 * radius + d)) / (12 * radius ** 3)
    overlap_vol = max(0.0, min(1.0, overlap_vol))
    return 1.0 - overlap_vol


class HybridDetector:
    """Fuse detections from multiple sources with configurable NMS."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        min_confidence: float = 0.1,
        voxel_size: tuple[float, float, float] = (2.0, 0.5, 0.5),
        dog_config: dict | None = None,
        unet_config: dict | None = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        self.voxel_size = voxel_size
        self.dog_config = dog_config or {}
        self.unet_detector = None

        if unet_config and unet_config.get("model_path"):
            self.unet_detector = UNetDetector(**unet_config)
            self._unet_available = True
        else:
            self._unet_available = False

    def detect(self, volume: np.ndarray) -> np.ndarray:
        """Detect cells in a 3D volume.

        Returns:
            Array of shape (N, 3) with (z, y, x) centroids.
        """
        all_detections = []
        confidences = []

        # DoG detection
        if self.dog_config.get("enabled", True):
            dog_dets = dog_blob_detect(
                volume,
                min_sigma=self.dog_config.get("min_sigma", 1.5),
                max_sigma=self.dog_config.get("max_sigma", 4.0),
                sigma_ratio=self.dog_config.get("sigma_ratio", 1.6),
                threshold=self.dog_config.get("threshold", 0.01),
                overlap=self.dog_config.get("overlap", 0.5),
                voxel_size=self.voxel_size,
            )
            if len(dog_dets) > 0:
                all_detections.append(dog_dets)
                confidences.append(np.full(len(dog_dets), 0.8))

        # UNet detection
        if self._unet_available:
            unet_dets = self.unet_detector.detect(volume)
            if len(unet_dets) > 0:
                all_detections.append(unet_dets)
                confidences.append(np.full(len(unet_dets), 0.9))

        if not all_detections:
            return np.empty((0, 3), dtype=np.float32)

        dets = np.concatenate(all_detections, axis=0)
        confs = np.concatenate(confidences, axis=0)

        # Remove low confidence
        mask = confs >= self.min_confidence
        dets = dets[mask]
        confs = confs[mask]

        if len(dets) <= 1:
            return dets

        # IoU-based NMS
        return _nms_3d(dets, confs, self.iou_threshold, radius=5.0)


def _nms_3d(
    detections: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    radius: float = 5.0,
) -> np.ndarray:
    """NMS for 3D centroids using approximate IoU."""
    order = np.argsort(-scores)
    keep = []

    suppressed = np.zeros(len(detections), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            iou = 1.0 - _iou_distance(detections[i], detections[j], radius)
            if iou > iou_threshold:
                suppressed[j] = True

    return detections[np.array(keep)]
