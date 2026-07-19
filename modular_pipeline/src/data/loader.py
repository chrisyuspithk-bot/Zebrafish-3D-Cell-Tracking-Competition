"""Chunked Zarr data loader for 3D+t microscopy volumes.

Handles large Zarr arrays with shape (T, Z, Y, X) using lazy loading
and chunked processing to stay within memory constraints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import zarr


class ZarrVolume:
    """Lazy-loaded 3D+t Zarr volume with chunked access patterns."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.root = zarr.open(self.path, mode="r")
        # Zarr v2 uses group keys like "0"; Zarr v3 arrays are direct
        if isinstance(self.root, zarr.Array):
            self.data = self.root
        else:
            keys = list(self.root.keys())
            self.data = self.root[keys[0]] if keys else self.root
        self.shape = self.data.shape  # (T, Z, Y, X)
        self.dtype = self.data.dtype
        self._validate_shape()

    def _validate_shape(self) -> None:
        if len(self.shape) != 4:
            msg = f"Expected 4D array (T, Z, Y, X), got shape {self.shape}"
            raise ValueError(msg)

    @property
    def T(self) -> int:
        return self.shape[0]

    @property
    def Z(self) -> int:
        return self.shape[1]

    @property
    def Y(self) -> int:
        return self.shape[2]

    @property
    def X(self) -> int:
        return self.shape[3]

    def read_timepoint(self, t: int) -> np.ndarray:
        """Read a single timepoint as (Z, Y, X) array."""
        return np.asarray(self.data[t])

    def read_timepoints(self, start: int, end: int) -> np.ndarray:
        """Read a range of timepoints as (T_slice, Z, Y, X) array."""
        return np.asarray(self.data[start:end])

    def read_chunk(self, t: int, z_slice: slice, y_slice: slice, x_slice: slice) -> np.ndarray:
        """Read a spatial chunk at a specific timepoint."""
        return np.asarray(self.data[t, z_slice, y_slice, x_slice])

    def sliding_window(
        self,
        window_t: int = 3,
        spatial_chunks: tuple[int, int, int] = (16, 128, 128),
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (t_center, volume_window) for sliding window processing.

        Args:
            window_t: Temporal window radius (yields 2*window_t+1 frames).
            spatial_chunks: (Z, Y, X) chunk sizes for spatial tiling.

        Yields:
            (t_center, numpy array of shape (2*window_t+1, Z, Y, X))
        """
        T = self.T
        for t in range(T):
            t_start = max(0, t - window_t)
            t_end = min(T, t + window_t + 1)
            pad_before = t - t_start
            pad_after = (t + window_t + 1) - t_end

            block = self.read_timepoints(t_start, t_end)
            if pad_before > 0 or pad_after > 0:
                block = np.pad(block, ((pad_before, pad_after), (0, 0), (0, 0), (0, 0)),
                               mode="reflect")

            yield t, block

    def iter_timepoints(self, start: int = 0, end: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (t, volume) for each timepoint."""
        end = end or self.T
        for t in range(start, end):
            yield t, self.read_timepoint(t)

    def __repr__(self) -> str:
        return f"ZarrVolume(path={self.path!r}, shape={self.shape}, dtype={self.dtype})"


def find_zarr_volumes(data_dir: str | Path) -> list[Path]:
    """Find all .zarr directories under data_dir."""
    data_dir = Path(data_dir)
    return sorted(data_dir.rglob("*.zarr"))


def find_geff_files(data_dir: str | Path) -> list[Path]:
    """Find all .geff annotation files under data_dir."""
    data_dir = Path(data_dir)
    return sorted(data_dir.rglob("*.geff"))
