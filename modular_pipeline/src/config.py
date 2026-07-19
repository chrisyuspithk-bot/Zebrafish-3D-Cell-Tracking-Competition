"""Configuration loader with sensible defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load pipeline configuration from YAML.

    Args:
        path: Path to config file. Uses default if not provided.

    Returns:
        Config dict with all sections.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return _default_config()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Fill in any missing sections with defaults
    defaults = _default_config()
    for section, values in defaults.items():
        if section not in config:
            config[section] = values
        elif isinstance(values, dict):
            for k, v in values.items():
                if k not in config[section]:
                    config[section][k] = v

    return config


def _default_config() -> dict[str, Any]:
    return {
        "data": {
            "train_dir": "/kaggle/input/biohub-cell-tracking-during-development/train",
            "test_dir": "/kaggle/input/biohub-cell-tracking-during-development/test",
            "voxel_size": [2.0, 0.5, 0.5],
            "max_edge_distance_um": 7.0,
        },
        "detection": {
            "dog": {"enabled": True, "min_sigma": 1.5, "max_sigma": 4.0,
                     "sigma_ratio": 1.6, "threshold": 0.01, "overlap": 0.5},
            "unet": {"enabled": False, "model_path": None, "in_channels": 1,
                      "out_channels": 1, "features": [32, 64, 128, 256],
                      "heatmap_threshold": 0.3, "peak_min_distance": 4},
            "fusion": {"iou_threshold": 0.5, "min_confidence": 0.1},
        },
        "tracking": {
            "graph": {"max_linking_distance": 50, "max_gap_frames": 4,
                      "motion_model": "constant_velocity",
                      "appearance_weight": 0.3, "distance_weight": 0.7},
            "ilp": {"enabled": True, "assignment_cost_scale": 1.0,
                     "appearance_gate": 0.5, "division_cost": 100.0,
                     "gap_closing_cost": 50.0},
            "gnn": {"enabled": True, "hidden_dim": 64, "num_layers": 3,
                     "dropout": 0.1},
        },
        "division": {
            "enabled": True, "min_cell_distance": 3, "max_cell_distance": 30,
            "temporal_window": 5,
        },
        "postprocess": {
            "gap_repair": {"enabled": True, "max_gap": 4,
                            "max_displacement": 30, "min_track_length": 3},
        },
        "output": {"submission_dir": "/workspace/project/submissions", "debug": False},
    }
