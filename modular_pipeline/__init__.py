"""Modular cell tracking pipeline — alternative to the LAP-based approach.

Key differences from the baseline (kaggle_solution.py):
- Graph-based tracking with ILP global optimization (not greedy frame-to-frame LAP)
- Hybrid detection: DoG + optional 3D UNet center heatmaps
- Dedicated gap repair with centroid interpolation
- .geff output format support
- Full competition metrics module (edge Jaccard + division Jaccard)
- YAML-driven configuration
"""
