"""GNN-based track linker for learning association scores.

Uses a lightweight message-passing network over the tracking graph
to predict edge probabilities, trained with margin-based ranking loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeScorer(nn.Module):
    """GNN edge scorer for tracking graph edges.

    Takes source and target node features (position, appearance embedding),
    concatenates them, and scores the edge with an MLP.
    """

    def __init__(
        self,
        input_dim: int = 7,   # 3 src pos + 3 dst pos + 1 time diff
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(prev, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GNNTracker:
    """GNN-based tracker for learned association."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = EdgeScorer(
            input_dim=7, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout,
        ).to(self.device)
        self.model.eval()

    def score_edges(
        self,
        src_positions: torch.Tensor,   # (E, 3)
        dst_positions: torch.Tensor,   # (E, 3)
        time_diffs: torch.Tensor,      # (E,)
    ) -> torch.Tensor:
        """Score candidate edges.

        Args:
            src_positions: (E, 3) source centroids.
            dst_positions: (E, 3) target centroids.
            time_diffs: (E,) time differences between frames.

        Returns:
            (E,) edge scores.
        """
        if src_positions.numel() == 0:
            return torch.empty(0, device=self.device)

        # Normalize positions
        src_norm = (src_positions - src_positions.mean(dim=0)) / (src_positions.std(dim=0) + 1e-8)
        dst_norm = (dst_positions - dst_positions.mean(dim=0)) / (dst_positions.std(dim=0) + 1e-8)
        t_norm = time_diffs.float() / 10.0

        features = torch.cat([src_norm, dst_norm, t_norm.unsqueeze(-1)], dim=-1)

        with torch.no_grad():
            scores = self.model(features)
        return scores

    @torch.no_grad()
    def link(
        self,
        graph: dict,
        score_threshold: float = 0.5,
    ) -> list[tuple[int, int, float]]:
        """Score and filter edges from the tracking graph.

        Returns:
            List of (src_node, dst_node, score) pairs above threshold.
        """
        move_edges = graph["move_edges"]
        gap_edges = graph["gap_edges"]
        positions = torch.as_tensor(graph["positions"], device=self.device)

        all_edges = []

        for edge_list in [move_edges, gap_edges]:
            if not edge_list:
                continue
            src_idx = torch.tensor([e[0] for e in edge_list])
            dst_idx = torch.tensor([e[1] for e in edge_list])
            time_diffs = torch.tensor(
                [1.0 if len(e) < 4 else float(e[3]) for e in edge_list]
            )

            scores = self.score_edges(
                positions[src_idx],
                positions[dst_idx],
                time_diffs,
            )

            mask = scores > score_threshold
            for i in mask.nonzero(as_tuple=True)[0]:
                all_edges.append((int(src_idx[i]), int(dst_idx[i]), float(scores[i])))

        return sorted(all_edges, key=lambda x: -x[2])
