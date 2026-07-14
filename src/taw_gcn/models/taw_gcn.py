from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from taw_gcn.models.layers import WeightedGCNLayer


class TAWGCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dimensions: list[int],
        dropout: float = 0.0,
        activation: str = "relu",
        graph_propagation: bool = True,
    ) -> None:
        super().__init__()
        if not hidden_dimensions:
            raise ValueError("At least one hidden dimension is required.")
        self.graph_propagation = graph_propagation
        dimensions = [input_dim, *hidden_dimensions]
        if self.graph_propagation:
            self.representation_layers = nn.ModuleList(
                WeightedGCNLayer(dimensions[i], dimensions[i + 1])
                for i in range(len(dimensions) - 1)
            )
        else:
            self.representation_layers = nn.ModuleList(
                nn.Linear(dimensions[i], dimensions[i + 1])
                for i in range(len(dimensions) - 1)
            )
        activations: dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dimensions[-1], 1)

    @classmethod
    def from_config(cls, input_dim: int, config: dict[str, Any]) -> TAWGCN:
        model_cfg = config["model"]
        return cls(
            input_dim=input_dim,
            hidden_dimensions=[int(value) for value in model_cfg["hidden_dimensions"]],
            dropout=float(model_cfg.get("dropout", 0.0)),
            activation=str(model_cfg.get("activation", "relu")),
            graph_propagation=bool(config["graph"].get("graph_propagation", True)),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.representation_layers:
            if self.graph_propagation:
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        return cast(torch.Tensor, self.classifier(x).squeeze(-1))
