from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn


class WeightedGCNLayer(nn.Module):
    """H^(l+1) = A_hat H^(l) W^(l) + b^(l)."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = nn.Parameter(torch.empty(output_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(max(self.weight.shape[1], 1))
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        support = x @ self.weight
        adjacency = torch.sparse_coo_tensor(
            edge_index,
            edge_weight,
            size=(x.shape[0], x.shape[0]),
            device=x.device,
            check_invariants=False,
        ).coalesce()
        output = torch.sparse.mm(adjacency, support)
        if self.bias is not None:
            output = output + self.bias
        return cast(torch.Tensor, output)
