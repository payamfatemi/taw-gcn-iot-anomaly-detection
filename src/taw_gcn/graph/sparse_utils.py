from __future__ import annotations

import scipy.sparse as sp
import torch


def scipy_to_torch_coo(matrix: sp.spmatrix) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()
