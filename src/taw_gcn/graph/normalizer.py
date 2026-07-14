from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def coalesce_max(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, n: int) -> sp.csr_matrix:
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    matrix.eliminate_zeros()
    return matrix


def symmetrize_max(matrix: sp.csr_matrix) -> sp.csr_matrix:
    return matrix.maximum(matrix.transpose()).tocsr()


def add_self_loops(matrix: sp.csr_matrix, weight: float = 1.0) -> sp.csr_matrix:
    n = matrix.shape[0]
    result = matrix + sp.identity(n, dtype=np.float32, format="csr") * float(weight)
    result.eliminate_zeros()
    return result


def symmetric_normalize(matrix: sp.csr_matrix, epsilon: float = 1e-12) -> sp.coo_matrix:
    degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inv_sqrt = np.power(np.maximum(degree, epsilon), -0.5)
    diagonal = sp.diags(inv_sqrt)
    normalized = diagonal @ matrix @ diagonal
    return normalized.tocoo()
