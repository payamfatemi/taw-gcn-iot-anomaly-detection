# Computational complexity

Let `N` be the number of records, `d` the feature dimension, `k` the behavioral neighbor count, and `E` the retained sparse edges.

- Dense pair enumeration is `O(N²)` and is prohibited.
- Temporal candidate generation is approximately `O(N log N + Nk_t)` after sorting.
- Communication candidates depend on group sizes and are bounded per value.
- Exact behavioral nearest-neighbor search can approach `O(N²d)`; use small `k`, subsampling, or an approved approximate backend for large datasets.
- A weighted GCN layer costs approximately `O(Ed + Nd h)` for sparse propagation and linear transformation.

Memory is dominated by node features and sparse edge tensors rather than an `N × N` adjacency matrix.
