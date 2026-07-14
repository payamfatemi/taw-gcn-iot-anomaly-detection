# Sparse topology-aware graph construction

A dense all-pairs graph is not constructed. Candidate pairs are the union of:

- temporally nearby records within a configured window;
- records sharing enabled communication attributes;
- behavioral top-k nearest neighbors under cosine distance.

Candidates are limited per node, scored, thresholded, symmetrized, augmented with self-loops, and symmetrically normalized. Sparse COO tensors are stored as `edge_index` and `edge_weight`.

The configuration validator enforces non-negative dependency weights summing to one. Ablation variants can disable individual dependency components and optionally renormalize the remaining coefficients.
