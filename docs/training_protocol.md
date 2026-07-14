# Training protocol

- Full-batch inductive training uses a train graph, validation graph, and test graph with no cross-split edges.
- The model outputs logits; `BCEWithLogitsLoss` provides a numerically stable implementation of sigmoid plus binary cross-entropy.
- Adam is used with configurable learning rate and weight decay.
- Optional balanced positive-class weighting is computed only from training labels.
- Validation F1 controls early stopping and best-checkpoint selection.
- The test graph is evaluated only after training and checkpoint selection.

For graphs that exceed full-batch memory, a validated graph-partition or neighbor-sampling backend must be added and reported explicitly; the software must not silently change the evaluation protocol.
