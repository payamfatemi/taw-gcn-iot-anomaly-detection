# Experiment protocol

The repository supports:

- standard training/evaluation;
- data-volume subsampling with explicit skip status when insufficient unique records exist;
- network-scale selection by unique device identifier;
- measured inference latency without artificial sleep;
- internal component ablation.

External baseline code and hard-coded article results are excluded. Every run must use the same split, seed, and evaluation definition for internal comparisons.
