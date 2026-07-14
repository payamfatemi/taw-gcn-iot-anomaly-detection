# Paper-to-code mapping

| Article component | Implementation |
|---|---|
| Dataset definition, Eq. (1) | `data/base_adapter.py`, `preprocessing/pipeline.py` |
| Feature vector and matrix, Eqs. (2)–(3) | `preprocessing/feature_transformer.py` |
| Min-max normalization, Eq. (4) | `preprocessing/feature_transformer.py` |
| Temporal proximity, Eq. (5) | `graph/scores.py::temporal_score` |
| Communication dependency, Eq. (6) | `graph/scores.py::communication_score` |
| Behavioral similarity, Eq. (7) | `graph/scores.py::behavioral_score` |
| Weighted dependency, Eqs. (8)–(10) | `graph/scores.py`, `graph/builder.py` |
| Self-loops and normalization, Eqs. (11)–(13) | `graph/normalizer.py` |
| GCN propagation, Eqs. (14)–(16) | `models/layers.py`, `models/taw_gcn.py` |
| Logit and sigmoid, Eqs. (17)–(18) | `models/taw_gcn.py`, `evaluation/evaluator.py` |
| BCE objective, Eq. (19) | `training/trainer.py` |
| Decision threshold, Eq. (20) | `evaluation/metrics.py` |
| Adam optimization, Eq. (21) | `training/trainer.py` |
| Accuracy–F1, Eqs. (22)–(25) | `evaluation/metrics.py` |
| Pseudocode Table 2 | `pipeline.py` and package modules |
| Ablation Table 8 | `experiments/ablation.py` and `configs/experiments/ablation.yaml` |
