# Project architecture

The repository follows a layered research-software architecture:

1. **Adapters** isolate dataset-specific schemas.
2. **Preprocessing** performs cleaning, splitting, leakage checks, and train-only transformations.
3. **Graph construction** implements candidate selection and Equations 5–13.
4. **Model** implements weighted graph convolution and Equations 14–18.
5. **Training** implements binary cross-entropy optimization and Equation 21.
6. **Evaluation** implements Equations 22–25.
7. **Experiments** orchestrate standard, scalability, and internal ablation runs.
8. **Reporting** stores provenance, checksums, metrics, and predictions.

External baseline models are outside the scope of this repository.
