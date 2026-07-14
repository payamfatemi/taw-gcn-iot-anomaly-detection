# Known limitations

1. Raw datasets are not included.
2. Dataset releases may use different column names; aliases must be verified.
3. Several article hyperparameters are unspecified and therefore represented as documented implementation defaults.
4. The reference trainer is full-batch and suitable for research-scale graphs that fit memory. Very large multi-million-record experiments require an explicitly validated sampling or partitioning backend and sufficient hardware.
5. Reported article figures are not embedded as target values; results are computed from actual predictions.
6. This research implementation is not a production intrusion-prevention system.
