# Split protocol

The default split is 70% train, 15% validation, and 15% test. The preferred strategy is grouped splitting by capture/scenario, selected through deterministic repeated group splits to reduce class-distribution error. When valid groups are unavailable, the implementation falls back to stratified random splitting. Time-ordered splitting is also supported.

The leakage checker verifies disjoint record IDs. Group overlap is reported. A fitted preprocessing transformer is created exclusively from training records.
