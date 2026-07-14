# Contributing

1. Create a focused branch.
2. Add or update tests for every behavioral change.
3. Run `pytest -q`, `ruff check .`, and `mypy src`.
4. Do not add external baseline implementations to the core package.
5. Do not commit raw datasets, trained models, or run outputs.
6. Preserve paper-to-code traceability in `docs/paper_to_code_mapping.md`.
