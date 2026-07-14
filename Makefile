.PHONY: install test lint typecheck demo clean
install:
	python -m pip install -e '.[dev]'
test:
	pytest -q
lint:
	ruff check .
typecheck:
	mypy src
demo:
	python scripts/generate_demo_data.py --output data/raw/demo/demo.csv --rows 1200
	python scripts/run_pipeline.py --config configs/base.yaml --config configs/datasets/demo.yaml --raw data/raw/demo/demo.csv --run-name demo_smoke
clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in map(pathlib.Path, ['.pytest_cache','.mypy_cache','.ruff_cache','build','dist'])]"
