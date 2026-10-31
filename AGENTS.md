# Repository Guidelines

## Project Structure & Module Organization

- `src/data/` contains dataset contracts, ingestion, preprocessing, and optional YouTube comment retrieval.
- `src/models/` contains the shared traditional-model factory and PyTorch MLP.
- `src/training/` contains training, evaluation, and model-comparison workflows.
- `flask_app/` provides local inference and visualization endpoints; `dashboard/` contains the Streamlit UI.
- `tests/` contains `unittest` coverage for preprocessing, models, API behavior, and YouTube retrieval.
- `models/`, `reports/`, `mlruns/`, `data/`, and `.dvc/` contain generated or local artifacts; do not commit ignored outputs.
- `src/model/` is legacy compatibility code. Prefer the active `src/training/` workflow unless a change explicitly targets legacy behavior.

## Build, Test, and Development Commands

Run from the repository root with the project virtual environment active:

```powershell
python src/data/data_ingestion.py
python src/data/data_preprocessing.py
python -m src.training.compare --models naive_bayes logistic_regression linear_svm lightgbm mlp
python -m unittest discover -s tests -v
python -m compileall -q src flask_app dashboard tests
streamlit run dashboard/app.py
python flask_app/app.py
```

Use `dvc status` or `dvc repro --dry` to inspect pipeline state. Regenerate `dvc.lock` only after an intentional, successful pipeline validation.

## Coding Style & Naming Conventions

Use Python 3.10+ with four-space indentation, type hints for public interfaces, `snake_case` functions/variables, and `PascalCase` classes. Keep paths repository-relative through `pathlib`, preserve shared preprocessing between training and inference, and use `apply_patch` for focused edits. Ruff configuration is defined in `pyproject.toml`; run `ruff check` and `ruff format --check` when available.

## Testing Guidelines

Add tests under `tests/` using `unittest`, with files named `test_<area>.py` and methods named `test_<behavior>`. Mock external YouTube responses and avoid live API calls in tests. Cover schema validation, deterministic splits, model save/load, API validation, and local-only configuration.

## Commits, Pull Requests & Safety

History is sparse (`initial`, `Full pipeline`), so use concise imperative commit subjects such as `Add model calibration report`. PRs should describe the motivation, affected pipeline stages, verification commands, metric changes, and UI screenshots when applicable.

Do not commit credentials or modify `.env` with secrets. Obtain explicit approval before destructive operations, dependency changes, cloud or billable services, deployment, major architecture changes, or work outside the requested scope. Inspect relevant code first and preserve unrelated user changes.
