# AGENTS.md
# Guidance for agentic coding in this repo (Project Chocolate)

## Repo Snapshot
- Language: Python 3.8+
- Core libs: torch, torch-geometric, pandas, numpy, scikit-learn
- Domain: temporal graph modeling of congressional trades
- Data: proprietary, stored under `data/` (gitignored)

## Setup / Build / Install
- Conda env (preferred):
  ```bash
  conda env create -f environment.yml
  conda activate chocolate
  ```
- Pip install:
  ```bash
  python -m pip install -r requirements.txt
  python -m pip install -e .
  ```
- Sanity check:
  ```bash
  python -c "import src; print('Installation successful!')"
  ```

## Data Pipeline (Large / Slow)
- Build dataset from raw:
  ```bash
  python scripts/build_dataset.py
  ```
- Required input: `data/raw/v5_transactions.csv`
- Generates: parquet files, processed CSVs, and `data/price_sequences.pt`

## Build Graph / Prepare Temporal Data
- Build temporal graph:
  ```bash
  python src/temporal_data.py
  ```
- Output: `data/temporal_data.pt`

## Training / Evaluation Commands
- Rolling window (TGN):
  ```bash
  python scripts/run_rolling.py --horizon 1M --alpha 0.0
  ```
- Ablation study (full-only):
  ```bash
  python scripts/run_ablation.py --full-only --horizon 6M --alpha 0.05
  ```
- Full ablation run (long):
  ```bash
  python scripts/run_ablation.py --full-run
  ```

## Tests
- Unit tests (default):
  ```bash
  pytest tests/ -v
  ```
- Run a single test file:
  ```bash
  pytest tests/test_file.py -v
  ```
- Run a single test function:
  ```bash
  pytest tests/test_file.py::test_name -v
  ```
- Run a single test class/method:
  ```bash
  pytest tests/test_file.py::TestClass::test_name -v
  ```
- End-to-end pipeline (assumes data exists):
  ```bash
  bash scripts/test_pipeline.sh
  ```
- Full pipeline (rebuilds data, very slow):
  ```bash
  bash scripts/test_full_pipeline.sh
  ```

## Lint / Format
- No enforced formatter/linter configured in repo.
- `black`/`flake8` are listed as optional in `requirements.txt` but not wired.
- If you format, keep changes minimal and consistent with existing style.

## Configuration / Paths
- Path config lives in `src/config.py`.
- `CHOCOLATE_PROJECT_ROOT` env var can override project root.
- Outputs default to `results/` and `logs/` (created if missing).

## Coding Style Guidelines

### Imports
- Order: standard library, third-party, local (`src.`) imports.
- Prefer absolute package imports (`from src...`) over `sys.path.append`.
- Avoid wildcard imports.

### Formatting
- Follow PEP 8 (4-space indentation, 79-99 char lines where possible).
- Keep function bodies short and focused.
- Use blank lines to separate logical sections.

### Types
- Type hints are encouraged for public functions and complex interfaces.
- If adding typing, keep it consistent (don’t half-type a file).

### Naming
- Modules/files: snake_case
- Functions/vars: snake_case
- Classes: CamelCase
- Constants: UPPER_SNAKE_CASE

### Docstrings
- Google-style docstrings are preferred (see `CONTRIBUTING.md`).
- Add docstrings for public functions or non-obvious logic.

### Error Handling
- Prefer explicit exceptions over silent failures.
- Avoid bare `except:` in new code; catch specific exceptions.
- Log errors with context using `logging` where appropriate.

### Logging
- Prefer `logging` over `print` for long-running scripts.
- Use module-level loggers and avoid duplicate handlers.

### Data / Reproducibility
- Data files are proprietary and gitignored; never commit raw data.
- When creating outputs, ensure directories exist (`os.makedirs(..., exist_ok=True)`).
- If adding randomness, set seeds in a central, explicit place.

### Model / Training Conventions
- Temporal splits use monthly retraining with a one-month gap.
- Labels are dynamically computed by horizon in training/eval code.
- Keep “gap” handling consistent between baselines and TGN.

## Where Things Live
- Core model: `src/models_tgn.py`
- Temporal graph builder: `src/temporal_data.py`
- Config and paths: `src/config.py`
- Training scripts: `scripts/run_rolling.py`, `scripts/run_ablation.py`
- Results/logs: `results/`, `logs/`, `ablation_study/logs/`

## Notes for Agents
- Don’t assume data exists; many scripts require `data/` assets.
- Avoid destructive git commands; keep changes scoped.
- If adding new experiments, update the experiment log in `docs/`.
