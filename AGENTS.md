# Repository Guidelines

## Project Structure & Module Organization
This repository follows Kaggle's pandas and machine-learning lessons using NBA game data from PostgreSQL.
- `tutorial.py`: entry-point script that loads games and prints a preview and dimensions.
- `data/nbaDataImport.py`: shared `load_nba_games()` loader and SQL query against `bdl_nba_games`.
- `exploration.ipynb`: pandas basics and initial data inspection.
- `first_model.ipynb`: initial decision-tree classification pipeline.
- `first_model_detailed.ipynb`: historical feature engineering, season-based validation, and model evaluation.
- `requirements.txt`: currently empty; record dependencies here when changing the environment.

There is no dedicated test directory or asset directory. Keep reusable loading code in `data/` and experiments in clearly named notebooks.

## Build, Test, and Development Commands
Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy jupyter sqlalchemy python-dotenv psycopg2-binary scikit-learn
python tutorial.py
jupyter notebook
```

These commands create the environment, install the libraries used by the scripts and notebooks, run the loader smoke check, and launch Jupyter. Database-backed execution requires a configured PostgreSQL connection. There is no build step or configured lint/format command.

## Coding Style & Naming Conventions
Use four-space indentation, descriptive `snake_case` variables and functions, and grouped imports. Preserve the existing `data.nbaDataImport` import path unless updating all callers. Prefer explicit DataFrame column selections and `.copy()` before modifying subsets. Explain transformations in short notebook Markdown cells or comments; move shared logic into Python modules.

## Testing Guidelines
No automated test framework or coverage threshold is configured. For changes, run `python tutorial.py` when database access is available, then restart the affected notebook's kernel and run every cell in order. Check row counts, null handling, merge cardinality, and disjoint training/validation game IDs. Compute historical features with shifted results to prevent target leakage, and compare validation accuracy against the always-home-win baseline. Use fixed random seeds for reproducible model comparisons.

## Commit & Pull Request Guidelines
History uses short, descriptive subjects such as `Add first model workbook`; follow that style and keep commits focused. In pull requests, describe the change, affected notebooks, validation performed, and any dependency or schema requirements. Link related issues when applicable and report model metrics when behavior changes.

## Security & Configuration
Set `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, and `PGPASSWORD` in a local `.env`. Never commit credentials or sensitive notebook outputs. Keep `.env`, `.venv/`, and `.ipynb_checkpoints/` ignored.

## Resources
- https://www.kaggle.com/learn/intro-to-machine-learning
- https://www.kaggle.com/code/dansbecker/how-models-work
- https://www.kaggle.com/code/dansbecker/basic-data-exploration
- https://www.kaggle.com/code/dansbecker/your-first-machine-learning-model
- https://www.kaggle.com/code/dansbecker/model-validation
- https://www.kaggle.com/code/dansbecker/random-forests
- https://www.kaggle.com/code/alexisbcook/machine-learning-competitions
- https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting

