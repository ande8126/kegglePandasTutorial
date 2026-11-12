# kagglePandasTutorial
Exploratory work as I follow along with the Basic Data Exploration tutorial on kaggle.com, lol mispelled the name of the website in the repo title.

Link: https://www.kaggle.com/learn/intro-to-machine-learning


# instructions
From terminal run:

```
source .venv/bin/activate
```

then

```
jupyter lab
```

## Reuse the notebook setup

From a notebook in the repository root:

```python
import pandas as pd
from data.nba_setup import setup_nba_experiment

# Train on all available seasons before 2025; validate on 2025.
experiment = setup_nba_experiment(validation_season=2025)

X_train = experiment.X_train
y_train = experiment.y_train
X_validation = experiment.X_validation
y_validation = experiment.y_validation

nba_model_data = experiment.nba_model_data
team_games = experiment.team_games
```

The setup loads PostgreSQL data using the existing `.env` configuration and
builds all features from `random_forest.ipynb`: season and last-five-game
statistics, rest days, home/visitor records, and matchup differences. History
resets by season and excludes the current game's result. Earlier games within
the validation season contribute to later games' features.

The default model inputs are the eight currently selected in that notebook.
All generated features remain available in `experiment.nba_model_data`, including
rows with missing early-season history. `experiment.model_data` drops rows missing
the selected inputs. Training and validation are copies of that filtered table;
seasons after the validation season are excluded from both splits.

Choose fewer inputs in the setup call:

```python
experiment = setup_nba_experiment(
    validation_season=2025,
    model_features=[
        "season_point_margin_diff",
        "last5_point_margin_diff",
        "rest_days_diff",
    ],
)
```

Or use every generated model input:

```python
from data.nba_setup import ALL_MODEL_FEATURES

experiment = setup_nba_experiment(
    validation_season=2025,
    model_features=ALL_MODEL_FEATURES,
)
```

The result also exposes `home`, `away`, `team_features`, `home_features`,
`visitor_features`, `train`, `validation`, `features` (team feature names),
`nba_model_features` (selected model inputs), and `all_model_features`.
Pass `nba_data=existing_dataframe` to reuse loaded games without another query.
Changing selected inputs can change which rows survive missing-value filtering;
use the same game IDs when comparing feature sets. Regular-season and postseason
games are included together, matching the existing notebooks.

See `experiment_setup.ipynb` for a ready-to-run starting notebook.
