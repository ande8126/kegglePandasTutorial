# NBA game prediction: feature study

Snapshot: 5,452 completed imported games, seasons 2022–2026, latest game November 11, 2026. This study is about predicting the winner before each game with the information then available. The study left production loading, feature definitions, defaults, and existing notebooks unchanged. A subsequent configuration update made all 24 features the setup default; the historical `current_8` experiment reference remains fixed at its original eight inputs.

## Recommendation

**Use the existing full 24-feature set as the working feature set for the next random-forest models.** It had the highest average accuracy and virtually the lowest log loss. There is no evidence here that aggressive feature removal improves the model. My earlier conversational suggestion to pare the features down was a hypothesis; this experiment does not support it.

```python
from data.nba_setup import ALL_MODEL_FEATURES, setup_nba_experiment

experiment = setup_nba_experiment(
    validation_season=2025,
    model_features=ALL_MODEL_FEATURES,
)
```

This is a practical starting point, not a proven universally best combination. Its improvement over the current eight inputs is small, and per-season paired uncertainty intervals include zero. Keep the current eight as a compact benchmark. Do not switch features solely because one incomplete season favors them.

The recommended 24 inputs are these ten team statistics for **both** the home team and visitor (20 columns), plus four matchup differences:

| Team statistic | Reason to retain |
|---|---|
| `season_win_percentage` | Season results; somewhat redundant, but removing both columns did not improve accuracy. |
| `average_point_differential` | Long-run strength from winning/losing margins. |
| `points_scored_per_game` | Offensive scoring profile beyond net margin. |
| `points_allowed_per_game` | Defensive scoring profile beyond net margin. |
| `win_percentage_last_five_games` | Recent results; evaluate as part of the recent-form group. |
| `points_scored_last_five_games` | Recent offensive profile. |
| `points_allowed_last_five_games` | Recent defensive profile. |
| `record_at_home` | Venue-specific results. |
| `record_as_visitor` | Venue-specific results. |
| `rest_days_since_last_game` | Retained provisionally; low incremental signal here. |

Matchup differences: `season_point_margin_diff`, `last5_point_margin_diff`, `last5_win_pct_diff`, `rest_days_diff`. Difference columns give trees a direct team comparison. They deliberately overlap the individual-team inputs; redundancy alone did not make the larger feature set worse.

## Design: features, not a tuning search

- Train on 2022, validate on 2023; then train through 2023 and validate on 2024; then train through 2024 and validate on 2025.
- Compare **22 feature sets**, three seeds each (0, 42, 123), with identical eligible games in every comparison.
- Fixed random forest: 200 trees, minimum 10 games per leaf, all features considered at each split (`max_features=1.0`), other sklearn defaults. No hyperparameter grid search. This differs from the notebook's default forest; conclusions apply to this configuration. Using all split candidates avoids changing the number of candidate features indirectly when feature-set size changes.
- Compare accuracy, log loss, and Brier score. Lower log loss/Brier is better; they assess probabilities, not just selected winners. Log loss was the probability-quality ranking metric; accuracy is also reported, rather than choosing a different metric for each set.
- Average seeds within each season and then weight seasons equally. Seeds are repeated fits, not extra independent observations.
- Exclude 2026 from feature selection. Its 170 imported games leave just 92 eligible games after historical-feature requirements. It has already been inspected in prior notebook work, so it is not an untouched final test.
- Compute all history using earlier games only. Earlier validation-season results may inform later validation games, matching daily prediction. These are not preseason forecasts of an entire season.

| Validation season | Training games | Validation games | Always home | Current 8 accuracy | Full 24 accuracy | Full 24 log loss |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | 1,241 | 1,238 | 54.60% | 63.35% | 64.27% | 0.6354 |
| 2024 | 2,479 | 1,243 | 55.03% | 62.54% | 63.93% | 0.6370 |
| 2025 | 3,722 | 1,242 | 55.23% | 67.15% | 67.55% | 0.6059 |

The training-home-win-rate constant-probability baseline had log losses of 0.6913, 0.6884, and 0.6877 respectively. The model learns a useful matchup signal beyond home advantage.

## What mattered

Grouped permutation of the full model's validation inputs (five repeats, seed-42 model per season) gave:

| Feature family shuffled together | Average accuracy reduction | Average log-loss increase |
|---|---:|---:|
| Season strength: season margins, scoring, win percentages | 10.47 percentage points | 0.0733 |
| Recent form: last-five statistics and differences | 1.08 points | 0.0079 |
| Venue records | 0.55 points | 0.0035 |
| Rest | -0.06 points | 0.0003 |

Season strength is the main signal. Recent form and venue history add smaller contributions. Rest is weak here. These are model-dependence diagnostics, not causal effects or additive contributions; shuffling correlated groups can create unrealistic matchups. Individual impurity importance would obscure overlapping features, so grouped shuffling and refitted removal tests were both used.

Refitted removal tests support retaining scoring profiles and recent form:

| Feature set | Mean accuracy | Mean log loss |
|---|---:|---:|
| Full 24 | **65.25%** | 0.6261 |
| Remove season win percentages | 65.08% | **0.6260** |
| Remove rest | 65.16% | 0.6273 |
| Remove all venue records | 65.03% | 0.6290 |
| Remove recent form | 64.58% | 0.6310 |
| Remove raw points-scored/allowed averages | 64.04% | 0.6286 |
| Current 8 | 64.35% | 0.6298 |

Removing season win percentages produces a negligible 0.00008 log-loss gain and slightly lower accuracy. That is not a persuasive reason to remove those features. Rest is the first family I would reconsider if fewer inputs become a requirement.

## Additional fields tested

All are built from the existing ten imported columns, with no new database schema or dependencies:

| Candidate | Definition / outcome |
|---|---|
| Last-10 margin difference | Prior 10 games, minimum five; a three-input margin/rest/last-10 model achieved 64.96% accuracy and 0.6337 log loss. Promising simple alternative, but less effective probabilities than the full set. |
| Last-20 margin difference | Prior 20 games, minimum five; 64.23%, 0.6348. Did not improve the leading model. |
| Exponentially weighted margin | Shifted history, span 10, minimum five; 63.98%, 0.6383. No consistent advantage. |
| Season win-percentage difference | Added to season-margin/rest core: 64.51%, 0.6399. Useful representation, but not a replacement for the full inputs. |
| Smoothed venue records | Beta(2,2) prior: `(prior wins + 2)/(prior venue games + 4)`; 63.80%, 0.6457 when added as a difference to the core. Shrinkage was sensible in principle, but did not help this comparison. |
| Back-to-back / schedule congestion | Back-to-back indicator difference and difference in prior games within three calendar days; 64.34%, 0.6423 with core. Little added value. |
| Opponent-strength-adjusted margin | Average prior game margin plus that opponent's pregame season margin; zero for an opponent with no history. This is a simple proxy, not SRS. 64.24%, 0.6368 with core; no clear reason to add it. |
| Compact 7 / 9 | Win percentages, season/last-10 margins, rest difference, relevant venue records; nine adds separate rest days. 64.28% / 64.20%, with log losses 0.6299 / 0.6301. Not better than the full set. |

Most candidate additions were tested against the same two-input season-margin/rest core; that does **not** establish their incremental value when added to all 24 inputs. No new candidate earned promotion into production from this evidence. Their implementation stays isolated in `nba_feature_study.py` for further experiments.

Net rating uses points per 100 possessions; per-game margin does not adjust for pace ([NBA glossary](https://www.nba.com/stats/help/glossary)). This import has scores, dates, teams, and postseason status, so it cannot supply genuine possession-adjusted offense/defense, shooting efficiency, turnover/rebounding rates, or player availability. Those require richer historical, pregame-available sources; no benefit from those untested fields is claimed here.

## Data and transformation audit

Passed checks on the actual imported data:

- Unique game IDs; no raw nulls; positive scores; no ties or self-opponents; no duplicate team/date rows. Scores range 66–176; all 30 teams appear in each season.
- Exactly two team-perspective rows per game and no lost/duplicated games after feature merges.
- Target matches scores on the merged row. The notebooks assign the target from `nba_data`; that relies on matching indexes. The shared setup already computes it from `nba_model_data`, avoiding that alignment hazard.
- Season histories reset. Explicitly recalculated every row's prior wins, average margin, venue win/loss counts, last-five win rate and scoring/allowed averages.
- Changed a middle game's score separately in every season: its own and all earlier candidate inputs remained unchanged.
- Training/validation game IDs are disjoint. Filtering is identical across candidate sets.
- Existing four setup tests pass, including original-notebook feature parity, no input mutation, season boundaries, scheduled-game exclusion, and rejection of invalid inputs.

The common eligible counts are 1,241 / 1,238 / 1,243 / 1,242 / 92 for seasons 2022–2026. Early-season games are omitted because full five-game histories and venue histories are unavailable. These results apply to that eligible population; they do not measure opening-week performance. Longer windows use a minimum of five to preserve the same comparison population.

Regular-season and postseason games share histories, matching current setup behavior. Imported `postseason=False` counts are 1,236–1,237 per complete season; these are not exactly 1,230. The import alone cannot establish tournament/play-in classification or official schedule completeness. Team names/venue fields also cannot establish neutral-site status. The audit checks internal consistency, not independent verification of provider scores or coverage.

## How reliable is the feature ranking?

The full set improves on eight inputs by 0.92 / 1.39 / 0.40 accuracy percentage points across the three seasons. Paired date-block bootstrap 95% intervals are approximately -1.03 to +2.73 / -0.64 to +3.24 / -1.27 to +2.08 points. All include zero. Log-loss improvement intervals also include zero.

Bootstrap samples preserve same-day game groups, but not longer team/season dependence. Feature selection across 22 sets adds selection optimism; these intervals are exploratory and are not multiple-comparison adjusted. There are only three validation seasons. No claim of statistical superiority or permanently optimal features is warranted. The broader conclusion—season-long strength matters most—is considerably stronger than the exact ranking of closely matched feature lists.

Freeze a feature set and configuration before measuring future, untouched games. Do not repeatedly tune against 2026 and call it a final test. No estimator tuning is required to use this recommendation.

## Reproduction and artifacts

Run from the repository root with the existing database configuration:

```bash
.venv/bin/python -m experiments.nba_feature_study
.venv/bin/python -m unittest discover -s tests -v
```

`results/metrics.csv` contains every season/seed comparison; `summary.csv` contains equally weighted means; `paired_uncertainty.csv` contains paired intervals against the eight-input reference; `grouped_importance.csv` contains permutation diagnostics. `audit.json`, `coverage.csv`, and `metadata.json` record checks, feature definitions, coverage, package versions, a snapshot hash, and the fixed forest configuration. No credentials, raw database dump, or per-game prediction dump is committed.
