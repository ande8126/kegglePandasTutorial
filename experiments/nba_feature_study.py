"""Reproducible feature comparison; production setup and models are untouched.

Run from the repository root: python -m experiments.nba_feature_study
Results contain aggregate metrics only, never connection details or raw games.
"""
import argparse
import contextlib
import io
import hashlib

import sklearn
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from data.nba_setup import ALL_MODEL_FEATURES, setup_nba_experiment


# Freeze the historical eight-input reference independently of setup defaults.
BASELINE_MODEL_FEATURES = (
    "home_season_win_percentage",
    "visitor_season_win_percentage",
    "season_point_margin_diff",
    "last5_point_margin_diff",
    "rest_days_diff",
    "last5_win_pct_diff",
    "home_record_at_home",
    "visitor_record_as_visitor",
)

CORE = ["season_point_margin_diff", "rest_days_diff"]


def candidate_data(raw):
    """All inputs are known before tipoff; no final season ratings are used."""
    setup = setup_nba_experiment(nba_data=raw, validation_season=2025)
    tg = setup.team_games.copy()
    groups = tg.groupby(["team_id", "season"])
    added = []
    for window in [10, 20]:
        name = f"margin_last_{window}"
        tg[name] = groups.point_differential.transform(
            lambda s: s.shift(1).rolling(window, min_periods=5).mean()
        )
        added.append(name)
    tg["margin_ewm10"] = groups.point_differential.transform(
        lambda s: s.shift(1).ewm(span=10, adjust=False, min_periods=5).mean()
    )
    tg["games_before"] = groups.cumcount()
    tg["back_to_back"] = tg.rest_days_since_last_game.eq(1).astype(int)
    # Count prior games within the preceding three calendar days (3-in-4 load).
    tg["games_previous_3_days"] = 0
    for _, indices in groups.groups.items():
        dates = tg.loc[indices, "date"].to_numpy()
        counts = np.arange(len(dates)) - np.searchsorted(
            dates, dates - pd.Timedelta(days=3), side="left"
        )
        tg.loc[indices, "games_previous_3_days"] = counts
    # Beta(2,2) prior: shrinks small venue samples toward a 50% record.
    tg["smoothed_home_record"] = (tg.home_wins_before + 2) / (
        tg.home_wins_before + tg.home_losses_before + 4
    )
    tg["smoothed_visitor_record"] = (tg.visitor_wins_before + 2) / (
        tg.visitor_wins_before + tg.visitor_losses_before + 4
    )
    added += ["margin_ewm10", "games_before", "back_to_back", "games_previous_3_days",
              "smoothed_home_record", "smoothed_visitor_record"]
    # Opponent's pregame strength, not their eventual end-of-season strength.
    opponents = tg[["id", "team_id", "average_point_differential"]].rename(
        columns={"team_id": "opponent_id", "average_point_differential": "opponent_margin"}
    )
    paired = tg[["id", "team_id"]].merge(opponents, on="id")
    paired = paired.loc[paired.team_id.ne(paired.opponent_id)]
    tg = tg.merge(paired[["id", "team_id", "opponent_margin"]],
                  on=["id", "team_id"], validate="one_to_one")
    tg = tg.sort_values(["team_id", "season", "date", "id"]).reset_index(drop=True)
    tg["adjusted_margin_result"] = tg.point_differential + tg.opponent_margin.fillna(0)
    tg["schedule_adjusted_margin"] = tg.groupby(["team_id", "season"]).adjusted_margin_result.transform(
        lambda s: s.shift(1).expanding().mean()
    )
    added += ["schedule_adjusted_margin"]
    frame = setup.nba_model_data.copy()
    for side in ["home", "visitor"]:
        table = tg[["id", "team_id"] + added].rename(columns={
            "team_id": f"{side}_team_id", **{c: f"{side}_{c}" for c in added}
        })
        frame = frame.merge(table, on=["id", f"{side}_team_id"], validate="one_to_one")
    for name in ["margin_last_10", "margin_last_20", "margin_ewm10", "back_to_back",
                 "games_previous_3_days", "schedule_adjusted_margin"]:
        frame[f"{name}_diff"] = frame[f"home_{name}"] - frame[f"visitor_{name}"]
    frame["season_win_pct_diff"] = frame.home_season_win_percentage - frame.visitor_season_win_percentage
    frame["venue_record_diff"] = frame.home_record_at_home - frame.visitor_record_as_visitor
    frame["smoothed_venue_diff"] = frame.home_smoothed_home_record - frame.visitor_smoothed_visitor_record
    return setup, frame


def feature_sets():
    sets = {
        "margin_only": ["season_point_margin_diff"],
        "core_margin_rest": CORE,
        "current_8": list(BASELINE_MODEL_FEATURES),
        "full_24": list(ALL_MODEL_FEATURES),
        "core_plus_last5_margin": CORE + ["last5_point_margin_diff"],
        "core_plus_last10_margin": CORE + ["margin_last_10_diff"],
        "core_plus_last20_margin": CORE + ["margin_last_20_diff"],
        "core_plus_ewm_margin": CORE + ["margin_ewm10_diff"],
        "core_plus_win_pct": CORE + ["season_win_pct_diff"],
        "core_plus_last5_wins": CORE + ["last5_win_pct_diff"],
        "core_plus_venue": CORE + ["home_record_at_home", "visitor_record_as_visitor"],
        "core_plus_smoothed_venue": CORE + ["smoothed_venue_diff"],
        "core_plus_schedule": CORE + ["back_to_back_diff", "games_previous_3_days_diff"],
        "core_plus_adjusted_margin": CORE + ["schedule_adjusted_margin_diff"],
        "balanced_5": CORE + ["season_win_pct_diff", "margin_last_10_diff", "smoothed_venue_diff"],
    }

    full = list(ALL_MODEL_FEATURES)
    groups = {
        "without_recent_form": [c for c in full if "last_five" in c or "last5" in c],
        "without_venue": [c for c in full if "record_" in c],
        "without_rest": [c for c in full if "rest_days" in c],
        "without_season_win_pct": [c for c in full if "season_win_percentage" in c],
        "without_raw_scoring": [c for c in full if "points_scored" in c or "points_allowed" in c],
    }
    for name, excluded in groups.items():
        sets[name] = [c for c in full if c not in excluded]
    sets["compact_7"] = ["home_season_win_percentage", "visitor_season_win_percentage",
        "season_point_margin_diff", "margin_last_10_diff", "rest_days_diff",
        "home_record_at_home", "visitor_record_as_visitor"]
    sets["compact_9"] = sets["compact_7"] + ["home_rest_days_since_last_game", "visitor_rest_days_since_last_game"]
    return sets



def audit(raw, setup, frame):
    tg = setup.team_games
    checks = {
        "unique_game_ids": not raw.id.duplicated().any(),
        "no_raw_nulls": not raw.isna().any().any(),
        "no_ties": not raw.home_team_score.eq(raw.visitor_team_score).any(),
        "positive_scores": bool((raw[["home_team_score", "visitor_team_score"]] > 0).all().all()),
        "distinct_opponents": not raw.home_team_id.eq(raw.visitor_team_id).any(),
        "unique_team_date": not tg.duplicated(["team_id", "date"]).any(),
        "two_team_rows_per_game": len(tg) == 2 * len(raw),
        "merge_preserves_games": len(frame) == len(raw),
        "target_matches_merged_scores": bool(frame.home_win.eq(frame.home_team_score.gt(frame.visitor_team_score).astype(int)).all()),
    }
    first = tg.groupby(["team_id", "season"]).head(1)
    checks["season_history_resets"] = bool(first.season_win_percentage.isna().all() and first.home_wins_before.eq(0).all() and first.visitor_wins_before.eq(0).all())
    # Independent explicit prior-game calculations for every team-season row.
    for _, games in tg.groupby(["team_id", "season"]):
        for position in range(len(games)):
            game, prior = games.iloc[position], games.iloc[:position]
            if position:
                assert np.isclose(game.season_win_percentage, prior.won.mean())
                assert np.isclose(game.average_point_differential, prior.point_differential.mean())
            for venue, is_home in [("home", True), ("visitor", False)]:
                history = prior.loc[prior.is_home.eq(is_home)]
                assert game[f"{venue}_wins_before"] == history.won.sum()
                assert game[f"{venue}_losses_before"] == len(history) - history.won.sum()
            if position >= 5:
                assert np.isclose(game.win_percentage_last_five_games, prior.won.tail(5).mean())
                assert np.isclose(game.points_scored_last_five_games, prior.points_for.tail(5).mean())
                assert np.isclose(game.points_allowed_last_five_games, prior.points_against.tail(5).mean())
    checks["explicit_pregame_history_checks"] = True
    # Mutate each season's middle game. Its own and all earlier inputs must be unchanged.
    columns = sorted(set(sum(feature_sets().values(), [])))
    changed = raw.copy()
    thresholds = {}
    for season, games in raw.groupby("season"):
        row = games.sort_values(["game_date", "id"]).iloc[len(games)//2]
        thresholds[season] = pd.Timestamp(row.game_date, tz="UTC")
        changed.loc[changed.id.eq(row.id), "home_team_score"] += 37
    _, other = candidate_data(changed)
    mask = pd.Series(False, index=frame.index)
    for season, date in thresholds.items():
        mask |= frame.season.eq(season) & frame.date.le(date)
    pd.testing.assert_frame_equal(frame.loc[mask, columns], other.loc[mask, columns])
    checks["score_perturbation_no_current_or_future_leakage"] = True
    assert all(checks.values()), checks
    return checks


def paired_uncertainty(predictions, out):
    """Paired date-block bootstrap; seeds are repeats, not independent games."""
    data = pd.DataFrame(predictions).copy()
    data["correct"] = (data.probability.gt(.5) == data.actual).astype(float)
    p = data.probability.clip(1e-12, 1 - 1e-12)
    data["loss"] = -(data.actual * np.log(p) + (1 - data.actual) * np.log(1 - p))
    # Average per-game metrics across seeds before resampling game dates.
    games = data.groupby(["season", "features", "id", "date"])[["correct", "loss"]].mean().reset_index()
    rows = []
    for season in [2023, 2024, 2025]:
        subset = games.loc[games.season.eq(season)]
        ref = subset.loc[subset.features.eq("current_8")]
        for name in subset.features.unique():
            paired = subset.loc[subset.features.eq(name)].merge(ref, on=["id", "date"], suffixes=("", "_reference"))
            paired["accuracy_gain"] = paired.correct - paired.correct_reference
            paired["log_loss_gain"] = paired.loss_reference - paired.loss
            daily = paired.groupby("date").agg(n=("id", "size"), accuracy_gain=("accuracy_gain", "sum"), log_loss_gain=("log_loss_gain", "sum"))
            rng = np.random.default_rng(42)
            indices = rng.integers(len(daily), size=(2000, len(daily)))
            weights = daily.n.to_numpy()[indices].sum(axis=1)
            row = {"season": season, "features": name}
            for metric in ["accuracy_gain", "log_loss_gain"]:
                samples = daily[metric].to_numpy()[indices].sum(axis=1) / weights
                row[metric] = paired[metric].mean()
                row[f"{metric}_lower95"] = np.quantile(samples, .025)
                row[f"{metric}_upper95"] = np.quantile(samples, .975)
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "paired_uncertainty.csv", index=False)


def grouped_importance(common, out):
    """Permute related inputs together to expose signal hidden by redundancy."""
    full = list(ALL_MODEL_FEATURES)
    groups = {
        "recent_form": [c for c in full if "last_five" in c or "last5" in c],
        "venue": [c for c in full if "record_" in c],
        "rest": [c for c in full if "rest_days" in c],
    }
    groups["season_strength"] = [c for c in full if c not in sum(groups.values(), [])]
    rows = []
    for season in [2023, 2024, 2025]:
        train, val = common.loc[common.season.lt(season)], common.loc[common.season.eq(season)]
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
            max_features=1.0, random_state=42, n_jobs=-1)
        model.fit(train[full], train.home_win)
        base = model.predict_proba(val[full])[:, 1]
        for group, cols in groups.items():
            for repeat in range(5):
                altered = val[full].copy()
                order = np.random.default_rng(repeat).permutation(len(val))
                altered[cols] = altered[cols].iloc[order].to_numpy()
                probability = model.predict_proba(altered)[:, 1]
                rows.append(dict(season=season, group=group, repeat=repeat,
                    accuracy_drop=accuracy_score(val.home_win, base > .5) - accuracy_score(val.home_win, probability > .5),
                    log_loss_increase=log_loss(val.home_win, probability) - log_loss(val.home_win, base)))
    pd.DataFrame(rows).to_csv(out / "grouped_importance.csv", index=False)


def write_metadata(raw, out):
    snapshot = raw.sort_values("id").to_csv(index=False).encode()
    metadata = {
        "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        "rows": len(raw), "snapshot_latest_game": str(raw.game_date.max()),
        "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "forest": {"n_estimators": 200, "min_samples_leaf": 10, "max_features": 1.0},
        "seeds": [0, 42, 123], "validation_seasons": [2023, 2024, 2025],
        "training_rule": "All available seasons strictly before validation season",
        "2026_used_for_selection": False,
        "comparison_rows": "Common complete cases across every candidate feature set",
        "uncertainty": "2000 paired date-block bootstrap replicates; 95% percentile intervals",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))


def run(raw, out):
    out.mkdir(parents=True, exist_ok=True)
    write_metadata(raw, out)
    setup, frame = candidate_data(raw)
    checks = audit(raw, setup, frame)
    sets = feature_sets()
    # Common eligible games across ALL candidate sets, including early-season omissions.
    common = frame.dropna(subset=sorted(set(sum(sets.values(), [])))).copy()
    coverage = frame.groupby("season").agg(imported=("id", "size"), first_date=("date", "min"), last_date=("date", "max"))
    coverage["eligible"] = common.groupby("season").size()
    coverage.to_csv(out / "coverage.csv")
    (out / "audit.json").write_text(json.dumps({"checks": checks, "feature_sets": sets}, indent=2))
    rows, predictions = [], []
    for season in [2023, 2024, 2025]:
        train, val = common.loc[common.season.lt(season)], common.loc[common.season.eq(season)]
        assert set(train.id).isdisjoint(val.id)
        y = val.home_win
        for name, cols in sets.items():
            for seed in [0, 42, 123]:
                model = RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                    max_features=1.0, random_state=seed, n_jobs=-1)
                model.fit(train[cols], train.home_win)
                p = model.predict_proba(val[cols])[:, 1]
                rows.append(dict(season=season, features=name, seed=seed, n=len(y),
                    accuracy=accuracy_score(y, p > .5), log_loss=log_loss(y, p),
                    brier=brier_score_loss(y, p), home_baseline=y.mean(),
                    constant_probability_log_loss=log_loss(y, np.repeat(train.home_win.mean(), len(y)))))
                predictions.extend(dict(season=season, features=name, seed=seed, id=gid,
                    date=str(date), actual=int(actual), probability=float(prob))
                    for gid,date,actual,prob in zip(val.id,val.date,y,p))
        print(f'Finished season {season}: {len(train)} train / {len(val)} validation games', flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(out / 'metrics.csv', index=False)
    paired_uncertainty(predictions, out)
    grouped_importance(common, out)
    summary = results.groupby('features')[['accuracy','log_loss','brier']].mean().sort_values('log_loss')
    summary.to_csv(out / 'summary.csv')
    print(summary.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', help='Optional local pickle from a trusted source; otherwise query DB.')
    args = parser.parse_args()
    if args.snapshot:
        raw = pd.read_pickle(args.snapshot)
    else:
        from data.nbaDataImport import load_nba_games
        with contextlib.redirect_stdout(io.StringIO()):
            raw = load_nba_games()
    run(raw, Path('experiments/results'))
