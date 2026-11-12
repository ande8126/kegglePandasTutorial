"""Shared, pregame NBA features for notebook experiments."""

from types import SimpleNamespace

import pandas as pd


TEAM_FEATURES = (
    "season_win_percentage",
    "win_percentage_last_five_games",
    "rest_days_since_last_game",
    "average_point_differential",
    "points_scored_per_game",
    "points_scored_last_five_games",
    "points_allowed_per_game",
    "points_allowed_last_five_games",
    "record_at_home",
    "record_as_visitor",
)

DIFFERENCE_FEATURES = (
    "last5_win_pct_diff",
    "season_point_margin_diff",
    "last5_point_margin_diff",
    "rest_days_diff",
)

ALL_MODEL_FEATURES = tuple(
    f"{side}_{feature}"
    for side in ("home", "visitor")
    for feature in TEAM_FEATURES
) + DIFFERENCE_FEATURES

# The study's recommended working set; the advantage over eight inputs is modest.
DEFAULT_MODEL_FEATURES = ALL_MODEL_FEATURES


def setup_nba_experiment(*, validation_season=2025, model_features=None, nba_data=None):
    """Load completed games, build historical features, and split by season.

    Returns an object with nba_data, home, away, team_games, team_features,
    home_features, visitor_features, nba_model_data, model_data, train,
    validation, X_train, y_train, X_validation, y_validation, features,
    nba_model_features, and all_model_features attributes.

    Training uses all available seasons before validation_season; validation
    uses that season only. Later seasons remain in nba_model_data/model_data.
    Defaults to all 24 pregame model features, following the feature study.
    Pass a list of model_features to compare a smaller set of inputs.
    Rows missing selected inputs are dropped only from model_data.

    All historical features reset by season and exclude the current result.
    Earlier validation-season results inform later games, as in daily use.
    Regular-season and postseason games share the same season history.

    Supply nba_data to reuse an existing DataFrame without querying PostgreSQL.
    Input is copied; dates may be named game_date or date. Scheduled games
    with missing/zero scores are excluded before computing any history.
    """
    if nba_data is None:
        from data.nbaDataImport import load_nba_games

        nba_data = load_nba_games()

    nba_data = nba_data.rename(columns={"game_date": "date"}).copy()
    required = [
        "id", "date", "season", "home_team_id", "visitor_team_id",
        "home_team_score", "visitor_team_score",
    ]
    missing = set(required) - set(nba_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    completed = (
        nba_data["home_team_score"].gt(0)
        & nba_data["visitor_team_score"].gt(0)
    )
    nba_data = nba_data.loc[completed].copy().reset_index(drop=True)
    if nba_data[required].isna().any().any():
        raise ValueError("Completed games must have non-null identifiers and dates.")
    if nba_data["id"].duplicated().any():
        raise ValueError("Expected one row per game id.")
    if nba_data["home_team_id"].eq(nba_data["visitor_team_id"]).any():
        raise ValueError("A team cannot play itself.")
    nba_data["date"] = pd.to_datetime(nba_data["date"], utc=True, errors="raise")

    nba_model_features = list(
        DEFAULT_MODEL_FEATURES if model_features is None else model_features
    )
    if not nba_model_features or len(set(nba_model_features)) != len(nba_model_features):
        raise ValueError("Choose a nonempty list of distinct model features.")
    unknown = set(nba_model_features) - set(ALL_MODEL_FEATURES)
    if unknown:
        raise ValueError(f"Unknown model features: {sorted(unknown)}")

    # One row per game from the home team's perspective.
    home = nba_data[
        ["id", "date", "season", "home_team_id",
         "home_team_score", "visitor_team_score"]
    ].rename(columns={
        "home_team_id": "team_id",
        "home_team_score": "points_for",
        "visitor_team_score": "points_against"
    })

    # The same games from the visiting team's perspective.
    away = nba_data[
        ["id", "date", "season", "visitor_team_id",
         "visitor_team_score", "home_team_score"]
    ].rename(columns={
        "visitor_team_id": "team_id",
        "visitor_team_score": "points_for",
        "home_team_score": "points_against"
    })

    # Track the venue before stacking the two perspectives.
    home["is_home"] = True
    away["is_home"] = False

    # Stack the tables vertically, similar to SQL UNION ALL.
    team_games = pd.concat([home, away], ignore_index=True)

    team_games = team_games.sort_values(
        ["team_id", "season", "date", "id"]
    ).reset_index(drop=True)
    team_games["won"] = (
        team_games["points_for"] > team_games["points_against"]
    ).astype(int)

    team_games["previous_game_won"] = (
        team_games
        .groupby(["team_id", "season"])["won"]
        .shift(1)
    )
    team_games["point_differential"] = (
        team_games["points_for"] - team_games["points_against"]
    )

    groups = team_games.groupby(["team_id", "season"])

    # Win percentages BEFORE this game.
    team_games["season_win_percentage"] = groups["won"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    team_games["win_percentage_last_five_games"] = groups["won"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=5).mean()
    )

    # Calendar days between game dates: back-to-back games = 1.
    previous_date = groups["date"].shift(1)
    team_games["rest_days_since_last_game"] = (
        team_games["date"].dt.normalize() -
        previous_date.dt.normalize()
    ).dt.days

    # Average scoring margin over ALL earlier games this season.
    team_games["average_point_differential"] = (
        groups["point_differential"].transform(
            lambda s: s.shift(1).expanding().mean()
        ))

    for source, prefix in [
        ("points_for", "points_scored"),
        ("points_against", "points_allowed"),
    ]:
        team_games[f"{prefix}_per_game"] = groups[source].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        team_games[f"{prefix}_last_five_games"] = groups[source].transform(
            lambda s: s.shift(1).rolling(5, min_periods=5).mean()
        )

    # Pregame home and visitor records, carried forward through all games.
    for venue, venue_mask in [
        ("home", team_games["is_home"]),
        ("visitor", ~team_games["is_home"]),
    ]:
        # Indicators are zero for games played at the other venue.
        venue_games = venue_mask.astype(int)
        venue_wins = venue_games * team_games["won"]

        group_keys = [team_games["team_id"], team_games["season"]]

        # Subtract this game's contribution to get the PREGAME record.
        games_before = (
            venue_games.groupby(group_keys).cumsum() - venue_games
        )
        wins_before = (
            venue_wins.groupby(group_keys).cumsum() - venue_wins
        )

        team_games[f"{venue}_wins_before"] = wins_before
        team_games[f"{venue}_losses_before"] = games_before - wins_before

        record_column = (
            "record_at_home" if venue == "home" else "record_as_visitor"
        )
        team_games[record_column] = wins_before / games_before.where(
            games_before > 0
        )
    features = list(TEAM_FEATURES)

    # Select only identifiers and pregame features.
    # Exclude current-game scores/results to avoid leakage.
    team_features = team_games[["id", "team_id"] + features].copy()

    home_features = team_features.rename(columns={
        "team_id": "home_team_id",
        **{column: f"home_{column}" for column in features},
    })

    visitor_features = team_features.rename(columns={
        "team_id": "visitor_team_id",
        **{column: f"visitor_{column}" for column in features},
    })

    nba_model_data = nba_data.merge(
        home_features,
        on=["id", "home_team_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        visitor_features,
        on=["id", "visitor_team_id"],
        how="left",
        validate="one_to_one",
    )
    nba_model_data["home_win"] = (
        nba_model_data["home_team_score"] > nba_model_data["visitor_team_score"]
    ).astype(int)

    nba_model_data["last5_win_pct_diff"] = (
        nba_model_data["home_win_percentage_last_five_games"]
        - nba_model_data["visitor_win_percentage_last_five_games"]
    )

    nba_model_data["season_point_margin_diff"] = (
        nba_model_data["home_average_point_differential"]
        - nba_model_data["visitor_average_point_differential"]
    )

    nba_model_data["last5_point_margin_diff"] = (
        (nba_model_data["home_points_scored_last_five_games"]
         - nba_model_data["home_points_allowed_last_five_games"])
        - (nba_model_data["visitor_points_scored_last_five_games"]
           - nba_model_data["visitor_points_allowed_last_five_games"])
    )

    nba_model_data["rest_days_diff"] = (
        nba_model_data["home_rest_days_since_last_game"]
        - nba_model_data["visitor_rest_days_since_last_game"]
    )

    model_data = nba_model_data.dropna(subset=nba_model_features).copy()
    train = model_data.loc[model_data["season"] < validation_season].copy()
    validation = model_data.loc[
        model_data["season"] == validation_season
    ].copy()
    if train.empty or validation.empty:
        raise ValueError(
            "No eligible training or validation games. Check validation_season, "
            "available seasons, and missing selected features."
        )

    return SimpleNamespace(
        nba_data=nba_data,
        home=home,
        away=away,
        team_games=team_games,
        features=features,
        team_features=team_features,
        home_features=home_features,
        visitor_features=visitor_features,
        nba_model_data=nba_model_data,
        nba_model_features=nba_model_features,
        all_model_features=list(ALL_MODEL_FEATURES),
        model_data=model_data,
        train=train,
        validation=validation,
        X_train=train[nba_model_features].copy(),
        y_train=train["home_win"].copy(),
        X_validation=validation[nba_model_features].copy(),
        y_validation=validation["home_win"].copy(),
    )
