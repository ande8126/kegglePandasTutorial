"""Database-free checks for the shared notebook feature pipeline."""

import json
from pathlib import Path
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from data.nba_setup import ALL_MODEL_FEATURES, setup_nba_experiment


class NBASetupTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for season in [2023, 2024, 2025, 2026]:
            for day in range(1, 21):
                rows.append({
                    "id": season * 100 + day,
                    "game_date": f"{season}-11-{day:02}",
                    "season": season,
                    "home_team_id": 1 if day % 2 else 2,
                    "visitor_team_id": 2 if day % 2 else 1,
                    "home_team_score": 100 + day % 7,
                    "visitor_team_score": 99 + day % 5,
                })
        self.raw = pd.DataFrame(rows).sample(frac=1, random_state=42)

    def test_tables_splits_and_season_records(self):
        original = self.raw.copy(deep=True)
        result = setup_nba_experiment(nba_data=self.raw)
        assert_frame_equal(self.raw, original)
        self.assertEqual(len(result.team_games), 2 * len(self.raw))
        self.assertEqual(len(result.nba_model_data), len(self.raw))
        self.assertEqual(result.train.season.max(), 2024)
        self.assertTrue(result.validation.season.eq(2025).all())
        self.assertTrue(set(result.train.id).isdisjoint(result.validation.id))
        self.assertFalse(result.X_train.isna().any().any())
        self.assertTrue(set(ALL_MODEL_FEATURES).issubset(result.nba_model_data))
        for _, games in result.team_games.groupby(["team_id", "season"]):
            self.assertTrue(pd.isna(games.iloc[0].season_win_percentage))
            for position, (_, game) in enumerate(games.iterrows()):
                prior = games.iloc[:position]
                for venue, is_home in [("home", True), ("visitor", False)]:
                    history = prior.loc[prior.is_home.eq(is_home)]
                    self.assertEqual(game[f"{venue}_wins_before"], history.won.sum())
                    self.assertEqual(
                        game[f"{venue}_losses_before"], len(history) - history.won.sum()
                    )

    def test_feature_parity_with_notebook(self):
        notebook = Path(__file__).resolve().parents[1] / "random_forest.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        # Execute only the original preparation cells, never DB/model cells.
        # Reset the index to match the DB loader's index for its target assignment.
        namespace = {"pd": pd, "nba_data": self.raw.reset_index(drop=True)}
        for index in [6, 8, 10, 11, 12, 13, 14]:
            exec("".join(cells[index]["source"]), namespace)
        actual = setup_nba_experiment(nba_data=self.raw).nba_model_data
        columns = ["id", "home_win"] + list(ALL_MODEL_FEATURES)
        assert_frame_equal(
            actual[columns].sort_values("id").reset_index(drop=True),
            namespace["nba_model_data"][columns].sort_values("id").reset_index(drop=True),
        )

    def test_current_and_future_results_do_not_change_pregame_features(self):
        changed = self.raw.copy()
        target = 202510
        changed.loc[changed.id.eq(target), "home_team_score"] = 180
        before = setup_nba_experiment(nba_data=self.raw).nba_model_data
        after = setup_nba_experiment(nba_data=changed).nba_model_data
        columns = ["id"] + list(ALL_MODEL_FEATURES)
        assert_frame_equal(
            before.loc[before.id.le(target), columns].sort_values("id").reset_index(drop=True),
            after.loc[after.id.le(target), columns].sort_values("id").reset_index(drop=True),
        )

    def test_scheduled_games_custom_features_and_invalid_options(self):
        scheduled = self.raw.iloc[[0]].copy()
        scheduled["id"] = 999999
        scheduled["home_team_score"] = 0
        scheduled["visitor_team_score"] = None
        before = setup_nba_experiment(nba_data=self.raw)
        after = setup_nba_experiment(nba_data=pd.concat([self.raw, scheduled]))
        # Adding missing scores can change input dtypes, but not feature values.
        assert_frame_equal(before.nba_model_data, after.nba_model_data, check_dtype=False)
        custom = setup_nba_experiment(
            nba_data=self.raw, validation_season=2026,
            model_features=["season_point_margin_diff"],
        )
        self.assertEqual(list(custom.X_train), ["season_point_margin_diff"])
        self.assertEqual(custom.train.season.max(), 2025)
        for options in [
            {"model_features": ["home_team_score"]},
            {"model_features": []},
            {"validation_season": 2030},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                setup_nba_experiment(nba_data=self.raw, **options)
        with self.assertRaises(ValueError):
            setup_nba_experiment(nba_data=pd.concat([self.raw, self.raw.iloc[[0]]]))


if __name__ == "__main__":
    unittest.main()
