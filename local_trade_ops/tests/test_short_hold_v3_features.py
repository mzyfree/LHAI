from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.short_hold_v3_features import (
    V3FeatureColumns,
    build_inference_features,
    build_training_labels,
    stock_limit_up_pct,
)


class ShortHoldV3FeatureTests(unittest.TestCase):
    def setUp(self):
        index = pd.MultiIndex.from_tuples(
            [
                ("2026-06-10", "SH600001"),
                ("2026-06-11", "SH600001"),
                ("2026-06-12", "SH600001"),
                ("2026-06-15", "SH600001"),
                ("2026-06-10", "SZ300001"),
                ("2026-06-11", "SZ300001"),
                ("2026-06-12", "SZ300001"),
                ("2026-06-15", "SZ300001"),
                ("2026-06-10", "SZ000001"),
                ("2026-06-11", "SZ000001"),
                ("2026-06-12", "SZ000001"),
                ("2026-06-15", "SZ000001"),
            ],
            names=["datetime", "instrument"],
        )
        self.ohlcv = pd.DataFrame(
            {
                "open": [10.0, 10.3, 11.0, 11.5, 20.0, 20.8, 22.9, 23.5, 8.0, 8.3, 9.0, 9.9],
                "high": [10.4, 10.8, 11.5, 11.8, 21.0, 22.8, 24.8, 24.2, 8.4, 8.7, 9.9, 10.4],
                "low": [9.9, 10.1, 10.8, 11.1, 19.8, 20.5, 22.5, 23.0, 7.9, 8.1, 8.9, 9.6],
                "close": [10.2, 10.5, 11.2, 11.6, 20.5, 22.6, 23.1, 24.0, 8.2, 8.5, 9.1, 10.1],
                "volume": [1000, 1200, 1500, 1400, 2000, 2500, 3000, 2800, 900, 950, 1000, 1100],
                "amount": [
                    10_200_000,
                    12_600_000,
                    16_800_000,
                    16_240_000,
                    41_000_000,
                    56_500_000,
                    69_300_000,
                    67_200_000,
                    7_380_000,
                    8_075_000,
                    9_100_000,
                    11_110_000,
                ],
            },
            index=index,
        )
        self.ohlcv.index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(self.ohlcv.index.get_level_values("datetime")),
                self.ohlcv.index.get_level_values("instrument"),
            ],
            names=["datetime", "instrument"],
        )

    def test_stock_limit_up_pct_handles_board_prefix(self):
        self.assertEqual(stock_limit_up_pct("SH600001"), 0.10)
        self.assertEqual(stock_limit_up_pct("SZ000001"), 0.10)
        self.assertEqual(stock_limit_up_pct("SZ300001"), 0.20)
        self.assertEqual(stock_limit_up_pct("SH688001"), 0.20)

    def test_build_inference_features_uses_signal_date_only(self):
        rows = pd.DataFrame(
            {
                "instrument": ["SH600001", "SZ300001"],
                "return_score": [2.0, 3.0],
                "model_rank": [2, 1],
            }
        )
        features = build_inference_features(rows, self.ohlcv, pd.Timestamp("2026-06-11"))

        self.assertEqual(list(features["instrument"]), ["SH600001", "SZ300001"])
        for column in [
            "instrument",
            "ret_1d",
            "amount_mean_3d",
            "distance_to_limit_up",
            "limit_up_pct",
        ]:
            self.assertIn(column, features.columns)

        self.assertAlmostEqual(float(features.loc[0, "ret_1d"]), 10.5 / 10.2 - 1.0, places=8)
        self.assertAlmostEqual(float(features.loc[0, "amount_mean_3d"]), (10_200_000 + 12_600_000) / 2, places=8)
        self.assertAlmostEqual(float(features.loc[1, "limit_up_pct"]), 0.20, places=8)

    def test_build_training_labels_uses_future_dates_for_labels(self):
        rows = pd.DataFrame(
            {
                "instrument": ["SH600001", "SZ000001"],
                "return_score": [2.0, 4.0],
                "model_rank": [2, 1],
            }
        )
        labels = build_training_labels(
            rows,
            self.ohlcv,
            signal_date=pd.Timestamp("2026-06-11"),
            entry_date=pd.Timestamp("2026-06-12"),
            exit_date=pd.Timestamp("2026-06-15"),
        )

        self.assertEqual(list(labels["instrument"]), ["SH600001", "SZ000001"])
        self.assertIn("signal_date", labels.columns)
        self.assertEqual(pd.Timestamp(labels.loc[0, "signal_date"]), pd.Timestamp("2026-06-11"))
        self.assertAlmostEqual(float(labels.loc[0, "realized_return"]), 11.6 / 11.0 - 1.0, places=8)
        self.assertEqual(int(labels.loc[0, "open_gt_5pct"]), 0)
        self.assertEqual(int(labels.loc[0, "buyability_bad"]), 0)
        self.assertEqual(int(labels.loc[0, "strong_label"]), 0)
        self.assertAlmostEqual(float(labels.loc[1, "realized_return"]), 10.1 / 9.0 - 1.0, places=8)
        self.assertEqual(int(labels.loc[1, "open_gt_5pct"]), 1)
        self.assertEqual(int(labels.loc[1, "buyability_bad"]), 1)
        self.assertEqual(int(labels.loc[1, "strong_label"]), 1)

    def test_candidate_scores_fall_back_when_primary_value_is_invalid(self):
        rows = pd.DataFrame(
            {
                "instrument": ["SH600001"],
                "return_score": [float("nan")],
                "score": [9.5],
                "model_rank": [""],
                "rank": [4],
            }
        )

        features = build_inference_features(rows, self.ohlcv, pd.Timestamp("2026-06-11"))

        self.assertAlmostEqual(float(features.loc[0, "return_score"]), 9.5, places=8)
        self.assertAlmostEqual(float(features.loc[0, "model_rank"]), 4.0, places=8)

    def test_duplicate_candidate_instruments_keep_first_row(self):
        rows = pd.DataFrame(
            {
                "instrument": ["SH600001", "SH600001"],
                "return_score": [2.0, 9.0],
                "score": [3.0, 10.0],
                "model_rank": [1, 99],
            }
        )

        features = build_inference_features(rows, self.ohlcv, pd.Timestamp("2026-06-11"))

        self.assertEqual(list(features["instrument"]), ["SH600001"])
        self.assertAlmostEqual(float(features.loc[0, "return_score"]), 2.0, places=8)
        self.assertAlmostEqual(float(features.loc[0, "score"]), 3.0, places=8)
        self.assertAlmostEqual(float(features.loc[0, "model_rank"]), 1.0, places=8)

    def test_empty_inference_and_training_results_keep_stable_schema(self):
        rows = pd.DataFrame({"instrument": ["SH999999"], "return_score": [1.0], "model_rank": [1]})

        empty_features = build_inference_features(pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2026-06-11"))
        self.assertTrue(empty_features.empty)
        self.assertIn("instrument", empty_features.columns)
        self.assertIn("return_score", empty_features.columns)

        empty_labels = build_training_labels(
            pd.DataFrame(),
            pd.DataFrame(),
            signal_date=pd.Timestamp("2026-06-11"),
            entry_date=pd.Timestamp("2026-06-12"),
            exit_date=pd.Timestamp("2026-06-15"),
        )
        self.assertTrue(empty_labels.empty)
        self.assertIn("signal_date", empty_labels.columns)
        self.assertIn("strong_label", empty_labels.columns)

        features = build_inference_features(rows, self.ohlcv, pd.Timestamp("2026-06-11"))

        expected_feature_columns = ["instrument", *V3FeatureColumns().numeric]
        self.assertTrue(features.empty)
        for column in expected_feature_columns:
            self.assertIn(column, features.columns)

        labels = build_training_labels(
            rows,
            self.ohlcv,
            signal_date=pd.Timestamp("2026-06-11"),
            entry_date=pd.Timestamp("2026-06-12"),
            exit_date=pd.Timestamp("2026-06-15"),
        )

        expected_label_columns = [
            *expected_feature_columns,
            "signal_date",
            "entry_date",
            "exit_date",
            "entry_open",
            "exit_close",
            "entry_gap",
            "realized_return",
            "open_gt_5pct",
            "open_limit_up",
            "buyability_bad",
            "strong_label",
        ]
        self.assertTrue(labels.empty)
        for column in expected_label_columns:
            self.assertIn(column, labels.columns)

    def test_duplicate_ohlcv_keys_raise_value_error(self):
        duplicate_ohlcv = pd.concat([self.ohlcv, self.ohlcv.iloc[[0]]])
        rows = pd.DataFrame({"instrument": ["SH600001"], "return_score": [2.0], "model_rank": [1]})

        with self.assertRaisesRegex(ValueError, "duplicate OHLCV rows"):
            build_inference_features(rows, duplicate_ohlcv, pd.Timestamp("2026-06-11"))


if __name__ == "__main__":
    unittest.main()
