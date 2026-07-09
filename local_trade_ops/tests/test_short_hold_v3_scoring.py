from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.short_hold_v3_scoring import (
    _liquidity_risk,
    _normalize_series,
    score_candidates_v3,
    sort_candidates_for_orders,
)


class ShortHoldV3ScoringTests(unittest.TestCase):
    def test_score_candidates_penalizes_high_buyability_risk_and_sets_source(self):
        candidates = pd.DataFrame(
            {
                "instrument": ["A", "B"],
                "return_score": [100.0, 0.0],
                "amount": [50_000_000, 50_000_000],
            }
        )
        side_scores = pd.DataFrame(
            {
                "instrument": ["A", "B"],
                "buyability_risk": [1.0, 0.0],
                "strong_prob": [0.0, 0.0],
            }
        )

        scored = score_candidates_v3(candidates, side_scores, alpha=1.0, beta=2.0, gamma=0.0)

        score_by_instrument = scored.set_index("instrument")
        self.assertLess(float(score_by_instrument.loc["A", "final_score"]), float(score_by_instrument.loc["B", "final_score"]))
        self.assertEqual(set(scored["score_source"]), {"v3"})

    def test_score_candidates_uses_score_when_return_score_is_missing(self):
        candidates = pd.DataFrame(
            {
                "instrument": ["A", "B"],
                "score": ["1.5", "3.5"],
            }
        )
        side_scores = pd.DataFrame({"instrument": ["A"]})

        scored = score_candidates_v3(candidates, side_scores)

        self.assertEqual(list(scored["return_score"]), [1.5, 3.5])
        self.assertEqual(list(scored["buyability_risk"]), [0.0, 0.0])
        self.assertEqual(list(scored["strong_prob"]), [0.0, 0.0])
        self.assertEqual(list(scored["liquidity_risk"]), [0.0, 0.0])
        self.assertFalse(scored["final_score"].isna().any())

    def test_empty_candidates_returns_stable_schema(self):
        scored = score_candidates_v3(pd.DataFrame(), pd.DataFrame())

        self.assertTrue(scored.empty)
        for column in [
            "instrument",
            "return_score",
            "buyability_risk",
            "strong_prob",
            "liquidity_risk",
            "final_score",
            "score_source",
        ]:
            self.assertIn(column, scored.columns)

    def test_normalize_series_returns_zeros_when_std_is_unavailable(self):
        constant = _normalize_series(pd.Series([5.0, 5.0]))
        empty = _normalize_series(pd.Series(dtype=float))

        self.assertEqual(list(constant), [0.0, 0.0])
        self.assertTrue(empty.empty)

    def test_normalize_series_keeps_invalid_values_below_valid_value(self):
        normalized = _normalize_series(pd.Series([5.0, np.nan, np.inf, "bad"]))

        valid_score = float(normalized.iloc[0])
        self.assertTrue(all(float(score) < valid_score for score in normalized.iloc[1:]))

    def test_liquidity_risk_clips_to_expected_range(self):
        risk = _liquidity_risk(pd.Series([0, 25_000_000, 50_000_000, 100_000_000]))

        self.assertEqual(list(risk), [1.0, 0.5, 0.0, 0.0])

    def test_sort_candidates_for_orders_prefers_final_score_descending(self):
        candidates = pd.DataFrame(
            {
                "instrument": ["A", "B", "C"],
                "score": [100.0, 1.0, 50.0],
                "final_score": [0.2, 0.5, -1.0],
            },
            index=[10, 11, 12],
        )

        sorted_candidates = sort_candidates_for_orders(candidates)

        self.assertEqual(list(sorted_candidates["instrument"]), ["B", "A", "C"])
        self.assertEqual(list(sorted_candidates.index), [0, 1, 2])

    def test_sort_candidates_for_orders_falls_back_to_score_descending(self):
        candidates = pd.DataFrame(
            {
                "instrument": ["A", "B", "C"],
                "score": [1.5, 3.5, 2.5],
            }
        )

        sorted_candidates = sort_candidates_for_orders(candidates)

        self.assertEqual(list(sorted_candidates["instrument"]), ["B", "C", "A"])

    def test_sort_candidates_for_orders_sorts_numeric_strings_and_invalid_last(self):
        candidates = pd.DataFrame(
            {
                "instrument": ["A", "B", "C", "D"],
                "score": ["2", "10", "bad", np.inf],
            }
        )

        sorted_candidates = sort_candidates_for_orders(candidates)

        self.assertEqual(list(sorted_candidates["instrument"]), ["B", "A", "C", "D"])

    def test_sort_candidates_for_orders_is_stable_for_tied_scores(self):
        candidates = pd.DataFrame(
            {
                "instrument": ["A", "B", "C", "D"],
                "final_score": ["1.0", 0.5, "1.0", 0.5],
            }
        )

        sorted_candidates = sort_candidates_for_orders(candidates)

        self.assertEqual(list(sorted_candidates["instrument"]), ["A", "C", "B", "D"])

    def test_score_candidates_keeps_invalid_return_scores_below_valid_scores(self):
        candidates = pd.DataFrame(
            {
                "instrument": ["VALID", "NAN", "INF", "BAD"],
                "return_score": [5.0, np.nan, np.inf, "bad"],
                "amount": [50_000_000] * 4,
            }
        )

        scored = score_candidates_v3(
            candidates,
            pd.DataFrame(),
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
        )
        score_by_instrument = scored.set_index("instrument")["final_score"]

        self.assertTrue(
            all(
                float(score_by_instrument.loc[instrument]) < float(score_by_instrument.loc["VALID"])
                for instrument in ["NAN", "INF", "BAD"]
            )
        )

    def test_score_candidates_aggregates_duplicate_side_scores_by_max(self):
        candidates = pd.DataFrame(
            {
                "instrument": ["A", "B"],
                "return_score": [1.0, 2.0],
                "amount": [50_000_000, 50_000_000],
            }
        )
        side_scores = pd.DataFrame(
            {
                "instrument": ["A", "B", "A"],
                "buyability_risk": [0.2, 0.4, 0.9],
                "strong_prob": [0.8, 0.3, 0.1],
            }
        )

        scored = score_candidates_v3(candidates, side_scores)
        by_instrument = scored.set_index("instrument")

        self.assertEqual(float(by_instrument.loc["A", "buyability_risk"]), 0.9)
        self.assertEqual(float(by_instrument.loc["A", "strong_prob"]), 0.8)
        self.assertEqual(float(by_instrument.loc["B", "buyability_risk"]), 0.4)
        self.assertEqual(float(by_instrument.loc["B", "strong_prob"]), 0.3)

    def test_score_candidates_rejects_duplicate_candidate_instruments(self):
        candidates = pd.DataFrame(
            {
                "instrument": ["A", "A"],
                "return_score": [1.0, 2.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate instrument"):
            score_candidates_v3(candidates, pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
