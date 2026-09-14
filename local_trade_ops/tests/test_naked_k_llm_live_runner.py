import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import naked_k_llm_live_runner as runner


class FrozenOpenPoolTest(unittest.TestCase):
    def test_keeps_only_rows_allowed_by_the_frozen_open_guards(self):
        base = {
            "ts_code": "600001.SH",
            "open_gap": 0.03,
            "prior_return_20d": 0.20,
            "prior_limit_count60": 1,
        }
        rows = [
            base,
            {**base, "ts_code": "600002.SH", "open_gap": 0.051},
            {**base, "ts_code": "600003.SH", "open_gap": 0.095},
            {**base, "ts_code": "600004.SH", "prior_return_20d": 0.251},
            {**base, "ts_code": "600005.SH", "prior_limit_count60": 0},
            {**base, "ts_code": "600006.SH", "open_gap": None},
        ]

        self.assertEqual(runner.frozen_open_pool(rows), [base])

    def test_limit_threshold_respects_board_rules(self):
        self.assertEqual(runner.limit_threshold("600001.SH"), 0.095)
        self.assertEqual(runner.limit_threshold("300001.SZ"), 0.195)
        self.assertEqual(runner.limit_threshold("688001.SH"), 0.195)
        self.assertEqual(runner.limit_threshold("830001.BJ"), 0.295)


if __name__ == "__main__":
    unittest.main()
