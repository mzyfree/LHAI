import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd


OPS_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPS_HOME / "bin"))

from backtest_naked_k_deepseek_intraday import intraday_facts  # noqa: E402
from naked_k_llm_live_runner import frozen_open_pool, tencent_5m  # noqa: E402


class LiveIntradayFactsTest(unittest.TestCase):
    def test_frozen_pool_does_not_filter_limit_memory_or_prior_20d_return(self):
        rows = [
            {"ts_code": "600001.SH", "open_gap": 0.01, "prior_return_20d": 0.40, "prior_limit_count60": 0},
            {"ts_code": "600002.SH", "open_gap": 0.051, "prior_return_20d": 0.01, "prior_limit_count60": 3},
            {"ts_code": "600003.SH", "open_gap": 0.02},
        ]

        self.assertEqual(
            [row["ts_code"] for row in frozen_open_pool(rows)],
            ["600001.SH", "600003.SH"],
        )

    @patch("naked_k_llm_live_runner.requests.get")
    def test_tencent_hand_volume_converted_to_shares(self, get):
        response = Mock()
        response.json.return_value = {
            "data": {
                "sh600353": {
                    "data": {
                        "data": [
                            "0930 35.20 100 352000.00",
                            "0931 35.00 200 702000.00",
                        ]
                    }
                }
            }
        }
        response.raise_for_status.return_value = None
        get.return_value = response

        with patch("naked_k_llm_live_runner.datetime") as current_datetime:
            current_datetime.now.return_value.astimezone.return_value.strftime.return_value = "2026-09-18"
            frame = tencent_5m("600353.SH", "2026-09-18")

        self.assertEqual(frame.iloc[0]["vol"], 20_000)
        self.assertAlmostEqual(frame.iloc[0]["amount"] / frame.iloc[0]["vol"], 35.1)

    def test_eastmoney_hand_volume_converted_to_shares_produces_price_vwap(self):
        frame = pd.DataFrame([
            {
                "trade_time": pd.Timestamp("2026-09-18 09:35:00"),
                "open": 35.20,
                "high": 35.20,
                "low": 34.20,
                "close": 34.26,
                "vol": 108_958 * 100,
                "amount": 381_862_478,
            }
        ])

        facts = intraday_facts(
            frame,
            pd.Timestamp("2026-09-18 09:35:00"),
            sector_edge=0.02,
            item={"prior5d": 0.06, "patternCount": 0, "limitMemory60": 7},
        )

        self.assertAlmostEqual(facts["vwap"], 35.0468, places=4)
        self.assertFalse(facts["aboveVwap"])


if __name__ == "__main__":
    unittest.main()
