import importlib.util
import unittest
from datetime import date
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "bin" / "resolve_required_date.py"


def load_module():
    spec = importlib.util.spec_from_file_location("resolve_required_date", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ResolveRequiredDateTest(unittest.TestCase):
    def test_explicit_required_date_is_preserved(self):
        module = load_module()
        self.assertEqual(module.resolve_required_date("2026-06-21", today=date(2026, 6, 22)), "2026-06-21")

    def test_auto_skips_weekends_and_configured_market_holidays(self):
        module = load_module()
        self.assertEqual(
            module.resolve_required_date(
                "auto",
                today=date(2026, 6, 22),
                holidays={"2026-06-19"},
            ),
            "2026-06-18",
        )

    def test_auto_keeps_prior_weekday_when_no_holiday_is_configured(self):
        module = load_module()
        self.assertEqual(module.resolve_required_date("auto", today=date(2026, 6, 22), holidays=set()), "2026-06-19")

    def test_auto_skips_weekend_to_previous_friday(self):
        module = load_module()
        self.assertEqual(module.resolve_required_date("auto", today=date(2026, 6, 15), holidays=set()), "2026-06-12")


if __name__ == "__main__":
    unittest.main()
