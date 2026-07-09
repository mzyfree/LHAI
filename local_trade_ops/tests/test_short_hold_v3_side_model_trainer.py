import json
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin.short_hold_v3_features import V3FeatureColumns
from bin.short_hold_v3_side_model import QuantileSideModel


TRAINER = ROOT / "bin" / "train_short_hold_v3_side_models.py"
PYTHON = Path(sys.executable)


def _sample_columns() -> list[str]:
    return [
        *V3FeatureColumns().numeric,
        "turnover_change_5",
        "buyability_bad_label",
        "strong_next_label",
    ]


def _sample_frame() -> pd.DataFrame:
    rows = []
    for index in range(8):
        row = {column: float(index + 1) for column in V3FeatureColumns().numeric}
        row.update(
            {
                "distance_to_limit_up": [0.09, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.005][index],
                "turnover_change_5": [0.1, 0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0][index],
                "buyability_bad_label": int(index in {5, 6, 7}),
                "strong_next_label": int(index in {4, 6, 7}),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=_sample_columns())


def _run_trainer(samples_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(PYTHON),
            str(TRAINER),
            "--samples",
            str(samples_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class ShortHoldV3SideModelTests(unittest.TestCase):
    def test_predict_proba_uses_invalid_probability_for_non_finite_inputs(self):
        model = QuantileSideModel("feature", "high", invalid_probability=0.05).fit(
            pd.DataFrame({"feature": [1.0, 2.0, 3.0], "label": [0, 1, 1]}),
            "label",
        )

        probabilities = model.predict_proba(pd.DataFrame({"feature": [3.0, np.nan, np.inf, "bad"]}))

        self.assertGreater(float(probabilities[0, 1]), 0.05)
        self.assertEqual(list(probabilities[1:, 1]), [0.05, 0.05, 0.05])

    def test_fit_rejects_all_invalid_feature_values(self):
        model = QuantileSideModel("feature", "high")

        with self.assertRaisesRegex(ValueError, "no finite training samples"):
            model.fit(pd.DataFrame({"feature": [np.nan, np.inf, "bad"], "label": [1, 0, 1]}), "label")

    def test_fit_falls_back_to_all_finite_samples_when_labels_are_all_negative(self):
        model = QuantileSideModel("feature", "high").fit(
            pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0], "label": [0, 0, 0, 0]}),
            "label",
        )

        probabilities = model.predict_proba(pd.DataFrame({"feature": [1.0, 4.0]}))

        self.assertTrue(model.training_summary["fallback_to_all_finite"])
        self.assertEqual(model.training_summary["selected_sample_count"], 4)
        self.assertLess(float(probabilities[0, 1]), float(probabilities[1, 1]))


class ShortHoldV3SideModelTrainerTests(unittest.TestCase):
    def test_cli_trains_models_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            samples_path = tmp / "samples.csv"
            output_dir = tmp / "models"
            _sample_frame().to_csv(samples_path, index=False)

            result = _run_trainer(samples_path, output_dir)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output_dir / "buyability_model.pkl").exists())
            self.assertTrue((output_dir / "strong_model.pkl").exists())
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["rows"], 8)
            self.assertEqual(metadata["models"]["buyability_model.pkl"]["column"], "distance_to_limit_up")
            self.assertEqual(metadata["models"]["strong_model.pkl"]["column"], "turnover_change_5")
            self.assertEqual(metadata["models"]["buyability_model.pkl"]["invalid_probability"], 0.95)
            self.assertEqual(metadata["models"]["strong_model.pkl"]["invalid_probability"], 0.05)

    def test_pickle_artifacts_load_in_fresh_process_without_cli_module_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            samples_path = tmp / "samples.csv"
            output_dir = tmp / "models"
            _sample_frame().to_csv(samples_path, index=False)
            train_result = _run_trainer(samples_path, output_dir)
            self.assertEqual(train_result.returncode, 0, train_result.stderr)

            code = textwrap.dedent(
                f"""
                import pickle
                from pathlib import Path

                model = pickle.loads(Path({str(output_dir / "buyability_model.pkl")!r}).read_bytes())
                assert model.__class__.__module__ == "bin.short_hold_v3_side_model"
                print(model.predict_proba(__import__("pandas").DataFrame({{"distance_to_limit_up": [float("nan")]}}))[0, 1])
                """
            )
            load_result = subprocess.run(
                [str(PYTHON), "-c", code],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(load_result.returncode, 0, load_result.stderr)
            self.assertEqual(float(load_result.stdout.strip()), 0.95)

    def test_empty_csv_with_correct_columns_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            samples_path = tmp / "empty.csv"
            output_dir = tmp / "models"
            pd.DataFrame(columns=_sample_columns()).to_csv(samples_path, index=False)

            result = _run_trainer(samples_path, output_dir)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("contains no rows", result.stderr)
            self.assertFalse((output_dir / "buyability_model.pkl").exists())

    def test_invalid_numeric_values_are_not_treated_as_zero_during_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            samples = _sample_frame()
            samples["distance_to_limit_up"] = [np.nan, np.inf, "bad", "", None, np.nan, np.inf, "bad"]
            samples_path = tmp / "samples.csv"
            output_dir = tmp / "models"
            samples.to_csv(samples_path, index=False)

            result = _run_trainer(samples_path, output_dir)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("distance_to_limit_up", result.stderr)
            self.assertIn("no finite training samples", result.stderr)
            self.assertFalse((output_dir / "metadata.json").exists())

    def test_all_negative_labels_train_with_finite_feature_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            samples = _sample_frame()
            samples["buyability_bad_label"] = 0
            samples["strong_next_label"] = 0
            samples_path = tmp / "samples.csv"
            output_dir = tmp / "models"
            samples.to_csv(samples_path, index=False)

            result = _run_trainer(samples_path, output_dir)

            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["models"]["buyability_model.pkl"]["fallback_to_all_finite"])
            self.assertTrue(metadata["models"]["strong_model.pkl"]["fallback_to_all_finite"])
            self.assertEqual(metadata["models"]["buyability_model.pkl"]["selected_sample_count"], 8)


if __name__ == "__main__":
    unittest.main()
