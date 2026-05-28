from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QLIB_SRC = PROJECT_ROOT / "qlib"
if str(QLIB_SRC) not in sys.path:
    sys.path.insert(0, str(QLIB_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import torch
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow.record_temp import SignalRecord

from run_prediction_fusion import load_config, signal_metrics
from qlib.contrib.model.pytorch_tcts import GRUModel, MLPModel


def build_models(model_cfg: dict) -> tuple[GRUModel, MLPModel]:
    kwargs = model_cfg["kwargs"]
    fore_model = GRUModel(
        d_feat=kwargs.get("d_feat", 6),
        hidden_size=kwargs.get("hidden_size", 64),
        num_layers=kwargs.get("num_layers", 2),
        dropout=kwargs.get("dropout", 0.0),
    )
    weight_model = MLPModel(
        d_feat=kwargs.get("input_dim", 360) + 3 * kwargs.get("output_dim", 3) + 1,
        hidden_size=kwargs.get("hidden_size", 64),
        num_layers=kwargs.get("num_layers", 2),
        dropout=kwargs.get("dropout", 0.0),
        output_dim=kwargs.get("output_dim", 3),
    )
    return fore_model, weight_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe TCTS weight files against a dataset config.")
    parser.add_argument("--config", required=True, help="TCTS workflow config path.")
    parser.add_argument("--fore-bin", required=True, help="Path to *_fore_model.bin")
    parser.add_argument("--weight-bin", required=True, help="Path to *_weight_model.bin")
    parser.add_argument("--provider-uri", default=None, help="Optional provider_uri override.")
    parser.add_argument("--save-pred", default=None, help="Optional path to save generated pred.pkl")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    provider_uri = args.provider_uri or config["qlib_init"]["provider_uri"]

    qlib.init(provider_uri=str(Path(provider_uri).expanduser()), region=REG_CN)
    dataset = init_instance_by_config(config["task"]["dataset"])
    label = SignalRecord.generate_label(dataset)
    if label is None or label.empty:
        raise ValueError("Failed to generate labels from dataset.")

    model_cfg = config["task"]["model"]
    model = init_instance_by_config(model_cfg)
    fore_model, weight_model = build_models(model_cfg)
    device = model.device

    fore_state = torch.load(Path(args.fore_bin).expanduser().resolve(), map_location=device)
    weight_state = torch.load(Path(args.weight_bin).expanduser().resolve(), map_location=device)

    fore_model.load_state_dict(fore_state)
    weight_model.load_state_dict(weight_state)
    fore_model.to(device)
    weight_model.to(device)

    model.fore_model = fore_model
    model.weight_model = weight_model
    model.fitted = True

    pred = model.predict(dataset).to_frame("score")
    pred.index = pred.index.set_names(["datetime", "instrument"])

    if args.save_pred:
        save_path = Path(args.save_pred).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_pickle(save_path)
        print(f"Saved pred to {save_path}")

    label = label.loc[pred.index]
    sig = signal_metrics(pred, label)

    print("TCTS weight probe:")
    print(f"- config: {config_path}")
    print(f"- provider_uri: {provider_uri}")
    print(f"- fore_bin: {Path(args.fore_bin).resolve()}")
    print(f"- weight_bin: {Path(args.weight_bin).resolve()}")
    print()
    pprint(sig)


if __name__ == "__main__":
    main()
