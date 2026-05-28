from __future__ import annotations

import argparse
from pathlib import Path

import qlib
from qlib.constant import REG_CN
from qlib.utils import flatten_dict, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord
from ruamel.yaml import YAML


def load_config(config_path: Path) -> dict:
    yaml = YAML(typ="safe", pure=True)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Qlib workflow.")
    parser.add_argument(
        "--config",
        default="src/configs/workflow_config_lightgbm_a_share.yaml",
        help="Path to the Qlib workflow yaml config.",
    )
    parser.add_argument(
        "--experiment-name",
        default="lhai_ashare_baseline",
        help="Recorder experiment name.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    qlib_init = config["qlib_init"]
    provider_uri = str(Path(qlib_init["provider_uri"]).expanduser())
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    task = config["task"]
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    with R.start(experiment_name=args.experiment_name):
        R.log_params(config_path=str(config_path), **flatten_dict(config))
        model.fit(dataset)

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        sar = SigAnaRecord(recorder)
        sar.generate()

        port_cfg = config["port_analysis_config"]
        par = PortAnaRecord(recorder, port_cfg, "day")
        par.generate()


if __name__ == "__main__":
    main()
