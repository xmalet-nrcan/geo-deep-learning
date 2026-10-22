#!/usr/bin/env python3
"""Generate isolated Lightning configs for run_band_sweep.sh (no training imports)."""

import argparse
from copy import deepcopy
from pathlib import Path, PurePosixPath
import re
import sys

import yaml


BAND_COMBINATIONS = {
    "intensity": ["LOCALINCANGLE", "RR", "RL"],
    "mchi": ["LOCALINCANGLE", "PDN", "PSN", "PVN", "S0"],
    "mchi_intensity": ["LOCALINCANGLE", "PDN", "PSN", "PVN", "S0", "RR", "RL"],
}
ALL9 = ["LOCALINCANGLE", "PDN", "PSN", "PVN", "S0", "NDSV", "RFDI", "RR", "RL"]


def generate_configs(base_path: Path, output_dir: Path, run_id: str,
                     include_all9: bool = False) -> list[Path]:
    """Preserve base settings except bands, initial checkpoint and output identity."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must contain only letters, digits, '_' or '-'")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if base["data"]["init_args"].get("separate_metadata") is not True:
        raise ValueError("Band sweep requires separate_metadata: true (channels = bands + 1)")
    trainer = base["trainer"]
    if trainer["logger"]["class_path"] != "lightning.pytorch.loggers.TensorBoardLogger":
        raise ValueError("Band sweep requires TensorBoardLogger for isolated run names")
    checkpoints = [cb for cb in trainer["callbacks"]
                   if cb["class_path"] == "lightning.pytorch.callbacks.ModelCheckpoint"]
    if not checkpoints or any(not cb.get("init_args", {}).get("dirpath") for cb in checkpoints):
        raise ValueError("Band sweep requires ModelCheckpoint with explicit dirpath")

    combinations = dict(BAND_COMBINATIONS)
    if include_all9:
        combinations["all9"] = ALL9
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, bands in combinations.items():
        config = deepcopy(base)
        config["data"]["init_args"]["band_names"] = list(bands)
        model = config["model"]["init_args"]
        # configure_model infers channels from the DataModule's selected bands.
        # Drop legacy overrides inherited from the base YAML as well.
        model.pop("in_channels", None)
        # strict=False does NOT allow tensor shape mismatches. Never transfer All9
        # weights implicitly, nor restore optimizer state through Lightning ckpt_path.
        model["weights_from_checkpoint_path"] = None
        config["ckpt_path"] = None

        suffix = PurePosixPath("band_sweep", run_id, name)
        trainer = config["trainer"]
        logger = trainer["logger"]["init_args"]
        logger["name"] = str(PurePosixPath(logger.get("name") or "default") / suffix)
        # Avoid reusing an explicitly pinned logger version from the base.
        logger["version"] = None
        trainer["default_root_dir"] = str(PurePosixPath("/app/lightning_logs") / suffix)
        for cb in trainer["callbacks"]:
            if cb["class_path"] == "lightning.pytorch.callbacks.ModelCheckpoint":
                args = cb["init_args"]
                args["dirpath"] = str(PurePosixPath(args["dirpath"]) / suffix)
                args["filename"] = name + "-" + (args.get("filename") or "{epoch}-{step}")

        path = output_dir / f"{base_path.stem}_{name}.yaml"
        # Refuse to overwrite a previous sweep's configs.
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--include-all9", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(newline="\n")  # Manifest consumed by Bash, also on Windows.
    for path in generate_configs(args.base, args.output_dir, args.run_id, args.include_all9):
        print(path.as_posix())


if __name__ == "__main__":
    main()