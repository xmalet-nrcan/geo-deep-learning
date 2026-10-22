"""Band sweep config tests; no torch, Lightning, Docker or GPU required."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from scripts.generate_band_sweep_configs import ALL9, generate_configs


BASE = Path(__file__).resolve().parents[1] / "configs/rcm_change_detection_changestar_farseg.yaml"
EXPECTED = [
    ["LOCALINCANGLE", "RR", "RL"],
    ["LOCALINCANGLE", "PDN", "PSN", "PVN", "S0"],
    ["LOCALINCANGLE", "PDN", "PSN", "PVN", "S0", "RR", "RL"],
]


class BandSweepConfigTests(unittest.TestCase):
    def test_cli_manifest_uses_lf_and_lists_existing_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(BASE.parents[1] / "scripts/generate_band_sweep_configs.py"),
                 str(BASE), tmp, "--run-id", "cli_test"],
                check=True, capture_output=True,
            )
            self.assertNotIn(b"\r", result.stdout)
            paths = result.stdout.decode("utf-8").splitlines()
            self.assertEqual(len(paths), len(EXPECTED))
            self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_default_subsets_preserve_base_and_disable_checkpoint_loads(self):
        original = BASE.read_bytes()
        base = yaml.safe_load(original)
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_configs(BASE, Path(tmp), "test_run")
            self.assertEqual(len(paths), len(EXPECTED))
            for path, bands in zip(paths, EXPECTED):
                cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(cfg["data"]["init_args"]["band_names"], bands)
                self.assertNotIn("in_channels", cfg["model"]["init_args"])
                self.assertIsNone(cfg["model"]["init_args"]["weights_from_checkpoint_path"])
                self.assertIsNone(cfg["ckpt_path"])

                # All settings outside the intentional sweep overrides stay identical.
                restored = deepcopy(cfg)
                restored["data"]["init_args"]["band_names"] = base["data"]["init_args"]["band_names"]
                restored["model"]["init_args"]["weights_from_checkpoint_path"] = (
                    base["model"]["init_args"]["weights_from_checkpoint_path"]
                )
                restored.pop("ckpt_path")
                restored["trainer"]["logger"] = base["trainer"]["logger"]
                restored["trainer"]["callbacks"] = base["trainer"]["callbacks"]
                restored["trainer"].pop("default_root_dir")
                self.assertEqual(restored, base)
        self.assertEqual(BASE.read_bytes(), original)

    def test_paths_isolated_between_combos_and_sweeps(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_configs(BASE, Path(tmp) / "first", "first")
            paths += generate_configs(BASE, Path(tmp) / "second", "second")
            configs = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in paths]
            for getter in (
                lambda c: c["trainer"]["logger"]["init_args"]["name"],
                lambda c: c["trainer"]["default_root_dir"],
                lambda c: next(cb["init_args"]["dirpath"] for cb in c["trainer"]["callbacks"]
                               if cb["class_path"].endswith(".ModelCheckpoint")),
            ):
                self.assertEqual(len({getter(c) for c in configs}), 2 * len(EXPECTED))
            with self.assertRaises(FileExistsError):
                generate_configs(BASE, Path(tmp) / "first", "first")

    def test_all9_requires_opt_in_and_also_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_configs(BASE, Path(tmp), "all9_test", include_all9=True)
            self.assertEqual(len(paths), len(EXPECTED) + 1)
            cfg = yaml.safe_load(paths[-1].read_text(encoding="utf-8"))
            self.assertEqual(cfg["data"]["init_args"]["band_names"], ALL9)
            self.assertNotIn("in_channels", cfg["model"]["init_args"])
            self.assertIsNone(cfg["model"]["init_args"]["weights_from_checkpoint_path"])

    def test_explicit_resume_cleared_and_legacy_channels_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
            base["ckpt_path"] = "last"
            base["model"]["init_args"]["in_channels"] = 10
            source = Path(tmp) / "base.yaml"
            source.write_text(yaml.safe_dump(base), encoding="utf-8")
            paths = generate_configs(source, Path(tmp) / "out", "resume_test", include_all9=True)
            for path in paths:
                cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIsNone(cfg["ckpt_path"])
                self.assertNotIn("in_channels", cfg["model"]["init_args"])

    def test_unsupported_metadata_layout_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
            base["data"]["init_args"]["separate_metadata"] = False
            source = Path(tmp) / "base.yaml"
            source.write_text(yaml.safe_dump(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "separate_metadata"):
                generate_configs(source, Path(tmp) / "out", "invalid")


if __name__ == "__main__":
    unittest.main()