"""Tests for the temporal-confirmation ``merged_all`` mosaic (``utils/geotiff_merge.py``)."""

from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import Affine

from geo_deep_learning.utils.geotiff_merge import (
    CLASS_NODATA,
    PROBABILITY_NODATA,
    merge_predictions,
    merged_all_filename,
    probability_path,
    temporal_confirmation_merge,
)

PIXEL = 10.0
SIZE = 2


def _write_pair(path: Path, prob: list[float], x0: float = 0.0, *, threshold: float = 0.5) -> Path:
    """Write a 1x2 class raster + ``_prob`` sibling; ``-1`` = nodata pixel."""
    p = np.array([prob], dtype=np.float32)
    cls = np.where(p < 0, CLASS_NODATA, p >= threshold).astype(np.uint16)
    common = {
        "driver": "GTiff", "count": 1, "height": 1, "width": SIZE, "crs": "EPSG:3979",
        "transform": Affine(PIXEL, 0, x0, 0, -PIXEL, 100.0),
    }
    with rio.open(path, "w", **common, dtype="uint16", nodata=CLASS_NODATA) as dst:
        dst.write(cls[np.newaxis])
    with rio.open(probability_path(path), "w", **common, dtype="float32", nodata=PROBABILITY_NODATA) as dst:
        dst.write(p[np.newaxis])
    return path


def _read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rio.open(path) as src:
        cls = src.read(1)
    with rio.open(probability_path(path)) as src:
        prob = src.read(1)
    return cls, prob


def test_early_negative_pair_does_not_erase_later_detections(tmp_path: Path) -> None:
    # Pixel 0: early pair says 0 (fire not there yet), 3 later pairs detect → burned.
    # Pixel 1: a single isolated detection among 4 pairs → not burned.
    pairs = [
        [_write_pair(tmp_path / "d1.tif", [0.1, 0.2])],
        [_write_pair(tmp_path / "d2.tif", [0.8, 0.9])],
        [_write_pair(tmp_path / "d3.tif", [0.7, 0.1])],
        [_write_pair(tmp_path / "d4.tif", [0.9, 0.1])],
    ]
    out = tmp_path / "merged.tif"
    temporal_confirmation_merge(pairs, out, min_pairs=3, prob_threshold=0.5)

    cls, prob = _read(out)
    assert cls.tolist() == [[1, 0]]
    # persistence probability: mean from first detection onward
    np.testing.assert_allclose(prob[0, 0], (0.8 + 0.7 + 0.9) / 3, rtol=1e-6)
    np.testing.assert_allclose(prob[0, 1], (0.9 + 0.1 + 0.1) / 3, rtol=1e-6)


def test_required_pairs_capped_by_valid_pairs_and_nodata(tmp_path: Path) -> None:
    # Pixel 0 valid in one pair only (detected) → burned despite min_pairs=3.
    # Pixel 1 never valid → nodata in both outputs.
    pairs = [
        [_write_pair(tmp_path / "d1.tif", [-1.0, -1.0])],
        [_write_pair(tmp_path / "d2.tif", [0.9, -1.0])],
    ]
    out = tmp_path / "merged.tif"
    temporal_confirmation_merge(pairs, out, min_pairs=3)

    cls, prob = _read(out)
    assert cls.tolist() == [[1, CLASS_NODATA]]
    assert prob[0, 1] == PROBABILITY_NODATA


def test_persistence_probability_can_veto_counted_detections(tmp_path: Path) -> None:
    # 3 detections just above threshold, then 3 strong negatives → mean < 0.5.
    probs = [0.55, 0.55, 0.55, 0.0, 0.0, 0.0]
    pairs = [[_write_pair(tmp_path / f"d{i}.tif", [v, 0.0])] for i, v in enumerate(probs)]
    out = tmp_path / "merged.tif"
    temporal_confirmation_merge(pairs, out, min_pairs=3)
    cls, _ = _read(out)
    assert cls[0, 0] == 0


def test_cells_are_placed_on_common_grid(tmp_path: Path) -> None:
    a = _write_pair(tmp_path / "a.tif", [0.9, 0.9], x0=0.0)
    b = _write_pair(tmp_path / "b.tif", [0.1, 0.1], x0=SIZE * PIXEL)
    out = tmp_path / "merged.tif"
    temporal_confirmation_merge([[a, b]], out, min_pairs=1)
    cls, _ = _read(out)
    assert cls.tolist() == [[1, 1, 0, 0]]


@pytest.mark.parametrize(("beam", "sat_pass"), [("A", "ASC"), ("A", "DESC")])
def test_merge_predictions_writes_one_mosaic_per_beam_and_pass(
    tmp_path: Path, beam: str, sat_pass: str,
) -> None:
    def key(post_date: str, s_pass: str) -> tuple[str, ...]:
        return (str(tmp_path), "42", "2025-07-01", "2025-07-25", "1", "2025-06-20",
                post_date, post_date, beam, s_pass)

    asc = [_write_pair(tmp_path / f"asc{i}.tif", [0.9, 0.9]) for i in range(3)]
    desc = [_write_pair(tmp_path / f"desc{i}.tif", [0.1, 0.1]) for i in range(2)]
    groups = {key(f"2025-07-1{i}", "ASC"): [p] for i, p in enumerate(asc)}
    groups |= {key(f"2025-07-2{i}", "DESC"): [p] for i, p in enumerate(desc)}

    merge_predictions(groups, {str(tmp_path): asc + desc}, merged_all_min_pairs=3)

    cls, _ = _read(tmp_path / merged_all_filename(beam, sat_pass))
    assert cls.tolist() == ([[1, 1]] if sat_pass == "ASC" else [[0, 0]])
    assert not (tmp_path / "merged_all.tif").exists()
