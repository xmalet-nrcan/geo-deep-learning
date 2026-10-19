"""``merge_average`` must keep NO_DATA (masked / uncovered pixels) as NO_DATA."""

from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.transform import Affine

from geo_deep_learning.utils.geotiff_merge import (
    chunked_merge,
    merge_predictions,
    merge_probability_rasters,
    merged_group_filename,
    probability_path,
)

NO_DATA = 32767
SIZE = 4


def _write(path: Path, arr: np.ndarray, col: int) -> Path:
    profile = {
        "driver": "GTiff", "dtype": "uint16", "count": 1, "height": SIZE, "width": SIZE,
        "nodata": NO_DATA, "crs": "EPSG:3979",
        "transform": Affine(100, 0, col * SIZE * 100, 0, -100, 1000),
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr[None].astype("uint16"))
    return path


def _read(path: Path) -> np.ndarray:
    with rio.open(path) as src:
        return src.read(1)


def test_masked_pixels_stay_nodata_across_adjacent_cells(tmp_path: Path) -> None:
    a = np.ones((SIZE, SIZE))
    a[1, 1] = a[2, 2] = NO_DATA  # e.g. water
    b = np.zeros((SIZE, SIZE))  # valid "unburned" cell next to it
    out = tmp_path / "merged.tif"
    chunked_merge([_write(tmp_path / "a.tif", a, 0), _write(tmp_path / "b.tif", b, 1)], out)

    merged = _read(out)
    np.testing.assert_array_equal(merged[:, :SIZE], a.astype("uint16"))  # NO_DATA kept, 1 kept
    np.testing.assert_array_equal(merged[:, SIZE:], 0)  # valid class 0 is not nodata


def test_coverage_gap_is_nodata(tmp_path: Path) -> None:
    a, e = np.ones((SIZE, SIZE)), np.ones((SIZE, SIZE))
    out = tmp_path / "merged.tif"
    chunked_merge([_write(tmp_path / "a.tif", a, 0), _write(tmp_path / "e.tif", e, 3)], out)

    merged = _read(out)
    assert merged.shape[1] == 4 * SIZE
    assert (merged[:, SIZE:3 * SIZE] == NO_DATA).all()


def test_overlap_still_averages_valid_pixels(tmp_path: Path) -> None:
    a = np.ones((SIZE, SIZE))
    a[0, 0] = NO_DATA
    c = np.zeros((SIZE, SIZE))
    out = tmp_path / "merged.tif"
    chunked_merge([_write(tmp_path / "a.tif", a, 0), _write(tmp_path / "c.tif", c, 0)], out)

    merged = _read(out)
    assert merged[0, 0] == 0  # only the valid tile contributes
    assert merged[1, 1] == 1  # (1 + 0) / 2 = 0.5 rounds half-up to the higher class


# ---------------------------------------------------------------------------
# Merged probability rasters (``merge_predictions`` pass 1)
# ---------------------------------------------------------------------------

PROB_NODATA = -1.0


def _write_cell(path: Path, cls: np.ndarray, prob: np.ndarray, col: int) -> Path:
    """Class raster ``path`` + its ``_prob.tif`` sibling."""
    _write(path, cls, col)
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1, "height": SIZE, "width": SIZE,
        "nodata": PROB_NODATA, "crs": "EPSG:3979",
        "transform": Affine(100, 0, col * SIZE * 100, 0, -100, 1000),
    }
    with rio.open(probability_path(path), "w", **profile) as dst:
        dst.write(prob[None].astype("float32"))
    return path


def test_probability_merge_averages_without_filtering(tmp_path: Path) -> None:
    water = np.full((SIZE, SIZE), 0.9)
    water[1, 1] = PROB_NODATA  # masked pixel: no probability
    cls_a = np.where(water == PROB_NODATA, NO_DATA, 1)
    a = _write_cell(tmp_path / "a.tif", cls_a, water, 0)
    b = _write_cell(tmp_path / "b.tif", np.zeros((SIZE, SIZE)), np.full((SIZE, SIZE), 0.05), 1)
    c = _write_cell(tmp_path / "c.tif", np.ones((SIZE, SIZE)), np.full((SIZE, SIZE), 0.3), 0)  # overlaps a

    out = tmp_path / "merged.tif"
    prob_out = merge_probability_rasters([a, b, c], out)

    assert prob_out == probability_path(out)
    with rio.open(prob_out) as src:
        assert src.dtypes[0] == "float32" and src.nodata == PROB_NODATA
        prob = src.read(1)
    assert prob.shape == (SIZE, 2 * SIZE)
    np.testing.assert_allclose(prob[:, SIZE:], 0.05, rtol=1e-5)  # low probabilities are kept (no threshold)
    assert prob[0, 0] == np.float32((0.9 + 0.3) / 2)  # overlap: mean of both
    assert prob[1, 1] == np.float32(0.3)  # masked in a, only c contributes


def test_probability_merge_keeps_masked_pixels_nodata(tmp_path: Path) -> None:
    prob_a = np.full((SIZE, SIZE), 0.8)
    prob_a[2, 2] = PROB_NODATA
    a = _write_cell(tmp_path / "a.tif", np.where(prob_a < 0, NO_DATA, 1), prob_a, 0)
    b = _write_cell(tmp_path / "b.tif", np.ones((SIZE, SIZE)), np.full((SIZE, SIZE), 0.8), 1)

    prob_out = merge_probability_rasters([a, b], tmp_path / "merged.tif")
    with rio.open(prob_out) as src:
        prob = src.read(1)
    assert prob[2, 2] == PROB_NODATA
    assert (prob[prob != PROB_NODATA] > 0).all()


def test_probability_merge_is_skipped_when_a_tile_has_no_probability(tmp_path: Path) -> None:
    a = _write_cell(tmp_path / "a.tif", np.ones((SIZE, SIZE)), np.full((SIZE, SIZE), 0.8), 0)
    b = _write(tmp_path / "b.tif", np.ones((SIZE, SIZE)), 1)  # class only

    out = tmp_path / "merged.tif"
    assert merge_probability_rasters([a, b], out) is None
    assert not probability_path(out).exists()


def test_merge_predictions_writes_the_merged_probability(tmp_path: Path) -> None:
    a = _write_cell(tmp_path / "a.tif", np.ones((SIZE, SIZE)), np.full((SIZE, SIZE), 0.7), 0)
    b = _write_cell(tmp_path / "b.tif", np.zeros((SIZE, SIZE)), np.full((SIZE, SIZE), 0.1), 1)
    key = (str(tmp_path), "42", "20250701", "20250725", "1", "20250620", "3", "20250710", "A", "ASC")

    merge_predictions({key: [a, b]}, {str(tmp_path): [a, b]}, write_merged_all=False)

    merged = merged_group_filename("42", "20250701", "20250725", "1", "20250620", "3", "20250710", "A", "ASC")
    assert (tmp_path / merged).exists()
    prob = _read(probability_path(tmp_path / merged))
    assert prob.shape == (SIZE, 2 * SIZE)
