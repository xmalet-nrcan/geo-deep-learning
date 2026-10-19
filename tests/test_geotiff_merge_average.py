"""``merge_average`` must keep NO_DATA (masked / uncovered pixels) as NO_DATA."""

from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.transform import Affine

from geo_deep_learning.utils.geotiff_merge import chunked_merge

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
