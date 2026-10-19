"""Tests for overlap-blended tile reassembly (``utils/tile_reassembly.py``)."""

import numpy as np
import torch

from geo_deep_learning.utils.tile_reassembly import _blend_tile_group

NO_DATA = 32767
PROB_NODATA = -1.0


def _tile(r: int, c: int, size: int, valid: torch.Tensor, source: int, buf: int, cell: int) -> dict:
    probs = torch.full((2, size, size), 0.5)
    probs[1] = 0.3  # garbage-like probability everywhere, incl. invalid pixels
    probs[0] = 0.7
    return {
        "name": "src",
        "probabilities": probs,
        "original_height": size,
        "original_width": size,
        "tile_row_start": r,
        "tile_col_start": c,
        "source_height": source,
        "source_width": source,
        "buffer_size": buf,
        "cell_orig_height": cell,
        "cell_orig_width": cell,
        "mask_common": valid[r:r + size, c:c + size].unsqueeze(0).float(),
        "profile_raw": {"transform": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0], "crs": "EPSG:3979"},
    }


def test_probability_is_nodata_where_prediction_is_nodata():
    buf, cell, size = 4, 12, 16
    source = cell + 2 * buf  # 20 → tiles at origins {0, 4}
    valid = torch.ones((source, source), dtype=torch.bool)
    valid[buf + cell // 2:, :] = False  # bottom half of the cell is invalid

    tiles = [_tile(r, c, size, valid, source, buf, cell) for r in (0, 4) for c in (0, 4)]
    out = _blend_tile_group(
        tiles, overlap_h=12, overlap_w=12, blend_window_cache={},
        no_data_value=NO_DATA, positive_class_index=1, probability_nodata=PROB_NODATA,
    )

    pred, prob = out["predictions"], out["probability"]
    assert pred.shape == prob.shape == (cell, cell)
    invalid = pred == NO_DATA
    assert invalid[cell // 2:].all() and not invalid[:cell // 2].any()
    assert np.all(prob[invalid] == PROB_NODATA)
    np.testing.assert_allclose(prob[~invalid], 0.3, rtol=1e-5)


def test_water_and_ground_truth_ignore_become_nodata_after_blending():
    size = 8
    valid = torch.ones((size, size), dtype=torch.bool)
    tile = _tile(0, 0, size, valid, source=size, buf=0, cell=size)
    water = torch.zeros((1, size, size))
    water[0, :2, :] = 1.0  # top rows are water
    ignore = torch.zeros((1, size, size), dtype=torch.bool)
    ignore[0, :, -2:] = True  # right columns are ignored by the ground truth
    tile["water_mask"], tile["ignore_mask"] = water, ignore

    out = _blend_tile_group(
        [tile], overlap_h=4, overlap_w=4, blend_window_cache={},
        no_data_value=NO_DATA, positive_class_index=1, probability_nodata=PROB_NODATA,
    )

    expected = np.zeros((size, size), dtype=bool)
    expected[:2, :] = True
    expected[:, -2:] = True
    np.testing.assert_array_equal(out["predictions"] == NO_DATA, expected)
    np.testing.assert_array_equal(out["probability"] == PROB_NODATA, expected)
