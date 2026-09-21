"""Reassembly of overlapping prediction tiles into source-level rasters.

Extracted from ``tasks_with_models/change_detection_changeformer.py``.  When
tiling is used with overlap (``tile_size > tile_stride``), each source image
is covered by several overlapping tiles whose softmax probabilities must be
blended (rather than naively cropped) to avoid seam artefacts at tile
boundaries.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from rasterio.transform import Affine

from geo_deep_learning.utils.geotiff_merge import extract_scalar, parse_crs, transform_coeffs

logger = logging.getLogger(__name__)

# Metadata keys copied verbatim from the first tile of a group onto the
# assembled source-level result (order documents intent, not significance).
_SOURCE_METADATA_KEYS = (
    "cell_id", "pair_id", "event_id", "db_nbac_fire_id",
    "event_start_date", "event_end_date", "beam", "sat_pass", "output_name",
    "group_date_pre", "group_date_post", "group_id_pre", "group_id_post",
)


def create_blend_window(
    height: int,
    width: int,
    overlap_h: int,
    overlap_w: int,
) -> np.ndarray:
    """Create a 2D blending window with cosine ramps in overlap regions.

    Pixels in the non-overlapping center get weight 1.0.  Pixels in the
    overlap zone smoothly ramp from 0→1 using a raised-cosine profile,
    ensuring seamless transitions between adjacent tiles.

    Args:
        height: Tile height in pixels.
        width: Tile width in pixels.
        overlap_h: Vertical overlap in pixels (tile_h − stride_h).
        overlap_w: Horizontal overlap in pixels (tile_w − stride_w).

    Returns:
        2D ``float32`` array of shape ``[height, width]`` with values in ``(0, 1]``.
    """

    def _ramp(size: int, overlap: int) -> np.ndarray:
        win = np.ones(size, dtype=np.float32)
        if overlap > 0:
            ramp_vals = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
            ramp_vals = 0.5 * (1.0 - np.cos(np.pi * ramp_vals))
            win[:overlap] = ramp_vals
            win[-overlap:] = ramp_vals[::-1]
        return win

    win_h = _ramp(height, overlap_h)
    win_w = _ramp(width, overlap_w)
    window = np.outer(win_h, win_w)
    return np.maximum(window, 1e-6).astype(np.float32)


def build_source_profile(
    tile_info: dict[str, Any],
    source_h: int,
    source_w: int,
) -> dict[str, Any]:
    """Reconstruct the full source image's GeoTIFF profile from a tile's profile.

    Inverts the tile-level transform translation so the saved GeoTIFF
    covers the original spatial extent.
    """
    raw_profile = tile_info["profile_raw"]
    r = tile_info.get("tile_row_start", 0)
    c = tile_info.get("tile_col_start", 0)

    t_list = transform_coeffs(raw_profile["transform"], 0)

    # Undo tile translation: source_transform = tile_transform * translation(−c, −r)
    tile_transform = Affine(*t_list)
    source_transform = tile_transform * Affine.translation(-c, -r)

    return {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": 1,
        "nodata": 32767,
        "height": source_h,
        "width": source_w,
        "crs": parse_crs(raw_profile.get("crs")),
        "transform": source_transform,
    }


def _collect_tile_info(batch_result: dict[str, Any], index: int) -> dict[str, Any]:
    """Extract one sample's worth of tile metadata from a collated batch dict."""
    name = batch_result["pre_post_name"][index].replace("\n", "")
    tile_info: dict[str, Any] = {
        "name": name,
        "probabilities": batch_result["probabilities"][index].cpu(),
    }

    for dim_key in ("original_height", "original_width"):
        v = batch_result[dim_key]
        tile_info[dim_key] = v[index].item() if isinstance(v, torch.Tensor) else int(v[index])

    for key in ("tile_row_start", "tile_col_start", "source_height", "source_width"):
        if key in batch_result:
            v = batch_result[key]
            tile_info[key] = v[index].item() if isinstance(v, torch.Tensor) else int(v[index])

    for key in _SOURCE_METADATA_KEYS:
        if key in batch_result:
            tile_info[key] = extract_scalar(batch_result[key], index, default="unknown")

    # Profile (GeoTIFF) — extract per-sample values from the collated profile.
    # PyTorch's default_collate *transposes* a Python list of length N:
    # a list of 9-element transform lists becomes a list of 9 tensors each
    # of shape (batch_size,).  ``transform_coeffs`` normalizes all supported
    # shapes (transposed list, stacked tensor, dict) into a flat coefficient
    # list for a single sample.
    profile_raw: dict[str, Any] = {}
    for pk, pv in batch_result["profile"].items():
        if pk == "transform":
            profile_raw[pk] = transform_coeffs(pv, index)
        elif isinstance(pv, (list, tuple, torch.Tensor)):
            profile_raw[pk] = pv[index]
        else:
            profile_raw[pk] = pv
    tile_info["profile_raw"] = profile_raw

    if "mask_common" in batch_result:
        tile_info["mask_common"] = batch_result["mask_common"][index].cpu()
    if "water_mask" in batch_result:
        tile_info["water_mask"] = batch_result["water_mask"][index].cpu()

    return tile_info


def _blend_tile_group(
    tiles: list[dict[str, Any]],
    overlap_h: int,
    overlap_w: int,
    blend_window_cache: dict[tuple[int, int], np.ndarray],
    *,
    no_data_value: int,
) -> dict[str, Any]:
    """Blend one group of overlapping tiles into a single source-level prediction."""
    source_h = tiles[0].get("source_height", tiles[0]["original_height"])
    source_w = tiles[0].get("source_width", tiles[0]["original_width"])
    num_classes = tiles[0]["probabilities"].shape[0]

    prob_accum = np.zeros((num_classes, source_h, source_w), dtype=np.float64)
    weight_accum = np.zeros((source_h, source_w), dtype=np.float64)
    mask_accum = np.zeros((source_h, source_w), dtype=np.float32)
    water_accum = np.zeros((source_h, source_w), dtype=bool)

    for tile in tiles:
        r = tile.get("tile_row_start", 0)
        c = tile.get("tile_col_start", 0)
        th = tile["original_height"]
        tw = tile["original_width"]

        win_key = (th, tw)
        if win_key not in blend_window_cache:
            blend_window_cache[win_key] = create_blend_window(th, tw, overlap_h, overlap_w)
        win = blend_window_cache[win_key]

        probs = tile["probabilities"].numpy()[:, :th, :tw]
        prob_accum[:, r:r + th, c:c + tw] += probs * win[np.newaxis, :, :]
        weight_accum[r:r + th, c:c + tw] += win

        # Combine common masks (OR logic: valid in any tile = valid)
        if "mask_common" in tile:
            cm = tile["mask_common"].numpy()
            if cm.ndim == 3:  # noqa: PLR2004
                cm = cm.squeeze(0)
            mask_accum[r:r + th, c:c + tw] = np.maximum(mask_accum[r:r + th, c:c + tw], cm[:th, :tw])
        if "water_mask" in tile:
            wm = tile["water_mask"].numpy()
            if wm.ndim == 3:  # noqa: PLR2004
                wm = wm.squeeze(0)
            water_accum[r:r + th, c:c + tw] |= wm[:th, :tw] > 0

    weight_accum = np.maximum(weight_accum, 1e-8)
    blended_probs = (prob_accum / weight_accum[np.newaxis, :, :]).astype(np.float32)

    pred = np.argmax(blended_probs, axis=0).astype(np.uint16)
    invalid = (mask_accum < 0.5) | water_accum  # noqa: PLR2004
    pred[invalid] = no_data_value

    first_tile = tiles[0]
    return {
        "predictions": pred,
        "probabilities": blended_probs,
        "source_height": source_h,
        "source_width": source_w,
        "profile": build_source_profile(first_tile, source_h, source_w),
        "cell_id": first_tile.get("cell_id", "unknown"),
        "pair_id": first_tile.get("pair_id"),
        "event_id": first_tile.get("event_id", first_tile.get("db_nbac_fire_id", "unknown_event")),
        "event_start_date": first_tile.get("event_start_date"),
        "event_end_date": first_tile.get("event_end_date"),
        "beam": first_tile.get("beam"),
        "sat_pass": first_tile.get("sat_pass"),
        "output_name": first_tile.get("output_name"),
        "group_id_pre": first_tile.get("group_id_pre", "all"),
        "group_id_post": first_tile.get("group_id_post", "all"),
        "group_date_pre": first_tile.get("group_date_pre", "all"),
        "group_date_post": first_tile.get("group_date_post", "all"),
    }


def reassemble_overlapping_tiles(
    predictions: list[dict[str, Any]],
    tile_size: tuple[int, int] | None,
    tile_stride: tuple[int, int] | None,
    *,
    no_data_value: int,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Reassemble overlapping tiles into source-level predictions with cosine blending.

    Detects whether tiling with overlap was used.  If so, groups tiles by
    source image (using ``pre_post_name`` minus the tile suffix) and blends
    their softmax probabilities with a 2D cosine window to eliminate tile
    seam artefacts.

    Args:
        predictions: List of batch prediction dicts from ``predict_step``.
        tile_size: ``(tile_h, tile_w)`` used by the datamodule, or ``None``.
        tile_stride: ``(stride_h, stride_w)`` used by the datamodule, or ``None``.
        no_data_value: Value written to pixels invalid in every contributing tile.

    Returns:
        Tuple of:
        - Dict mapping *source_key* → assembled prediction info (numpy arrays,
          GeoTIFF profile, metadata scalars), keyed by ``pre_post_name``.
        - ``True`` if overlap blending was applied, ``False`` otherwise.
    """
    has_tiles = any("tile_row_start" in batch for batch in predictions)
    if not has_tiles or tile_size is None or tile_stride is None:
        return {}, False

    tile_h, tile_w = tile_size
    stride_h, stride_w = tile_stride
    overlap_h = max(tile_h - stride_h, 0)
    overlap_w = max(tile_w - stride_w, 0)

    if overlap_h <= 0 and overlap_w <= 0:
        return {}, False  # No overlap → skip blending

    logger.info(
        "Overlap detected: tile=%s, stride=%s, overlap=(%d, %d). Reassembling tiles with cosine blending…",
        tile_size, tile_stride, overlap_h, overlap_w,
    )

    # --- Group tiles by source image ---
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch_result in predictions:
        names = batch_result["pre_post_name"]
        for i in range(len(names)):
            name = names[i].replace("\n", "")
            source_key = name.split("|tile_")[0] if "|tile_" in name else name
            groups[source_key].append(_collect_tile_info(batch_result, i))

    # --- Reassemble each source image ---
    assembled: dict[str, dict[str, Any]] = {}
    blend_window_cache: dict[tuple[int, int], np.ndarray] = {}

    for source_key, tiles in groups.items():
        # Single-tile source without tile metadata → leave for per-tile path
        if len(tiles) == 1 and "tile_row_start" not in tiles[0]:
            continue

        assembled_source = _blend_tile_group(
            tiles, overlap_h, overlap_w, blend_window_cache, no_data_value=no_data_value,
        )
        assembled_source["pre_post_name"] = source_key
        assembled[source_key] = assembled_source

    logger.info("Reassembled %d source images from overlapping tiles.", len(assembled))
    return assembled, bool(assembled)
