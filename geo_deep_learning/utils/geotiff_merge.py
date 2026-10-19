"""Helpers to name, write and merge GeoTIFF prediction tiles.

Extracted from ``tasks_with_models/change_detection_changeformer.py`` so the
(fairly generic) raster I/O and merging logic can be unit-tested and reused
independently from the Lightning module.
"""

from __future__ import annotations

import gc
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
import torch
from rasterio import MemoryFile
from rasterio.crs import CRS as RioCRS
from rasterio.merge import merge as rio_merge
from rasterio.transform import Affine

logger = logging.getLogger(__name__)

DEFAULT_EPSG = 3979


def parse_crs(crs_val: Any, default_epsg: int = DEFAULT_EPSG) -> RioCRS:
    """Parse a CRS value, falling back to ``default_epsg`` when empty/invalid."""
    try:
        return RioCRS.from_user_input(crs_val) if crs_val else RioCRS.from_epsg(default_epsg)
    except Exception:
        logger.warning("Could not parse CRS %r — using default EPSG:%d.", crs_val, default_epsg)
        return RioCRS.from_epsg(default_epsg)


def transform_coeffs(transform_raw: Any, index: int) -> list[float]:
    """Extract the 6 affine-transform coefficients for sample ``index``.

    ``transform_raw`` may come back from PyTorch's default collate in
    several shapes depending on how it was produced upstream:
      - a list/tuple of length >= 6 whose elements are per-sample tensors
        (or scalars) — i.e. the *transposed* representation produced when
        collating a Python list of 6/9-element sequences.
      - a stacked tensor / numpy array of shape ``(batch_size, >=6)``.
      - a dict keyed by ``0..5`` or by ``"a".."f"``.
    """
    if isinstance(transform_raw, dict):
        try:
            raw = [transform_raw[k] for k in range(6)]
        except KeyError:
            raw = [transform_raw.get(k, 0.0) for k in ("a", "b", "c", "d", "e", "f")]
    elif isinstance(transform_raw, (list, tuple)):
        first = transform_raw[0]
        if isinstance(first, (torch.Tensor, np.ndarray, list, tuple)):
            # Transposed: transform_raw[k] holds the k-th coefficient for all samples.
            raw = [transform_raw[k][index] for k in range(6)]
        else:
            # Already a flat 6-element sequence for a single sample.
            raw = list(transform_raw[:6])
    elif isinstance(transform_raw, torch.Tensor):
        raw = transform_raw[index].tolist() if transform_raw.dim() > 1 else transform_raw.tolist()
    else:
        raw = list(transform_raw)

    coeffs = [t.item() if isinstance(t, torch.Tensor) else float(t) for t in raw]
    if len(coeffs) < 6:
        msg = (
            f"Cannot reconstruct transform: only {len(coeffs)} coefficient(s) found "
            f"— need >= 6. type={type(transform_raw).__name__!r}, raw={transform_raw!r}"
        )
        raise ValueError(msg)
    return coeffs[:6]


def extract_scalar(batch_field: Any, index: int, default: str | None = "unknown") -> str | None:
    """Extract a scalar string value from a batched field at position ``index``."""
    if batch_field is None:
        return default
    if isinstance(batch_field, torch.Tensor):
        return str(batch_field[index].item())
    if isinstance(batch_field, (list, tuple)):
        return str(batch_field[index])
    return str(batch_field)


def merge_date(value: Any) -> str:
    """Normalize a date-like metadata value to ``YYYYMMDD`` for filenames."""
    value_as_string = str(value).strip()
    digits = "".join(char for char in value_as_string if char.isdigit())
    return digits[:8] if len(digits) >= 8 else "NA"


def group_merge_key(  # noqa: PLR0913
        event_date_dir: str,
        event_id: Any,
        event_start_date: Any,
        event_end_date: Any,
        group_id_pre: Any,
        group_date_pre: Any,
        group_id_post: Any,
        group_date_post: Any,
        beam: Any,
        sat_pass: Any,
) -> tuple[str, ...]:
    """Return the complete provenance key for one cross-cell merge."""
    return tuple(map(str, (
        event_date_dir,
        event_id,
        event_start_date,
        event_end_date,
        group_id_pre,
        group_date_pre,
        group_id_post,
        group_date_post,
        beam,
        sat_pass,
    )))


def merged_group_filename(  # noqa: PLR0913
        event_id: str,
        event_start_date: str,
        event_end_date: str,
        group_id_pre: str,
        group_date_pre: str,
        group_id_post: str,
        group_date_post: str,
        beam: str,
        sat_pass: str,
) -> str:
    """Build the cross-cell merge name without a ``cell_id`` component."""
    return (
        f"event-{event_id}"
        f"_start-{merge_date(event_start_date)}"
        f"_end_{merge_date(event_end_date)}"
        f"_beam-{beam}_pass-{sat_pass}"
        f"_pre-{merge_date(group_date_pre)}-g{group_id_pre}"
        f"_post-{merge_date(group_date_post)}-g{group_id_post}"
        ".tif"

    )


def prediction_output_filename(
        output_name: str | None,
        *,
        pair_id: str | None,
        legacy_name: str,
        suffix: str = "",
) -> str:
    """Return a safe GeoTIFF filename, preferring the CSV output name.

    ``output_name`` is supplied by the SCANFIRE orchestrator and contains
    the event, group dates/IDs, cell, beam, and satellite pass.  The
    legacy fallback preserves compatibility with older prediction CSVs.
    """
    if output_name:
        requested = Path(str(output_name)).name
        if requested.lower().endswith(".tif") and requested != ".tif":
            return f"{Path(requested).stem}{suffix}.tif"

    return f"{pair_id}-{legacy_name}{suffix}.tif" if pair_id else f"{legacy_name}{suffix}.tif"


def safe_merge(datasets: list[Any], method: str = "average") -> tuple[np.ndarray, Affine]:
    """Merge raster datasets with averaging support for all rasterio versions.

    Standard GeoTIFFs are north-up (pixel height < 0). Some rasterio versions
    (e.g. 1.4.0) raise MergeError for these. Workaround: flip to positive pixel
    height in memory, merge, then flip the result back.

    When ``method='average'``, overlapping valid pixels are averaged using
    sum/count (compatible with all rasterio versions, since ``'average'``
    was only added in rasterio ≥ 1.4.x).

    Args:
        datasets: List of rasterio dataset readers to merge.
        method: ``'average'`` (default) averages valid (non-nodata) pixels.
            Any other value (``'first'``, ``'last'``, ``'min'``, ``'max'``)
            is passed directly to ``rasterio.merge``.
    """
    if method == "average":
        return merge_average(datasets)

    try:
        return rio_merge(datasets, method=method)
    except Exception as e:
        if "negative pixel height" not in str(e):
            raise
        return merge_with_flip(datasets, method=method)


def merge_average(datasets: list[Any]) -> tuple[np.ndarray, Affine]:
    """Merge datasets by averaging overlapping valid pixels.

    Uses two passes of ``rasterio.merge`` with ``method='sum'`` and
    ``method='count'`` to compute the average.  Falls back to the
    flip workaround if the rasterio version rejects negative pixel height.
    """
    try:
        mosaic_sum, transform = rio_merge(datasets, method="sum")
        mosaic_count, _ = rio_merge(datasets, method="count")
    except Exception as e:
        if "negative pixel height" not in str(e):
            raise
        mosaic_sum, transform = merge_with_flip(datasets, method="sum")
        mosaic_count, _ = merge_with_flip(datasets, method="count")

    # Average: sum / count, avoiding division by zero
    mask_no_coverage = mosaic_count == 0
    mosaic_count_safe = mosaic_count.astype(np.float64)
    mosaic_count_safe[mask_no_coverage] = 1.0
    mosaic = mosaic_sum.astype(np.float64) / mosaic_count_safe

    # Integer rasters hold class labels: round half-up (majority vote, ties →
    # higher class) instead of letting astype() truncate, which kept a pixel
    # burned only when *every* overlapping pair agreed (5/7 = 0.71 → 0).
    if np.issubdtype(np.dtype(datasets[0].dtypes[0]), np.integer):
        mosaic = np.floor(mosaic + 0.5)

    # Restore nodata where no tile contributed
    nodata = datasets[0].nodata
    if nodata is not None:
        mosaic[mask_no_coverage] = nodata

    mosaic = mosaic.astype(datasets[0].dtypes[0])
    return mosaic, transform


def merge_with_flip(datasets: list[Any], method: str = "first") -> tuple[np.ndarray, Affine]:
    """Merge datasets after flipping to positive pixel height.

    Workaround for rasterio, which rejects "upside down" rasters (pixel
    height ``transform.e > 0``) in :func:`rasterio.merge.merge`.  Such
    rasters are flipped vertically to north-up (negative ``e``) in memory,
    merged, then flipped back to preserve the original orientation.
    """
    mem_files = []
    flipped_datasets = []
    needs_flip = False

    for ds in datasets:
        # rasterio rejects rasters whose pixel height is POSITIVE
        # (``transform.e > 0`` → "upside down"). Flip exactly those to
        # north-up (negative ``e``) so the merge is accepted.
        if ds.transform.e > 0:
            needs_flip = True
            data = ds.read()[:, ::-1, :]  # flip vertically
            new_transform = Affine(
                ds.transform.a, ds.transform.b, ds.transform.c,
                ds.transform.d, -ds.transform.e,
                ds.transform.f + ds.transform.e * ds.height,
            )
            profile = ds.profile.copy()
            profile["transform"] = new_transform
            memfile = MemoryFile()
            with memfile.open(**profile) as mem_dst:
                mem_dst.write(data)
            flipped_datasets.append(memfile.open())
            mem_files.append(memfile)
        else:
            flipped_datasets.append(ds)

    mosaic, mosaic_transform = rio_merge(flipped_datasets, method=method)

    # Close flipped in-memory datasets
    for ds in flipped_datasets:
        if ds not in datasets:
            ds.close()
    for mf in mem_files:
        mf.close()

    # Flip result back to the original "upside down" orientation (positive e)
    if needs_flip:
        mosaic = mosaic[:, ::-1, :].copy()
        mosaic_transform = Affine(
            mosaic_transform.a, mosaic_transform.b, mosaic_transform.c,
            mosaic_transform.d, -mosaic_transform.e,
            mosaic_transform.f + mosaic_transform.e * mosaic.shape[1],
        )

    return mosaic, mosaic_transform


def single_merge(tile_paths: list[Path], output_path: Path) -> None:
    """Merge a list of tile GeoTIFFs into a single output file.

    All files in *tile_paths* are opened, merged via :func:`safe_merge`,
    written to *output_path*, then closed.
    """
    datasets_to_merge = []
    try:
        datasets_to_merge = [rio.open(str(p)) for p in tile_paths]
        mosaic, mosaic_transform = safe_merge(datasets_to_merge)

        merge_profile = datasets_to_merge[0].profile.copy()
        merge_profile.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_transform,
        })

        with rio.open(str(output_path), "w", **merge_profile) as dst:
            dst.write(mosaic)

        # Free large arrays immediately
        del mosaic

    finally:
        for ds in datasets_to_merge:
            try:
                ds.close()
            except Exception:
                pass


def chunked_merge(
        tile_paths: list[Path],
        output_path: Path,
        chunk_size: int = 100,
) -> None:
    """Merge many tiles without exceeding the OS open-file limit.

    When *tile_paths* contains more tiles than *chunk_size*, the merge is
    done in rounds: each chunk is merged into a temporary GeoTIFF, then
    the intermediate files are merged into the final output.  This avoids
    the ``Too many open files`` error that occurs when rasterio tries to
    hold hundreds of file descriptors simultaneously.

    Args:
        tile_paths: Paths to the individual prediction GeoTIFFs.
        output_path: Destination path for the merged result.
        chunk_size: Max number of files to open at once (default 100,
            conservative to account for FDs used by GDAL, Python, etc.).
    """
    if len(tile_paths) <= chunk_size:
        # Small enough → single-pass merge
        single_merge(tile_paths, output_path)
        return

    logger.info("Batched merge: %d tiles in chunks of %d", len(tile_paths), chunk_size)

    intermediate_paths: list[Path] = []
    tmp_dir = output_path.parent / "_merge_tmp"
    tmp_dir.mkdir(exist_ok=True)

    try:
        # --- Round 1: merge each chunk → intermediate file ---
        for chunk_idx in range(0, len(tile_paths), chunk_size):
            chunk = tile_paths[chunk_idx: chunk_idx + chunk_size]
            if len(chunk) == 1:
                # Single tile, no merge needed – use directly
                intermediate_paths.append(chunk[0])
                continue

            intermediate_path = tmp_dir / f"_chunk_{chunk_idx}.tif"
            single_merge(chunk, intermediate_path)
            intermediate_paths.append(intermediate_path)
            logger.debug(
                "  Chunk %d–%d merged → %s",
                chunk_idx, chunk_idx + len(chunk) - 1, intermediate_path.name,
            )
            # Force-release file descriptors held by rasterio / GDAL
            gc.collect()

        # --- Round 2: merge intermediates → final output ---
        if len(intermediate_paths) == 1:
            shutil.move(str(intermediate_paths[0]), str(output_path))
        else:
            single_merge(intermediate_paths, output_path)

    finally:
        # Clean up intermediate files
        for p in tmp_dir.glob("_chunk_*.tif"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


CLASS_NODATA = 32767
PROBABILITY_NODATA = -1.0
DEFAULT_MERGED_ALL_MIN_PAIRS = 3
DEFAULT_MERGED_ALL_PROB_THRESHOLD = 0.6


def merged_all_filename(beam: Any, sat_pass: Any, suffix: str = "") -> str:
    """Name of the temporal-confirmation mosaic for one beam / satellite pass."""
    return f"merged_all_beam-{beam}_pass-{sat_pass}{suffix}.tif"


def probability_path(class_path: Path) -> Path:
    """Sibling ``*_prob.tif`` written next to a class raster (may not exist)."""
    return class_path.with_name(f"{class_path.stem}_prob{class_path.suffix}")


def _read_class_and_probability(class_path: Path) -> tuple[np.ndarray, np.ndarray, Affine, Any]:
    """Return ``(valid, prob, transform, crs)`` for one prediction raster.

    ``prob`` is the positive-class probability read from the sibling
    ``*_prob.tif``; when that file is missing, the hard class (0/1) is used
    instead so the merge still works on class-only outputs.
    """
    with rio.open(class_path) as src:
        cls = src.read(1)
        transform, crs = src.transform, src.crs
        cls_nodata = src.nodata if src.nodata is not None else CLASS_NODATA
    valid = cls != cls_nodata

    prob_path = probability_path(class_path)
    if prob_path.exists():
        with rio.open(prob_path) as src:
            prob = src.read(1).astype(np.float32)
            prob_nodata = src.nodata if src.nodata is not None else PROBABILITY_NODATA
        valid &= prob != prob_nodata
    else:
        prob = (cls == 1).astype(np.float32)
    return valid, prob, transform, crs


def _common_grid(rasters: list[Path]) -> tuple[Affine, int, int]:
    """Union grid of north-up rasters sharing the same pixel size."""
    lefts, tops, rights, bottoms = [], [], [], []
    res = None
    for p in rasters:
        with rio.open(p) as src:
            t = src.transform
            if t.b != 0 or t.d != 0 or t.e >= 0:
                msg = f"temporal_confirmation_merge requires north-up rasters, got {t} for {p}"
                raise ValueError(msg)
            if res is None:
                res = (t.a, -t.e)
            elif not np.allclose(res, (t.a, -t.e)):
                msg = f"Inconsistent pixel size {(t.a, -t.e)} vs {res} for {p}"
                raise ValueError(msg)
            b = src.bounds
            lefts.append(b.left)
            tops.append(b.top)
            rights.append(b.right)
            bottoms.append(b.bottom)
    res_x, res_y = res
    left, top = min(lefts), max(tops)
    width = round((max(rights) - left) / res_x)
    height = round((top - min(bottoms)) / res_y)
    return Affine(res_x, 0.0, left, 0.0, -res_y, top), height, width


def temporal_confirmation_merge(
        pairs: list[list[Path]],
        output_path: Path,
        *,
        min_pairs: int = DEFAULT_MERGED_ALL_MIN_PAIRS,
        prob_threshold: float = DEFAULT_MERGED_ALL_PROB_THRESHOLD,
) -> None:
    """Mosaic several pre/post pairs with a per-pixel temporal confirmation rule.

    ``pairs`` holds, in **chronological order of the post date**, the cell
    rasters of each pre/post pair. Burned area only grows during a fire
    season, so instead of averaging pairs of different dates (where an early
    post image dilutes later detections), each pixel is classified from:

    * ``n_valid``: pairs with valid data at the pixel;
    * ``n_det``: pairs whose probability is ``>= prob_threshold``;
    * ``persist_prob``: mean probability over the valid pairs **from the first
      detection onward** (over all valid pairs if never detected).

    A pixel is burned iff ``n_det >= min(min_pairs, n_valid)`` **and**
    ``persist_prob >= prob_threshold``. Capping the required count at
    ``n_valid`` keeps pixels seen by fewer than ``min_pairs`` pairs
    classifiable (otherwise they could never be burned).

    Writes ``output_path`` (uint16, 0/1, nodata 32767) and its ``*_prob.tif``
    sibling holding ``persist_prob`` (float32, nodata -1).
    """
    rasters = [p for paths in pairs for p in paths]
    if not rasters:
        return

    grid, height, width = _common_grid(rasters)
    res_x, res_y, left, top = grid.a, -grid.e, grid.c, grid.f

    n_valid = np.zeros((height, width), dtype=np.int16)
    n_det = np.zeros((height, width), dtype=np.int16)
    seen = np.zeros((height, width), dtype=bool)
    sum_after = np.zeros((height, width), dtype=np.float32)
    cnt_after = np.zeros((height, width), dtype=np.int16)
    sum_all = np.zeros((height, width), dtype=np.float32)
    crs = None

    # Chronological order matters: "from the first detection onward".
    for paths in pairs:
        for p in paths:
            valid, prob, t, crs = _read_class_and_probability(p)
            r0 = round((top - t.f) / res_y)
            c0 = round((t.c - left) / res_x)
            h, w = prob.shape
            win = np.s_[r0:r0 + h, c0:c0 + w]

            det = valid & (prob >= prob_threshold)
            n_valid[win] += valid
            n_det[win] += det
            sum_all[win] += np.where(valid, prob, 0.0)
            seen[win] |= det
            after = valid & seen[win]
            sum_after[win] += np.where(after, prob, 0.0)
            cnt_after[win] += after

    covered = n_valid > 0
    persist_prob = np.where(
        cnt_after > 0,
        sum_after / np.maximum(cnt_after, 1),
        sum_all / np.maximum(n_valid, 1),
    ).astype(np.float32)
    required = np.minimum(n_valid, min_pairs)
    burned = covered & (n_det >= required) & (persist_prob >= prob_threshold)

    cls_out = np.where(covered, burned, CLASS_NODATA).astype(np.uint16)
    prob_out = np.where(covered, persist_prob, PROBABILITY_NODATA).astype(np.float32)

    profile = {
        "driver": "GTiff", "count": 1, "height": height, "width": width,
        "crs": crs, "transform": grid, "compress": "lzw",
    }
    with rio.open(output_path, "w", **profile, dtype="uint16", nodata=CLASS_NODATA) as dst:
        dst.write(cls_out[np.newaxis])
    with rio.open(
            probability_path(output_path), "w", **profile, dtype="float32", nodata=PROBABILITY_NODATA,
    ) as dst:
        dst.write(prob_out[np.newaxis])


def merge_predictions(  # noqa: PLR0913
        group_tile_paths: dict[tuple[str, ...], list[Path]],
        event_all_tile_paths: dict[str, list[Path]],
        *,
        write_merged_all: bool = True,
        merged_all_min_pairs: int = DEFAULT_MERGED_ALL_MIN_PAIRS,
        merged_all_prob_threshold: float = DEFAULT_MERGED_ALL_PROB_THRESHOLD,
) -> None:
    """Merge tiles in two passes.

    1. Per event/pre-post pair/beam/pass → one self-describing GeoTIFF
    2. Per event/date dir **and beam/satellite pass** →
       ``merged_all_beam-<beam>_pass-<pass>.tif`` (+ ``*_prob.tif``), built by
       :func:`temporal_confirmation_merge` over the pairs sorted by post date.
       Skipped when ``write_merged_all`` is False (e.g. benchmark layout).

    Pass 1 uses :func:`chunked_merge` to handle large tile counts without
    exceeding the OS open-file descriptor limit.
    """
    # --- Pass 1 : merge par paire pré/post et configuration SAR ---
    for (
            event_date_dir_str,
            event_id,
            event_start_date,
            event_end_date,
            group_pre,
            group_date_pre,
            group_post,
            group_date_post,
            beam,
            sat_pass,
    ), tile_paths in group_tile_paths.items():
        event_date_dir = Path(event_date_dir_str)
        if len(tile_paths) == 1:
            logger.info(
                "Writing single-cell merge for event %s, group %s/%s, beam %s, pass %s",
                event_id, group_pre, group_post, beam, sat_pass,
            )

        merged_name = merged_group_filename(
            event_id, event_start_date, event_end_date,
            group_pre, group_date_pre, group_post, group_date_post,
            beam, sat_pass,
        )
        merged_path = event_date_dir / merged_name
        logger.info("Merging %d tiles → %s/%s", len(tile_paths), event_date_dir, merged_name)

        try:
            chunked_merge(tile_paths, merged_path)
            logger.info("Saved merged group to %s", merged_path)
        except Exception:
            logger.exception(
                "Failed to merge event %s, group %s/%s, beam %s, pass %s in %s",
                event_id, group_pre, group_post, beam, sat_pass, event_date_dir,
            )

    # Force GC between passes to release all FDs from Pass 1
    gc.collect()

    if not write_merged_all:
        logger.info("Global merge (merged_all.tif) disabled — skipping pass 2.")
        return

    # --- Pass 2 : par EVENT_ID / PREDICTION_DATE / beam / pass, confirmation temporelle ---
    # key = (event_date_dir, event_id, start, end, g_pre, date_pre, g_post, date_post, beam, pass)
    by_config: dict[tuple[str, str, str], list[tuple[str, str, list[Path]]]] = defaultdict(list)
    for key, tile_paths in group_tile_paths.items():
        event_date_dir_str, group_date_pre, group_date_post, beam, sat_pass = (
            key[0], key[5], key[7], key[8], key[9]
        )
        by_config[(event_date_dir_str, beam, sat_pass)].append(
            (merge_date(group_date_post), merge_date(group_date_pre), tile_paths),
        )

    for (event_date_dir_str, beam, sat_pass), entries in by_config.items():
        event_date_dir = Path(event_date_dir_str)
        if len(event_all_tile_paths.get(event_date_dir_str, [])) < 2:  # noqa: PLR2004
            logger.info("Skipping global merge for %s (fewer than 2 tiles)", event_date_dir)
            continue

        entries.sort(key=lambda e: (e[0], e[1]))  # chronological: post date, then pre date
        merged_path = event_date_dir / merged_all_filename(beam, sat_pass)
        logger.info(
            "Temporal-confirmation merge of %d pairs (min_pairs=%d, threshold=%.2f) → %s",
            len(entries), merged_all_min_pairs, merged_all_prob_threshold, merged_path,
        )
        try:
            temporal_confirmation_merge(
                [paths for _, _, paths in entries], merged_path,
                min_pairs=merged_all_min_pairs, prob_threshold=merged_all_prob_threshold,
            )
            logger.info("Saved global merge to %s", merged_path)
        except Exception:
            logger.exception("Failed to create global merge %s", merged_path)
