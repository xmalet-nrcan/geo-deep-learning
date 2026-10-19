"""Tile geometry helpers for tiled / buffered change-detection batches.

Pure functions (no ``LightningModule`` dependency) operating on the batch
dictionaries produced by ``TiledChangeDetectionDataset`` /
``RCMChangeDetectionDataset``:

* **Tile ownership** — which pixels of an overlapping tile a tile *exclusively*
  owns, so tiled metrics count every source pixel exactly once.
* **Neighbour-context buffer** — which pixels of a tile fall inside the true
  (un-buffered) cell, so the loss and the saved predictions ignore the buffer
  ring added only to give the model spatial context.
"""

from typing import Any

import torch
from torch import Tensor


def scalar_at(field: Any, index: int) -> int:
    """Return ``int(field[index])``, handling both Tensor and list/tuple batches.

    Batched per-sample scalar fields (e.g. ``tile_row_start``, ``buffer_size``,
    ``original_height``) may arrive as a ``Tensor`` or as a plain Python list
    depending on the collate function.
    """
    value = field[index]
    return int(value.item()) if isinstance(field, torch.Tensor) else int(value)


# ----------------------------------------------------------------------
# Tile ownership (makes tiled metrics comparable to non-tiled)
# ----------------------------------------------------------------------


def axis_tile_starts(source: int, tile: int, stride: int) -> list[int]:
    """Reproduce the tile origins used at tiling time for one axis.

    Mirrors ``TiledChangeDetectionDataset._expand_files_with_tiles`` so the
    model can reconstruct, per sample, the full set of tile positions from
    ``source_height`` / ``source_width`` alone.
    """
    if source <= tile:
        return [0]
    starts = list(range(0, (source - tile) + 1, stride))
    last = source - tile
    if last not in starts:
        starts.append(last)
    return sorted(set(starts))


def axis_ownership_end(starts: list[int], start: int, tile: int) -> int:
    """Local exclusive end index of the region a tile *exclusively* owns.

    A source pixel is owned by the tile with the **largest** start that
    still covers it.  In tile-local coordinates a tile therefore owns
    ``[0, next_start - start)`` (or the full tile for the last one),
    guaranteeing a partition of the source with no overlap and no gap.
    """
    nexts = [s for s in starts if s > start]
    return tile if not nexts else (min(nexts) - start)


def tile_ownership_mask(
        batch: dict[str, Any],
        ref_shape: torch.Size,
        tile_size: tuple[int, int] | None,
        tile_stride: tuple[int, int] | None,
) -> Tensor | None:
    """Return a ``[B, 1, H, W]`` mask keeping only pixels each tile owns.

    Eliminates double-counting of overlapping tiles in the metrics so that
    tiled evaluation counts every source pixel exactly once — matching the
    non-tiled setup.  Returns ``None`` when tiling is inactive (no-op).

    Args:
        batch: Batch dict (needs ``tile_row_start``, ``tile_col_start``,
            ``source_height``, ``source_width``).
        ref_shape: Shape ``[B, C, H, W]`` of the tensor the mask is applied to.
        tile_size: ``(h, w)`` of a tile, from the DataModule.
        tile_stride: ``(h, w)`` stride between tiles, from the DataModule.
    """
    if "tile_row_start" not in batch or "source_height" not in batch:
        return None
    if not tile_size or not tile_stride:
        return None

    tile_h, tile_w = tile_size
    stride_h, stride_w = tile_stride
    b, _, h, w = ref_shape
    mask = torch.zeros((b, 1, h, w), dtype=torch.float32)

    for i in range(b):
        r = scalar_at(batch["tile_row_start"], i)
        c = scalar_at(batch["tile_col_start"], i)
        sh = scalar_at(batch["source_height"], i)
        sw = scalar_at(batch["source_width"], i)

        row_starts = axis_tile_starts(sh, tile_h, stride_h)
        col_starts = axis_tile_starts(sw, tile_w, stride_w)
        row_end = min(axis_ownership_end(row_starts, r, tile_h), h)
        col_end = min(axis_ownership_end(col_starts, c, tile_w), w)

        if row_end > 0 and col_end > 0:
            mask[i, 0, :row_end, :col_end] = 1.0
    return mask


# ----------------------------------------------------------------------
# Neighbour-context buffer (``train_overlap_buffer`` / ``predict_overlap_buffer``)
# ----------------------------------------------------------------------


def buffer_valid_region(
        buf: int, tr: int, tc: int, tile_h: int, tile_w: int, oh: int, ow: int,
) -> tuple[int, int, int, int]:
    """Valid (non-buffer) region of one tile, in tile-local pixel coordinates.

    Returns ``(row_start, row_end, col_start, col_end)`` bounding the
    pixels of a ``tile_h x tile_w`` tile — cropped from an expanded
    ``(cell + 2*buf)`` image at offset ``(tr, tc)`` — that fall inside
    the true, un-buffered ``oh x ow`` cell. Either span collapses
    (``*_end <= *_start``) when the tile lies entirely within the
    neighbour-context buffer ring, i.e. it owns zero true-cell pixels.

    The non-tiled case is the special case ``tr = tc = 0`` and
    ``tile_h, tile_w = oh + 2*buf, ow + 2*buf`` (the whole buffered
    image treated as a single "tile").
    """
    row_start = max(0, buf - tr)
    row_end = min(tile_h, buf + oh - tr)
    col_start = max(0, buf - tc)
    col_end = min(tile_w, buf + ow - tc)
    return row_start, row_end, col_start, col_end


def mask_buffer_zone(
        batch: dict[str, Any],
        common_data_mask: Tensor,
        batch_size: int,
) -> Tensor:
    """Zero out the neighbour-context buffer zone (``train_overlap_buffer``).

    When spatial context is loaded from adjacent cells the image is
    ``(h + 2b) × (w + 2b)`` but the ground-truth label mask only covers
    the central ``(h × w)`` region.  The buffer zone is zeroed in the
    returned mask so the loss is never computed there.  The dataset
    stores ``buffer_size``, ``cell_orig_height``, ``cell_orig_width``
    whenever a non-zero buffer was applied (train or predict).
    """
    if "buffer_size" not in batch:
        return common_data_mask

    buf = scalar_at(batch["buffer_size"], 0)
    if buf <= 0:
        return common_data_mask

    common_data_mask = common_data_mask.clone()

    if "tile_row_start" in batch:
        # --- Tiled + buffered ---
        # Each tile is a crop of the expanded (cell + 2*buf) image.  The
        # valid zone (central cell, excluding neighbour context) spans
        # rows [buf, buf + orig_h) × cols [buf, buf + orig_w) in the
        # expanded image.  We compute the intersection of this valid
        # zone with each tile's coverage area and mask everything
        # outside.  Without this per-tile logic, masking buf pixels
        # from ALL 4 edges of EVERY tile wrongly discarded up to 63 % of
        # valid pixels on interior tiles that don't touch the buffer
        # boundary at all.
        tile_h, tile_w = common_data_mask.shape[2], common_data_mask.shape[3]
        orig_h_batch = batch["cell_orig_height"]
        orig_w_batch = batch["cell_orig_width"]
        tile_row_start = batch["tile_row_start"]
        tile_col_start = batch["tile_col_start"]

        for i in range(batch_size):
            tr = scalar_at(tile_row_start, i)
            tc = scalar_at(tile_col_start, i)
            oh = scalar_at(orig_h_batch, i)
            ow = scalar_at(orig_w_batch, i)

            vr_start, vr_end, vc_start, vc_end = buffer_valid_region(
                buf, tr, tc, tile_h, tile_w, oh, ow,
            )

            if vr_start > 0:
                common_data_mask[i, :, :vr_start, :] = 0.0
            if vr_end < tile_h:
                common_data_mask[i, :, vr_end:, :] = 0.0
            if vc_start > 0:
                common_data_mask[i, :, :, :vc_start] = 0.0
            if vc_end < tile_w:
                common_data_mask[i, :, :, vc_end:] = 0.0
    else:
        # --- Non-tiled + buffered ---
        # The full expanded image: mask buf pixels from all edges.
        # BUGFIX: ``row_start``/``row_end`` (and column counterparts) are
        # clamped so a buffer >= half the image size correctly zeroes
        # the *entire* mask.  The previous code only masked when
        # ``buf < img_h`` and otherwise did nothing — but with
        # ``buf >= img_h`` the unconditional slice
        # ``common_data_mask[:, :, img_h - buf:, :]`` would have used a
        # *negative* start index (Python counts from the end), silently
        # zeroing the wrong rows instead of the whole mask.
        img_h, img_w = common_data_mask.shape[2], common_data_mask.shape[3]
        row_start = min(buf, img_h)
        row_end = max(img_h - buf, row_start)
        col_start = min(buf, img_w)
        col_end = max(img_w - buf, col_start)
        common_data_mask[:, :, :row_start, :] = 0.0
        common_data_mask[:, :, row_end:, :] = 0.0
        common_data_mask[:, :, :, :col_start] = 0.0
        common_data_mask[:, :, :, col_end:] = 0.0

    return common_data_mask
