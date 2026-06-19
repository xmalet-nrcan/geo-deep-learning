"""Utility functions related to models."""

import logging

import torch

logger = logging.getLogger(__name__)


def load_weights_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    load_parts: str | list[str] | None = None,
    map_location: torch.device | None = None,
    strict: bool = False,
) -> tuple[list[str], list[str]] | None:
    """
    Load weights from a checkpoint into a model.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        load_parts: List of model parts to load (e.g., ["encoder", "neck"])
        map_location: Optional device mapping for loading the checkpoint
        strict: If True, raises on mismatched keys. If False (default),
            loads compatible weights and logs warnings for the rest.
            Use strict=False when transferring between model variants
            (e.g. ChangeFormerV6 → V7).

    Returns:
        Tuple of (missing_keys, unexpected_keys) if selective loading,
        or None if full loading

    """
    logger.info("Loading weights from checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Lightning checkpoints store the full LightningModule state: model weights,
    # augmentation modules (_geo_aug, _pad_aug), metrics, etc.
    # Keep only keys belonging to the inner model (prefix "model.") and strip it.
    state_dict = {
        k.removeprefix("model."): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    if load_parts is not None:
        if isinstance(load_parts, str):
            load_parts = [load_parts]

        filtered_state_dict = {}
        loaded_parts_keys = {part: [] for part in load_parts}
        for k, v in state_dict.items():
            for part in load_parts:
                if k.startswith(f"{part}."):
                    filtered_state_dict[k] = v
                    loaded_parts_keys[part].append(k)

        result = model.load_state_dict(filtered_state_dict, strict=False)
        logger.info("Loaded weights for parts: %s", load_parts)
        for part in load_parts:
            num_keys = len(loaded_parts_keys[part])
            if num_keys > 0:
                logger.info("  - %s: %s parameters loaded", part, num_keys)
                examples = loaded_parts_keys[part][:3]
                if examples:
                    logger.info("    Examples: %s", ", ".join(examples))
            else:
                logger.info(
                    "  - %s: NO PARAMETERS FOUND - check if this part exists",
                    part,
                )

        logger.info("Missing keys: %s", len(result.missing_keys))
        logger.info("Unexpected keys: %s", len(result.unexpected_keys))
        return result

    result = model.load_state_dict(state_dict, strict=strict)
    n_loaded = len(state_dict) - len(result.unexpected_keys)
    n_total = len(dict(model.named_parameters())) + len(dict(model.named_buffers()))
    logger.info(
        "Loaded %d/%d parameters from checkpoint.", n_loaded, n_total,
    )
    if result.missing_keys:
        logger.warning(
            "Missing keys (%d) — these layers will be randomly initialized: %s",
            len(result.missing_keys),
            result.missing_keys[:10],
        )
    if result.unexpected_keys:
        logger.warning(
            "Unexpected keys (%d) — these checkpoint weights were ignored: %s",
            len(result.unexpected_keys),
            result.unexpected_keys[:10],
        )
    return result
