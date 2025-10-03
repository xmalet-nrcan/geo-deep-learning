"""
Target format converters for different segmentation tasks.

This module provides utilities to convert between different target formats,
particularly for converting semantic segmentation masks to instance-based
formats required by query-based models like Mask2Former.
"""

import torch
from torch import Tensor


def semantic_to_instance_masks(
    semantic_masks: Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> list[dict[str, Tensor]]:
    """
    Convert semantic segmentation masks to instance format for Mask2Former.

    Converts dense semantic masks where each pixel has a class label into
    a list of instance masks where each class gets its own binary mask.

    Args:
        semantic_masks: Tensor of shape [B, H, W] with class indices
        num_classes: Number of classes (excluding background/ignore)
        ignore_index: Index to ignore (typically 255 for unlabeled pixels)

    Returns:
        List of dicts (one per batch element) with keys:
            - "labels": Tensor of shape [N] with class indices for each instance
            - "masks": Tensor of shape [N, H, W] with binary masks for each instance

    Example:
        >>> # Batch of 2 images with 4 classes
        >>> semantic_masks = torch.randint(0, 4, (2, 128, 128))
        >>> targets = semantic_to_instance_masks(semantic_masks, num_classes=4)
        >>> print(len(targets))  # 2 (batch size)
        >>> print(targets[0]["labels"].shape)  # [N] where N = num classes present
        >>> print(targets[0]["masks"].shape)   # [N, 128, 128]

    """
    batch_size, height, width = semantic_masks.shape
    device = semantic_masks.device

    targets = []

    for b in range(batch_size):
        mask = semantic_masks[b]  # [H, W]

        labels = []
        masks = []

        # For each class, create a binary mask if that class exists
        for class_id in range(num_classes):
            # Create binary mask for this class
            class_mask = mask == class_id

            # Skip if this class doesn't exist in the image
            if not class_mask.any():
                continue

            # Skip ignore index
            if class_id == ignore_index:
                continue

            labels.append(class_id)
            masks.append(class_mask)

        # Handle case where no valid classes exist
        if len(labels) == 0:
            # Create dummy instance (background class = num_classes)
            labels.append(num_classes)
            masks.append(torch.zeros(height, width, dtype=torch.bool, device=device))

        # Stack into tensors
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        masks_tensor = torch.stack(masks).float()  # [N, H, W]

        targets.append(
            {
                "labels": labels_tensor,
                "masks": masks_tensor,
            },
        )

    return targets


def semantic_to_instance_masks_with_metadata(
    semantic_masks: Tensor,
    num_classes: int,
    ignore_index: int = 255,
    metadata: dict | None = None,
) -> list[dict[str, Tensor]]:
    """
    Convert semantic masks to instance format with optional metadata.

    Extended version that can include additional metadata like image names,
    original masks, etc.

    Args:
        semantic_masks: Tensor of shape [B, H, W] with class indices
        num_classes: Number of classes (excluding background/ignore)
        ignore_index: Index to ignore (typically 255)
        metadata: Optional dict with additional info to include in targets

    Returns:
        List of dicts with "labels", "masks", and optional metadata fields

    """
    targets = semantic_to_instance_masks(semantic_masks, num_classes, ignore_index)

    # Add metadata if provided
    if metadata is not None:
        for i, target in enumerate(targets):
            for key, value in metadata.items():
                # Handle both batched and non-batched metadata
                if isinstance(value, (list, tuple)) or (
                    isinstance(value, Tensor) and value.dim() > 0
                ):
                    target[key] = value[i]
                else:
                    target[key] = value

    return targets


def validate_targets(
    targets: list[dict[str, Tensor]],
    num_classes: int,
) -> bool:
    """
    Validate target format for Mask2Former.

    Args:
        targets: List of target dicts
        num_classes: Expected number of classes

    Returns:
        True if valid, raises ValueError otherwise

    """
    for i, target in enumerate(targets):
        # Check required keys
        if "labels" not in target:
            msg = f"Target {i} missing 'labels' key"
            raise ValueError(msg)
        if "masks" not in target:
            msg = f"Target {i} missing 'masks' key"
            raise ValueError(msg)

        labels = target["labels"]
        masks = target["masks"]

        # Check shapes
        if labels.dim() != 1:
            msg = f"Target {i}: labels must be 1D, got shape {labels.shape}"
            raise ValueError(msg)

        if masks.dim() != 3:  # noqa: PLR2004
            msg = f"Target {i}: masks must be 3D [N, H, W], got shape {masks.shape}"
            raise ValueError(msg)

        if labels.shape[0] != masks.shape[0]:
            msg = (
                f"Target {i}: labels/masks N mismatch: {labels.shape[0]} vs "
                f"{masks.shape[0]}"
            )
            raise ValueError(msg)

        # Check label values
        if labels.max() > num_classes:
            msg = f"Target {i}: label {labels.max().item()} > num_classes {num_classes}"
            raise ValueError(msg)

        if labels.min() < 0:
            msg = f"Target {i}: negative label {labels.min().item()} found"
            raise ValueError(msg)

    return True


# if __name__ == "__main__":
#     print("Testing target converters...")

#     # Test basic conversion
#     print("\n1. Testing semantic_to_instance_masks...")
#     batch_size, h, w = 2, 64, 64
#     num_classes = 4

#     # Create semantic masks with classes 0-3
#     semantic_masks = torch.randint(0, num_classes, (batch_size, h, w))

#     # Add some variety - make class 2 not present in first image
#     semantic_masks[0][semantic_masks[0] == 2] = 0

#     targets = semantic_to_instance_masks(semantic_masks, num_classes)

#     print(f"   Batch size: {batch_size}")
#     print(f"   Num classes: {num_classes}")
#     print(f"   Targets length: {len(targets)}")

#     for i, target in enumerate(targets):
#         print(f"\n   Batch {i}:")
#         print(f"      Labels: {target['labels'].tolist()}")
#         print(f"      Masks shape: {target['masks'].shape}")
#         print(f"      Num instances: {len(target['labels'])}")

#     # Test with metadata
#     print("\n2. Testing semantic_to_instance_masks_with_metadata...")
#     metadata = {
#         "image_names": ["img1.tif", "img2.tif"],
#         "original_shape": torch.tensor([[512, 512], [512, 512]]),
#     }

#     targets_with_meta = semantic_to_instance_masks_with_metadata(
#         semantic_masks,
#         num_classes,
#         metadata=metadata,
#     )

#     print("   Batch 0 metadata:")
#     print(f"      image_name: {targets_with_meta[0]['image_names']}")
#     print(f"      original_shape: {targets_with_meta[0]['original_shape'].tolist()}")

#     # Test validation
#     print("\n3. Testing validate_targets...")
#     try:
#         validate_targets(targets, num_classes)
#         print("   ✅ Validation passed")
#     except ValueError as e:
#         print(f"   ❌ Validation failed: {e}")

#     # Test with all classes present
#     print("\n4. Testing with all classes present...")
#     semantic_masks_full = torch.zeros(1, 64, 64, dtype=torch.long)
#     for c in range(num_classes):
#         semantic_masks_full[0, c * 16 : (c + 1) * 16, :] = c

#     targets_full = semantic_to_instance_masks(semantic_masks_full, num_classes)
#     print(f"   Classes present: {targets_full[0]['labels'].tolist()}")
#     print(f"   Expected: {list(range(num_classes))}")

#     # Test with no classes (edge case)
#     print("\n5. Testing with ignore_index...")
#     semantic_masks_ignore = torch.full((1, 64, 64), 255, dtype=torch.long)
#     targets_ignore = semantic_to_instance_masks(
#         semantic_masks_ignore,
#         num_classes,
#         ignore_index=255,
#     )
#     print(f"Num instances (should be 1 dummy): {len(targets_ignore[0]['labels'])}")
#     print(f"Label (should be {num_classes}): {targets_ignore[0]['labels'].tolist()}")

#     print("\n✅ All tests passed!")
