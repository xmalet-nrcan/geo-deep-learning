"""Visualization tools."""
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_prediction(  # noqa: PLR0913
    image: torch.Tensor,
    mask: torch.Tensor,
    prediction: torch.Tensor,
    *,
    sample_name: str | None = None,
    num_classes: int = 1,
    class_colors: list[str] | None = None,
    save_samples: bool = False,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize the input image, ground truth mask, and prediction mask side by side.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W)
        mask (torch.Tensor): Ground truth mask tensor of shape (H, W)
        prediction (torch.Tensor): Predicted mask tensor of shape (H, W)
        sample_name (str, optional): Name of the sample
        num_classes (int): Number of classes in the segmentation
        class_colors (list, optional): List of colors for each class
        save_samples (bool, optional): Whether to save the samples
        save_path (str, optional): Path to save the visualization

    Returns:
        plt.Figure: The figure containing the visualization

    """
    num_classes = num_classes + 1 if num_classes == 1 else num_classes
    image = image.cpu().numpy()
    prediction = prediction.cpu().numpy()

    # Masque optionnel
    if mask is not None:
        mask = mask.squeeze(0).long().cpu().numpy()

    image = np.transpose(image, (1, 2, 0))
    num_channels = image.shape[-1]
    rgb_channels = 3
    if num_channels > rgb_channels:
        image = image[..., :rgb_channels]

    # Create a color map for the masks
    # On ajoute une couleur grise en position num_classes pour les pixels invalides
    if class_colors is None:
        cmap = plt.cm.get_cmap("tab20")
        vmax = num_classes - 1
    else:
        # class_colors contient les couleurs des classes [0, num_classes-1]
        # On ajoute "#808080" (gris) pour l'index num_classes = pixels invalides
        colors_with_nodata = class_colors + ["#808080"]
        cmap = ListedColormap(colors_with_nodata)
        vmax = len(colors_with_nodata) - 1  # = num_classes (inclut le gris)


    sample_name = "sample" if sample_name is None else sample_name

    if save_samples and save_path is not None:
        save_path = Path(save_path)
        plt.imsave(save_path / f"{sample_name}_image.png", image)
        if mask is not None:
            plt.imsave(
                save_path / f"{sample_name}_mask.png",
                mask,
                cmap=cmap,
                vmin=0,
                vmax=vmax,
            )
        plt.imsave(
            save_path / f"{sample_name}_prediction.png",
            prediction,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
        )

    # Nombre de colonnes : 3 si masque présent, 2 sinon
    n_cols = 3 if mask is not None else 2
    plt.close("all")
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    ax_idx = 0

    # Plot original image
    axes[ax_idx].imshow(image)
    axes[ax_idx].set_title("Input Image")
    axes[ax_idx].axis("off")
    axes[ax_idx].text(
        0.5,
        -0.1,
        f"{sample_name}",
        transform=axes[ax_idx].transAxes,
        ha="center",
        va="top",
        wrap=True,
    )
    ax_idx += 1

    # Plot ground truth mask (seulement si disponible)
    if mask is not None:
        axes[ax_idx].imshow(mask, cmap=cmap, vmin=0, vmax=vmax)
        axes[ax_idx].set_title("Ground Truth Mask")
        axes[ax_idx].axis("off")
        ax_idx += 1

    # Plot predicted mask
    axes[ax_idx].imshow(prediction, cmap=cmap, vmin=0, vmax=vmax)
    axes[ax_idx].set_title("Predicted Mask")
    axes[ax_idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)
    return fig

