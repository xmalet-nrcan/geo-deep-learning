import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss, DiceLoss


class DiceBCELoss(nn.Module):
    def __init__(self,
                 smooth=1e-6,
                 alpha=0.5,
                 eps=1e-7,
                 ignore_index=None,
                 mode='binary'):
        """
        Combinaison Dice + BCE Loss pour segmentation binaire
        :param smooth: terme de lissage pour éviter les divisions par zéro
        :param alpha: poids de BCE vs Dice (0.5 = équilibré)
        """
        super(DiceBCELoss, self).__init__()
        self.alpha = alpha
        self.bce = SoftBCEWithLogitsLoss(ignore_index=ignore_index, smooth_factor=smooth)
        self.dice = DiceLoss(smooth=smooth, eps=eps,  mode=mode, ignore_index=ignore_index)

    def forward(self, y_pred, y_true):
        # BCE Loss
        bce_loss = self.bce(y_pred, y_true)

        # Dice Loss
        dice_loss = self.dice(y_pred, y_true)

        # Combinaison
        loss = self.alpha * bce_loss + (1 - self.alpha) * dice_loss
        return loss
