"""
This class contains generalized losses which works across different models
"""
import torch
import torch.nn as nn
import numpy as np


def nwrmsle(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray):
    """
    Calculates and returns the Normalized Weighted Root Mean Squared Logarithmic Error
    Args:
        y_true (np.ndarray): The true labels
        y_pred (np.ndarray): The predicted labels
        weights (np.ndarray): The weights for each predictions

    Returns:
        int: The NWRMSLE
    """
    assert y_true.shape == y_pred.shape == weights.shape, "Arguments are not of same shape"
    y_true = y_true.clip(min=0)
    y_pred = y_pred.clip(min=0)
    return np.sqrt(np.sum(weights * np.square(np.log1p(y_pred) - np.log1p(y_true))) / np.sum(weights))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class CharbonnierLoss(nn.Module):
    def __init__(self):
        """
        L1 Charbonnierloss.
        """
        super(CharbonnierLoss, self).__init__()

    def forward(self, x, y, eps=1e-6):
        diff = y - x
        error = torch.sqrt(diff * diff + eps)
        loss = torch.mean(error)
        return loss

