"""
This class contains generalized metrics which works across different models
"""
import torch
import numpy as np
import torchlite.torch.tools.ssim as ssim
import torch.nn.functional as F

from torchlite.common.metrics import Metric


class CategoricalAccuracy(Metric):
    def __call__(self, logger, y_true, y_pred):
        """
        Return the accuracy of the predictions across the whole batch
         Args:
            y_pred (Tensor): One-hot encoded tensor of shape (batch_size, preds)
            y_true (Tensor): Tensor of shape (batch_size, 1)

        Returns:
            float: Average accuracy
        """
        _, y_pred_dense = y_pred.max(1)
        assert y_true.size() == y_pred_dense.size(), "y_true and y_pred shapes differ"
        sm = torch.sum(y_true == y_pred_dense).float()
        return 100. * sm / y_true.size()[0]

    def __repr__(self):
        return "accuracy"


class RMSPE(Metric):
    def __init__(self, to_exp=False):
        """
        Root-mean-squared percent error
        Args:
            to_exp (bool): Set to True if the targets need to be turned
            to exponential before the metric is processed
        """
        super().__init__()
        self.to_exp = to_exp

    def __call__(self, logger, y_true, y_pred):
        """
        Root-mean-squared percent error
        Args:
            y_true (Tensor): Tensor of predictions
            y_pred (Tensor): One-hot encoded tensor

        Returns:
            The Root-mean-squared percent error
        """
        if self.to_exp:
            logits = torch.exp(y_pred)
            targ = torch.exp(y_true)
        else:
            logits = y_pred
            targ = y_true
        pct_var = (targ - logits) / targ
        res = torch.sqrt((pct_var ** 2).mean())
        return res

    def __repr__(self):
        return "rmspe"


class SSIM(Metric):
    def __init__(self):
        """
        Calculates SSIM
        Args:
            step (str, None): Either "training", "validation" or None to run this metric on all steps
        """

    def __call__(self, logger, y_true, y_pred):
        res = ssim.ssim(y_pred, y_true)
        return res

    def __repr__(self):
        return "ssim"


class PSNR(Metric):
    def __init__(self):
        """
        Calculates the PSNR
        """

    def __call__(self, logger, y_true, y_pred):
        logits, targets = y_pred.cpu().detach().numpy(), y_true.cpu().detach().numpy()
        mse = ((targets - logits) ** 2).mean()
        psnr = 10 * np.log10(1 / mse)
        return psnr

    def __repr__(self):
        return "psnr"


class RMSE(Metric):
    def __init__(self):
        """
        Calculates the RMSE
        """

    def __call__(self, logger, y_true, y_pred):
        error = torch.sqrt(F.mse_loss(y_pred, y_true))
        return error

    def __repr__(self):
        return "rmse"
