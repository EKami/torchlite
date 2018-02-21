"""
This class contains generalized metrics which works across different models
"""
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd.variable import Variable
import torchlite.nn.tools.ssim as ssim


class Metric:
    def __call__(self, step, logits, target):
        raise NotImplementedError()

    @property
    def get_name(self):
        raise NotImplementedError()

    def avg(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class MetricsList:
    def __init__(self, metrics):
        if metrics:
            self.metrics = [copy.deepcopy(m) for m in metrics]
        else:
            self.metrics = []

    def __call__(self, step, logits, target):
        logs = {}
        for metric in self.metrics:
            result = metric(step, logits, target)
            if result:
                logs[metric.get_name] = result
        return logs

    def avg(self):
        logs = {}
        for metric in self.metrics:
            logs[metric.get_name] = metric.avg()
        return logs

    def reset(self):
        logs = {}
        for metric in self.metrics:
            metric.reset()
        return logs


class CategoricalAccuracy(Metric):
    def __init__(self):
        self.correct_count = 0
        self.count = 0

    def __call__(self, step, y_pred, y_true):
        """
        Return the accuracy of the predictions across the whole dataset
         Args:
            y_pred (Tensor): One-hot encoded tensor of shape (batch_size, preds)
            y_true (Tensor): Tensor of shape (batch_size, 1)

        Returns:
            float: Average accuracy
        """
        _, y_pred_dense = y_pred.max(1)
        assert y_true.size() == y_pred_dense.size(), "y_true and y_pred shapes differ"
        if isinstance(y_true, Variable):
            y_true = y_true.data
        if isinstance(y_pred_dense, Variable):
            y_pred_dense = y_pred_dense.data
        sm = torch.sum(y_true == y_pred_dense)
        self.correct_count += sm
        self.count += y_true.size()[0]
        return self.avg()

    def avg(self):
        return 100. * self.correct_count / self.count

    @property
    def get_name(self):
        return "accuracy"

    def reset(self):
        self.correct_count = 0


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
        self.count = 0
        self.sum = 0

    def __call__(self, step, y_pred, y_true):
        """
        Root-mean-squared percent error
        Args:
            y_pred (Tensor): One-hot encoded tensor of shape (batch_size, preds)
            y_true (Tensor): Tensor of shape (batch_size, 1)

        Returns:
            The Root-mean-squared percent error
        """
        if self.to_exp:
            targ = torch.exp(y_true)
        else:
            targ = y_true
        pct_var = (targ - torch.exp(y_pred)) / targ
        res = torch.sqrt((pct_var ** 2).mean()).data[0]
        self.count += 1
        self.sum += res
        return res

    @property
    def get_name(self):
        return "rmspe"

    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.count = 0
        self.sum = 0


class SSIM(Metric):
    def __init__(self, step=None):
        """
        Calculates SSIM
        Args:
            step (str, None): Either "training", "validation" or None to run this metric on all steps
        """
        self.step = step
        self.count = 0
        self.sum = 0

    @property
    def get_name(self):
        return "ssim"

    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.count = 0
        self.sum = 0

    def __call__(self, step, logits, target):
        if not self.step or self.step == step:
            res = ssim.ssim(logits, target).data[0]
            self.count += logits.size()[0]  # Batch size
            self.sum += res
            return res


class PSNR(Metric):
    def __init__(self, step=None):
        """
        Calculates PSNR
        Args:
            step (str, None): Either "training", "validation" or None to run this metric on all steps
        """
        self.step = step
        self.count = 0
        self.sum = 0

    @property
    def get_name(self):
        return "psnr"

    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.count = 0
        self.sum = 0

    def __call__(self, step, logits, target):
        if not self.step or self.step == step:
            batch_mse = ((target - logits) ** 2).data.mean()
            psnr = 10 * np.log10((255 ** 2) / batch_mse)

            self.count += logits.size(0)
            self.sum += psnr
            return psnr