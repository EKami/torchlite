import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable


class Metric:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError()

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
            d = 0
        else:
            self.metrics = []

    def __call__(self, y_true, y_pred):
        logs = {}
        for metric in self.metrics:
            logs[metric.get_name()] = metric(y_true, y_pred)
        return logs

    def avg(self):
        logs = {}
        for metric in self.metrics:
            logs[metric.get_name()] = metric.avg()
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

    def __call__(self, y_true, y_pred):
        """
        Return the accuracy of the predictions across the whole dataset
         Args:
            y_true (Tensor): Tensor of shape (batch_size, 1)
            y_pred (Tensor): One-hot encoded tensor of shape (batch_size, preds)

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

    def get_name(self):
        return "accuracy"

    def reset(self):
        self.correct_count = 0


class RMSPE(Metric):
    def __init__(self, to_exp=False):
        """

        Args:
            to_exp (bool): Set to True if the targets need to be turned
            to exponential before the metric is processed
        """
        super().__init__()
        self.to_exp = to_exp
        self.count = 0
        self.sum = 0

    def __call__(self, y_true, y_pred):
        """
        Root-mean-squared percent error
        Args:
            y_true (Tensor): Tensor of shape (batch_size, 1)
            y_pred (Tensor): One-hot encoded tensor of shape (batch_size, preds)

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

    def get_name(self):
        return "rmspe"

    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.count = 0
        self.sum = 0

