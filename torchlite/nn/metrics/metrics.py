"""
This class contains generalized metrics which works across different models
"""
import gc
import torch
import copy
import numpy as np
from torch.autograd.variable import Variable
import torchlite.nn.tools.ssim as ssim


class Metric:
    def __call__(self, step, logits, target):
        raise NotImplementedError()

    @property
    def get_name(self):
        raise NotImplementedError()


class MetricsList:
    def __init__(self, metrics):
        if metrics:
            self.metrics = [copy.deepcopy(m) for m in metrics]
        else:
            self.metrics = []

        self.train_acc = None
        self.val_acc = None

    def acc_batch(self, step, logits, targets):
        """
        Called on each batch prediction.
        Will accumulate the Tensors on RAM for later computation
        Args:
            step (str): Either "training" or "validation"
            logits (Variable): The output logits
            targets (Variable): The output targets
        """
        logits = logits.cpu()
        targets = targets.cpu()

        if step == "training":
            if self.train_acc is None:
                self.train_acc = [logits, targets]
            else:
                self.train_acc[0] = torch.cat((self.train_acc[0], logits), 0)
                self.train_acc[1] = torch.cat((self.train_acc[1], targets), 0)
        elif step == "validation":
            if self.val_acc is None:
                self.val_acc = [logits, targets]
            else:
                self.val_acc[0] = torch.cat((self.val_acc[0], logits), 0)
                self.val_acc[1] = torch.cat((self.val_acc[1], targets), 0)

    def compute_flush(self, step):
        """
        Will calculate and return the metrics results and flush the accumulated Tensors.
        Args:
            step (str): Either "training" or "validation"
        Returns:

        """
        logs = {}
        if step == "training":
            for metric in self.metrics:
                result = metric(step, self.train_acc[0], self.train_acc[1])
                if result:
                    logs[metric.get_name] = result
            self.train_acc = None
        elif step == "validation":
            for metric in self.metrics:
                result = metric(step, self.val_acc[0], self.val_acc[1])
                if result:
                    logs[metric.get_name] = result
            self.val_acc = None

        gc.collect()
        return logs

    def reset(self):
        logs = {}
        for metric in self.metrics:
            metric.reset()
        return logs


class CategoricalAccuracy(Metric):
    def __call__(self, step, y_pred, y_true):
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
        if isinstance(y_true, Variable):
            y_true = y_true.data
        if isinstance(y_pred_dense, Variable):
            y_pred_dense = y_pred_dense.data
        sm = torch.sum(y_true == y_pred_dense)
        return 100. * sm / y_true.size()[0]

    @property
    def get_name(self):
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

    def __call__(self, step, y_pred, y_true):
        """
        Root-mean-squared percent error
        Args:
            y_pred (Tensor): One-hot encoded tensor
            y_true (Tensor): Tensor of predictions

        Returns:
            The Root-mean-squared percent error
        """
        if self.to_exp:
            targ = torch.exp(y_true)
        else:
            targ = y_true
        pct_var = (targ - y_pred) / targ
        res = np.sqrt((pct_var ** 2).mean())
        return res

    @property
    def get_name(self):
        return "rmspe"


class SSIM(Metric):
    def __init__(self, step=None):
        """
        Calculates SSIM
        Args:
            step (str, None): Either "training", "validation" or None to run this metric on all steps
        """
        self.step = step

    @property
    def get_name(self):
        return "ssim"

    def __call__(self, step, logits, target):
        if not self.step or self.step == step:
            res = ssim.ssim(logits, target).data[0]
            return res


class PSNR(Metric):
    def __init__(self, step=None):
        """
        Calculates PSNR
        Args:
            step (str, None): Either "training", "validation" or None to run this metric on all steps
        """
        self.step = step

    @property
    def get_name(self):
        return "psnr"

    def __call__(self, step, logits, target):
        if not self.step or self.step == step:
            mse = ((target - logits) ** 2).data.mean()
            psnr = 10 * np.log10((255 ** 2) / mse)
            return psnr
