"""
This class contains generalized metrics which works across different models
"""
import torch
import copy
import numpy as np
import torchlite.torch.tools.ssim as ssim
import torch.nn.functional as F


class Metric:
    def __call__(self, logits, targets):
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

        self.train_acc = {}
        self.val_acc = {}
        self.step_count = 0

    def acc_batch(self, step, logits, targets):
        """
        Called on each batch prediction.
        Will accumulate the metrics results.
        Args:
            step (str): Either "training" or "validation"
            logits (Tensor): The output logits
            targets (Tensor): The output targets
        """

        if step == "training":
            for metric in self.metrics:
                result = metric(logits, targets)
                if metric.get_name in self.train_acc.keys():
                    self.train_acc[metric.get_name] += result
                else:
                    self.train_acc[metric.get_name] = result
        elif step == "validation":
            for metric in self.metrics:
                result = metric(logits, targets)
                if metric.get_name in self.val_acc.keys():
                    self.val_acc[metric.get_name] += result
                else:
                    self.val_acc[metric.get_name] = result

        self.step_count += 1

    def avg(self, step):
        """
        Will calculate and return the metrics average results
        Args:
            step (str): Either "training" or "validation"
        Returns:
            dict: A dictionary containing the average of each metric
        """
        logs = {}
        if step == "training":
            for name, total in self.train_acc.items():
                logs[name] = total / self.step_count
        elif step == "validation":
            for name, total in self.val_acc.items():
                logs[name] = total / self.step_count
        return logs

    def reset(self):
        logs = {}
        for metric in self.metrics:
            metric.reset()
        return logs


class CategoricalAccuracy(Metric):
    def __call__(self, y_pred, y_true):
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

    def __call__(self, y_pred, y_true):
        """
        Root-mean-squared percent error
        Args:
            y_pred (Tensor): One-hot encoded tensor
            y_true (Tensor): Tensor of predictions

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

    @property
    def get_name(self):
        return "rmspe"


class SSIM(Metric):
    def __init__(self):
        """
        Calculates SSIM
        Args:
            step (str, None): Either "training", "validation" or None to run this metric on all steps
        """

    @property
    def get_name(self):
        return "ssim"

    def __call__(self, logits, targets):
        res = ssim.ssim(logits, targets)
        return res


class PSNR(Metric):
    def __init__(self):
        """
        Calculates the PSNR
        """

    @property
    def get_name(self):
        return "psnr"

    def __call__(self, logits, targets):
        logits, targets = logits.cpu().detach().numpy(), targets.cpu().detach().numpy()
        mse = ((targets - logits) ** 2).mean()
        psnr = 10 * np.log10(1 / mse)
        return psnr


class RMSE(Metric):
    def __init__(self):
        """
        Calculates the RMSE
        """

    @property
    def get_name(self):
        return "rmse"

    def __call__(self, logits, targets):
        error = torch.sqrt(F.mse_loss(logits, targets))
        return error
