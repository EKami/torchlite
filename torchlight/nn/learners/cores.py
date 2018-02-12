import torch.nn as nn
from torchlight.nn import tools


class BaseCore:
    def on_train_mode(self):
        raise NotImplementedError()

    def on_eval_mode(self):
        raise NotImplementedError()

    def to_gpu(self):
        """
        Move the model onto the GPU
        """
        raise NotImplementedError()

    @property
    def get_logs(self):
        """
        Returns the logs for display

        Returns:
            dict: The logs from the forward batch
        """
        raise NotImplementedError()

    def on_forward_batch(self, step, inputs, targets=None):
        """
        Callback called during training, validation and prediction batch processing steps
        Args:
            step (str): Either:
                - training
                - validation
                - prediction
            inputs (Tensor): The batch inputs to feed to the model
            targets (Tensor): The expected outputs

        Returns:
            Tensor: The logits
        """
        raise NotImplementedError()


class ClassifierCore(BaseCore):
    def __init__(self, model, optimizer, criterion):
        """
        The learner core for classification models
        Args:
            model (nn.Module): The pytorch model
            optimizer (Optimizer): The optimizer function
            criterion (callable): The objective criterion.
        """
        self.crit = criterion
        self.optim = optimizer
        self.model = model
        self.logs = {}
        self.avg_meter = tools.AverageMeter()

    @property
    def get_logs(self):
        return self.logs

    def on_train_mode(self):
        self.model.train()

    def on_eval_mode(self):
        self.model.eval()

    def to_gpu(self):
        tools.to_gpu(self.model)

    def on_forward_batch(self, step, inputs, targets=None):
        # forward
        logits = self.model.forward(*inputs)

        if step != "prediction":
            loss = self.crit(logits, targets)

            # Update logs
            self.avg_meter.update(loss.data[0])
            self.logs.update({"batch_logs": {"loss": loss.data[0]}})
            self.logs.update({"total_loss": self.avg_meter.debias_loss})

            # backward + optimize
            if step == "training":
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
        return logits
