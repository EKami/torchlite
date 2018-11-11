from torchlite.common.learner.cores import BaseCore
from torchlite.common.tools import tensor_tools


class ClassifierCore(BaseCore):
    def __init__(self, model, optimizer, criterion, input_index=0):
        """
        The learner core for classification models
        Args:
            model (nn.Module): The pytorch model
            optimizer (Optimizer): The optimizer function
            criterion (callable): The objective criterion.
            input_index (int): An index pointing to the location of the input data in the batch.
                N.B: The target is always the last item in the batch
        """
        self.input_index = input_index
        self.crit = criterion
        self.optim = optimizer
        self.model = model
        self.logs = {}
        self.avg_meter = tensor_tools.AverageMeter()

    @property
    def get_models(self):
        return {self.model.__class__.__name__: self.model}

    @property
    def get_logs(self):
        return self.logs

    def on_new_epoch(self):
        self.logs = {}
        self.avg_meter = tensor_tools.AverageMeter()

    def on_train_mode(self):
        self.model.train()

    def on_eval_mode(self):
        self.model.eval()

    def to_device(self, device):
        self.model.to(device)

    def on_forward_batch(self, step, inputs, targets=None):
        # forward
        logits = self.model.forward(inputs[self.input_index])

        if step != "prediction":
            loss = self.crit(logits, targets)

            # Update logs
            self.avg_meter.update(loss.item())
            self.logs.update({"batch_logs": {"loss": loss.item()}})

            # backward + optimize
            if step == "training":
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.logs.update({"epoch_logs": {"train loss": self.avg_meter.avg}})
            else:
                self.logs.update({"epoch_logs": {"valid loss": self.avg_meter.avg}})
        return logits