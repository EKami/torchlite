from datetime import datetime
import torch
import torch.nn as nn
import torchlight.nn.train_callbacks as train_callbacks
import torchlight.nn.test_callbacks as test_callbacks
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchlight.nn import tools
from torchlight.nn.metrics import MetricsList


class BaseCore:
    pass


class ClassifierCore(BaseCore):
    def __init__(self, model, optim, crit):
        """

        Args:
            model (nn.Module): The pytorch model
            optim (Optimizer): The optimizer function
            crit (callable): The objective criterion.
        """
        self.crit = crit
        self.optim = optim
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

    def on_forward_batch(self, step, inputs, targets):
        # forward
        logits = self.model.forward(*inputs)
        loss = self.crit(logits, targets)

        self.avg_meter.update(loss.data[0])
        self.logs.update({"loss": loss.data[0]})
        self.logs.update({"total_loss": self.avg_meter.debias_loss})

        # backward + optimize
        if step == "training":
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        elif step == "predict":
            # forward
            logits = logits.data
            if ret_logits is None:
                ret_logits = torch.zeros(len(test_loader.dataset), logits.shape[1])
            ret_logits[batch_size * ind:batch_size * ind + batch_size] = logits
            callback_list.on_batch_end(ind, logs={"batch_size": batch_size})

        # TODO return the losses as logs


class Learner:
    def __init__(self, learner_core, use_cuda=True):
        """
        The learner class used to train deep neural network
        Args:
            use_cuda (bool): If True moves the model onto the GPU
        """
        # TODO contains the model/models, the optimizer and the losses
        self.learner_core = learner_core
        self.epoch_counter = 0
        self.use_cuda = False
        if use_cuda:
            if torch.cuda.is_available():
                self.use_cuda = True
            else:
                print("/!\ Warning: Cuda set but not available, using CPU...")

    def _train_epoch(self, step, loader, metrics_list, callback_list):
        # Total training files count / batch_size
        batch_size = loader.batch_size
        # We can have multiple inputs
        for ind, (*inputs, targets) in enumerate(loader):
            callback_list.on_batch_begin(ind, logs={"step": step,
                                                    "batch_size": batch_size})
            if self.use_cuda:
                inputs = [tools.to_gpu(i) for i in inputs]
                targets = tools.to_gpu(targets)
            inputs, targets = [Variable(i) for i in inputs], Variable(targets)

            logits = self.learner_core.on_forward_batch(step, inputs, targets)
            metrics_logs = metrics_list(targets, logits)

            callback_list.on_batch_end(ind, logs={"step": step,
                                                  "batch_logs": self.learner_core.get_logs,
                                                  "metrics_logs": metrics_logs})
        return metrics_list

    def _run_epoch(self, train_loader, valid_loader, metrics, callback_list):

        # switch to train mode
        self.learner_core.on_train_mode()

        # Run a train pass on the current epoch
        step = "training"
        callback_list.on_epoch_begin(self.epoch_counter, {"step": step, 'epoch_count': self.epoch_counter})
        train_metrics = self._train_epoch(step, train_loader, MetricsList(metrics), callback_list)

        callback_list.on_epoch_end(self.epoch_counter, {"step": step,
                                                        'train_loss': train_loss,
                                                        'train_metrics': train_metrics})
        # switch to evaluate mode
        self.learner_core.on_eval_mode()

        # Run the validation pass
        step = "validation"
        callback_list.on_epoch_begin(self.epoch_counter, {"step": step, 'epoch_count': self.epoch_counter})
        val_loss, val_metrics = None, None
        if valid_loader:
            val_metrics = self._train_epoch(step, valid_loader, MetricsList(metrics), callback_list)

        callback_list.on_epoch_end(self.epoch_counter, {'step': step,
                                                        'train_loss': train_loss,
                                                        'train_metrics': train_metrics,
                                                        'val_loss': val_loss,
                                                        'val_metrics': val_metrics})

    def train(self, metrics, epochs, train_loader: DataLoader, valid_loader: DataLoader = None, callbacks=None):
        """
            Trains the neural net
        Args:
            metrics (list, None): Metrics to be evaluated by the model
                        during training and testing.
                        Typically you will use `metrics=['accuracy']`.
            epochs (int): number of epochs
            train_loader (DataLoader): The Dataloader for training
            valid_loader (DataLoader, optional): The Dataloader for validation
            callbacks (list, None): List of callbacks functions
        """
        train_start_time = datetime.now()
        if self.use_cuda:
            self.learner_core.to_gpu()

        if not callbacks:
            callbacks = []
        callbacks.append(train_callbacks.TQDM())

        callback_list = train_callbacks.TrainCallbackList(callbacks)
        callback_list.on_train_begin({'total_epochs': epochs,
                                      'epoch_count': self.epoch_counter,
                                      'train_loader': train_loader,
                                      'val_loader': valid_loader})

        for _ in range(epochs):
            epoch_start_time = datetime.now()
            self._run_epoch(train_loader, valid_loader, metrics, callback_list)
            print('Epoch time (hh:mm:ss.ms) {}'.format(datetime.now() - epoch_start_time))
            self.epoch_counter += 1
        callback_list.on_train_end()
        print('Total train time (hh:mm:ss.ms) {}'.format(datetime.now() - train_start_time))

    def predict(self, test_loader: DataLoader, callbacks=None):
        """
            Launch the prediction on the given loader and pass
            each predictions to the given callbacks.
        Args:
            test_loader (DataLoader): The loader containing the test dataset.
                This loader is expected to returns items with the same shape
                as the train_loader passed in train() with the difference that
                the targets will be ignored.
            callbacks (list, None): List of callbacks functions
        """
        test_start_time = datetime.now()
        # Switch to evaluation mode
        self.learner_core.on_eval_mode()

        if self.use_cuda:
            self.learner_core.to_gpu()

        if not callbacks:
            callbacks = []
        callbacks.append(test_callbacks.TQDM())

        callback_list = test_callbacks.TestCallbackList(callbacks)
        callback_list.on_test_begin({'loader': test_loader})

        ret_logits = None
        batch_size = test_loader.batch_size
        for ind, (*inputs, _) in enumerate(test_loader):
            callback_list.on_batch_begin(ind, logs={"batch_size": batch_size})
            if self.use_cuda:
                inputs = [tools.to_gpu(i) for i in inputs]

            inputs = [Variable(i, volatile=True) for i in inputs]



        callback_list.on_test_end({'loader': test_loader})
        print('Total prediction time (hh:mm:ss.ms) {}'.format(datetime.now() - test_start_time))
        return ret_logits.squeeze()

    def on_forward_batch(self, step, inputs, targets):
        # forward
        logits = self.model.forward(*inputs)
        logs = metrics_list(targets, logits)
        loss = self.crit(logits, targets)
        logs.update({"loss": loss.data[0]})
        losses.update(loss.data[0])

        # backward + optimize
        if step == "training":
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        self.logs.update(logs)