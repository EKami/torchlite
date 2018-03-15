"""
This class contains a generalized learner which works across all kind of models
"""
from datetime import datetime
import torch
import numpy as np
import torchlite.nn.train_callbacks as train_callbacks
import torchlite.nn.test_callbacks as test_callbacks
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchlite.nn.tools import tensor_tools
from torchlite.nn.metrics.metrics import MetricsList
from torchlite.nn.learners.cores import BaseCore


class Learner:
    def __init__(self, learner_core: BaseCore, use_cuda=True):
        """
        The learner class used to train deep neural network
        Args:
            learner_core (BaseCore): The learner core
            use_cuda (bool): If True moves the model onto the GPU
        """
        self.learner_core = learner_core
        self.epoch_id = 1
        self.use_cuda = False
        if use_cuda:
            if torch.cuda.is_available():
                self.use_cuda = True
            else:
                print("/!\ Warning: Cuda set but not available, using CPU...")

    def _run_batch(self, step, loader, metrics_list, callback_list):
        # Total training files count / batch_size
        batch_size = loader.batch_size
        # We can have multiple inputs
        logs = {"step": step, "batch_size": batch_size}
        for ind, (*inputs, targets) in enumerate(loader):
            callback_list.on_batch_begin(ind, logs=logs)
            if self.use_cuda:
                inputs = [tensor_tools.to_gpu(i) for i in inputs]
                targets = tensor_tools.to_gpu(targets)
            inputs, targets = [Variable(i) for i in inputs], Variable(targets)

            logits = self.learner_core.on_forward_batch(step, inputs, targets)
            metrics_list.acc_batch(step, logits, targets)

            logs.update(self.learner_core.get_logs)
            logs.update({"models": self.learner_core.get_models})
            callback_list.on_batch_end(ind, logs=logs)
        return logs

    def _run_epoch(self, train_loader, valid_loader, metrics, callback_list):

        # switch to train mode
        self.learner_core.on_train_mode()

        # Run a train pass on the current epoch
        step = "training"
        logs = {"step": step, "epoch_id": self.epoch_id}
        self.learner_core.on_new_epoch()
        callback_list.on_epoch_begin(self.epoch_id, logs)

        metric_list = MetricsList(metrics)
        train_logs = self._run_batch(step, train_loader, metric_list, callback_list)

        train_logs.update(logs)
        train_logs.update({"metrics_logs": metric_list.avg(step)})
        train_logs.update({"models": self.learner_core.get_models})
        callback_list.on_epoch_end(self.epoch_id, train_logs)

        # switch to evaluate mode
        self.learner_core.on_eval_mode()

        # Run the validation pass
        if valid_loader:
            step = "validation"
            logs = {"step": step, "epoch_id": self.epoch_id}
            self.learner_core.on_new_epoch()
            callback_list.on_epoch_begin(self.epoch_id, logs)

            metric_list = MetricsList(metrics)
            val_logs = self._run_batch(step, valid_loader, metric_list, callback_list)

            val_logs.update(logs)
            val_logs.update({"metrics_logs": metric_list.avg(step)})
            val_logs.update({"models": self.learner_core.get_models})
            callback_list.on_epoch_end(self.epoch_id, val_logs)

    def train(self, epochs, metrics, train_loader: DataLoader, valid_loader: DataLoader = None, callbacks=None):
        """
            Trains the neural net
        Args:
            epochs (int): number of epochs
            metrics (list, None): Metrics to be evaluated by the model
                        during training and testing.
                        Typically you will use `metrics=['accuracy']`.
            train_loader (DataLoader): The Dataloader for training
            valid_loader (DataLoader, optional): The Dataloader for validation
            callbacks (list, None): List of train callbacks functions
        """
        train_start_time = datetime.now()
        if self.use_cuda:
            self.learner_core.to_gpu()

        if not callbacks:
            callbacks = []
        callbacks.insert(0, train_callbacks.TQDM())

        callback_list = train_callbacks.TrainCallbackList(callbacks)
        callback_list.on_train_begin({'total_epochs': epochs,
                                      'train_loader': train_loader,
                                      'val_loader': valid_loader})

        for _ in range(epochs):
            epoch_start_time = datetime.now()
            self._run_epoch(train_loader, valid_loader, metrics, callback_list)
            print('Epoch time (hh:mm:ss.ms) {}\n'.format(datetime.now() - epoch_start_time))
            self.epoch_id += 1
        callback_list.on_train_end()
        print('Total train time (hh:mm:ss.ms) {}\n'.format(datetime.now() - train_start_time))

    def predict(self, test_loader: DataLoader, callbacks=None, flatten_predictions=True):
        """
            Launch the prediction on the given loader and pass
            each predictions to the given callbacks.
        Args:
            test_loader (DataLoader): The loader containing the test dataset.
                This loader is expected to returns items with the same shape
                as the train_loader passed in train() with the difference that
                the targets will be ignored.
            callbacks (list, None): List of test callbacks functions
            flatten_predictions (bool): If True will flatten the prediction array over all batch.
            Sometimes you don't want this to happen because you may have batch predictions of different
            shapes and flattening over all the batch won't work.
        """
        test_start_time = datetime.now()
        # Switch to evaluation mode
        self.learner_core.on_eval_mode()

        if self.use_cuda:
            self.learner_core.to_gpu()

        if not callbacks:
            callbacks = []
        callbacks.insert(0, test_callbacks.TQDM())

        callback_list = test_callbacks.TestCallbackList(callbacks)
        callback_list.on_test_begin({'loader': test_loader})

        ret_logits = []
        batch_size = test_loader.batch_size
        for ind, (*inputs, _) in enumerate(test_loader):
            callback_list.on_batch_begin(ind, logs={"batch_size": batch_size})
            if self.use_cuda:
                inputs = [tensor_tools.to_gpu(i) for i in inputs]

            inputs = [Variable(i, volatile=True) for i in inputs]

            logits = self.learner_core.on_forward_batch("prediction", inputs).data
            ret_logits.append(logits)
            callback_list.on_batch_end(ind, logs={"batch_size": batch_size})

        if flatten_predictions:
            ret_logits = np.array([pred.view(-1).cpu().numpy() for sublist in ret_logits for pred in sublist])
        callback_list.on_test_end({'loader': test_loader})
        print('Total prediction time (hh:mm:ss.ms) {}\n'.format(datetime.now() - test_start_time))
        return ret_logits
