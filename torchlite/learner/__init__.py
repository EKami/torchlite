"""
This class contains a generalized learner which works across all kind of models
"""
import torchlite
import importlib
from datetime import datetime
import numpy as np

import torchlite.common.test_callbacks
import torchlite.common.train_callbacks

from torchlite.common.metrics import MetricsList
from torchlite.learner.cores import BaseCore
from torchlite.common.data.datasets import DatasetWrapper


class Learner:
    def __init__(self, logger, learner_core: BaseCore, use_gpu=True):
        """
        The learner class used to train deep neural network
        Args:
            logger (Logger): A python logger
            learner_core (BaseCore): The learner core
            use_gpu (bool): If True moves the model onto the GPU
        """
        self.logger = logger
        self.learner_core = learner_core
        self.epoch_id = 1

        if torchlite.backend == "torch":
            torch = importlib.import_module("torch")
            self.device = torch.device("cpu")
            if use_gpu:
                if torch.cuda.is_available():
                    device = "cuda:0"
                else:
                    device = "cpu"
                    logger.info("/!\ Warning: Cuda set but not available, using CPU...")
                self.device = torch.device(device)
        else:
            if use_gpu:
                self.device = "/device:GPU:0"
            else:
                self.device = "/cpu:0"

    @classmethod
    def convert_data_structure(cls, structure, action):
        """
        This method should recursively iterate over data structure and move all Tensors to the selected device
        Args:
            structure (Any): A structure like a Tensor
            action (callable): Action to do with the Tensor

        Returns:
            structure, action
        """

        if torchlite.backend == torchlite.TORCH:
            torch = importlib.import_module("torch")
            if isinstance(structure, torch.Tensor):
                return action(structure)
            elif isinstance(structure, np.ndarray):
                return action(torch.from_numpy(structure))
            elif isinstance(structure, (list, tuple)):
                return [cls.convert_data_structure(x, action) for x in structure]
            elif isinstance(structure, dict):
                return dict((k, cls.convert_data_structure(v, action)) for k, v in structure.items())
            else:
                return structure  # can't deal with anything else
        else:
            return structure

    def _run_batch(self, step, dataset, metrics_list, callback_list):
        # Total training files count / batch_size
        batch_size = dataset.get_batch_size
        # We can have multiple inputs
        logs = {"step": step, "batch_size": batch_size}
        for ind, (*inputs, y_true) in enumerate(dataset):
            callback_list.on_batch_begin(ind, logs=logs)

            inputs = self.convert_data_structure(inputs, action=lambda x: x.to(self.device))
            y_true = self.convert_data_structure(y_true, action=lambda x: x.to(self.device))

            # Need to detach otherwise the Tensor gradients will accumulate in GPU memory
            y_pred = self.learner_core.on_forward_batch(step, inputs, y_true)
            y_pred = self.convert_data_structure(y_pred, action=lambda x: x.detach())

            metrics_list.acc_batch(step, y_true, y_pred)

            logs.update(self.learner_core.get_logs)
            logs.update({"models": self.learner_core.get_models})
            callback_list.on_batch_end(ind, logs=logs)
        return logs

    def _run_epoch(self, train_ds, valid_ds, metrics, callback_list):

        # switch to train mode
        self.learner_core.on_train_mode()

        # Run a train pass on the current epoch
        step = "training"
        logs = {"step": step, "epoch_id": self.epoch_id}
        self.learner_core.on_new_epoch()
        callback_list.on_epoch_begin(self.epoch_id, logs)

        metric_list = MetricsList(metrics)
        train_logs = self._run_batch(step, train_ds, metric_list, callback_list)

        train_logs.update(logs)
        train_logs.update({"metrics_logs": metric_list.avg(step)})
        train_logs.update({"models": self.learner_core.get_models})
        callback_list.on_epoch_end(self.epoch_id, train_logs)

        # switch to evaluate mode
        self.learner_core.on_eval_mode()

        # Run the validation pass
        if valid_ds:
            step = "validation"
            logs = {"step": step, "epoch_id": self.epoch_id}
            self.learner_core.on_new_epoch()
            callback_list.on_epoch_begin(self.epoch_id, logs)

            metric_list = MetricsList(metrics)
            val_logs = self._run_batch(step, valid_ds, metric_list, callback_list)

            val_logs.update(logs)
            val_logs.update({"metrics_logs": metric_list.avg(step)})
            val_logs.update({"models": self.learner_core.get_models})
            callback_list.on_epoch_end(self.epoch_id, val_logs)

    def train(self, epochs, metrics, train_ds: DatasetWrapper, valid_ds: DatasetWrapper = None, callbacks=None):
        """
            Trains the neural net
        Args:
            epochs (int): number of epochs
            metrics (list, None): Metrics to be evaluated by the model during training and validation
            train_ds (DatasetWrapper): A DatasetWrapper object for training
            valid_ds (DatasetWrapper, optional): A DatasetWrapper object for validation
            callbacks (list, None): List of train callbacks functions
        """
        train_start_time = datetime.now()
        self.learner_core.to_device(self.device)

        if not callbacks:
            callbacks = []
        callbacks.insert(0, torchlite.common.train_callbacks.TQDM())

        callback_list = torchlite.common.train_callbacks.TrainCallbackList(callbacks)
        callback_list.on_train_begin({'total_epochs': epochs,
                                      'train_steps': len(train_ds),
                                      'val_steps': len(valid_ds)})

        for _ in range(epochs):
            epoch_start_time = datetime.now()
            self._run_epoch(train_ds, valid_ds, metrics, callback_list)
            self.logger.info('Epoch time (hh:mm:ss.ms) {}\n'.format(datetime.now() - epoch_start_time))
            self.epoch_id += 1
        callback_list.on_train_end()
        self.logger.info('Total train time (hh:mm:ss.ms) {}\n'.format(datetime.now() - train_start_time))

    def _pred_loop(self, test_loader, callback_list, batch_size):
        ret_logits = []

        for ind, (*inputs, _) in enumerate(test_loader):
            callback_list.on_batch_begin(ind, logs={"batch_size": batch_size})
            inputs = [i.to(self.device) for i in inputs]

            # Need to detach and move to CPU otherwise the Tensor and gradients will accumulate in GPU memory
            logits = self.learner_core.on_forward_batch("prediction", inputs).cpu().detach()
            ret_logits.append(logits)
            callback_list.on_batch_end(ind, logs={"batch_size": batch_size})
        return ret_logits

    def predict(self, test_ds, callbacks=None, flatten_predictions=True):
        """
            Launch the prediction on the given loader and pass
            each predictions to the given callbacks.
        Args:
            test_ds (torch.utils.data.DataLoader, tf.data.Dataset): The loader/dataset containing the test dataset.
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
        self.learner_core.to_device(self.device)

        if not callbacks:
            callbacks = []
        callbacks.insert(0, torchlite.common.test_callbacks.TQDM())

        callback_list = torchlite.common.test_callbacks.TestCallbackList(callbacks)
        callback_list.on_test_begin({'loader': test_ds})

        if torchlite.backend == "torch":
            torch = importlib.import_module("torch")
            with torch.no_grad():
                batch_size = test_ds.batch_size
                ret_logits = self._pred_loop(test_ds, callback_list, batch_size)
        else:
            # TODO finish with Tensorflow
            ret_logits = []

        if flatten_predictions:
            ret_logits = np.array([pred.view(-1).cpu().numpy() for sublist in ret_logits for pred in sublist])
        callback_list.on_test_end({'loader': test_ds})
        self.logger.info('Total prediction time (hh:mm:ss.ms) {}\n'.format(datetime.now() - test_start_time))
        return ret_logits
