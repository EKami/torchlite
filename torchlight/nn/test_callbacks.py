"""
This module contains callbacks used during the test phase.
"""
from torchlight.data.datasets import ImagesDataset
from tqdm import tqdm
from collections import OrderedDict


class TestCallback:
    def __init__(self):
        self.validation_data = None
        self.params = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class TestCallbackList(object):
    """Container abstracting a list of callbacks.
    Args:
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        assert isinstance(callback, TestCallback), f"Your callback is not an instance of TestCallback: {callback}"
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_test_begin(self, logs=None):
        """Called at the beginning of testing.
        Args:
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        """Called at the end of testing.
        Args:
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.
        Args:
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.
        Args:
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def __iter__(self):
        return iter(self.callbacks)


class TQDM(TestCallback):
    def __init__(self):
        super().__init__()
        self.pbar = None

    def on_test_begin(self, logs=None):
        test_loader_len = len(logs["loader"])
        self.pbar = tqdm(total=test_loader_len, desc="Classifying")

    def on_batch_end(self, batch, logs=None):
        self.pbar.update(1)


class ActivationMapVisualizerCallback(TestCallback):
    def __init__(self, filename):
        """
            Store an image with heatmap activations in a heatmaps list
            using the Grad_cam++ technique: https://arxiv.org/abs/1710.11063
            # TODO may combines with TensorboardVisualizer?

            /!\ This technique only works with image torchlight.data.datasets.ImagesDataset
        Args:
            filename (str): The file name that you want to visualize
        """
        super().__init__()
        self.filename = filename
        self.heatmap = None

    def on_test_end(self, logs=None):
        model = self.model
        ds = logs["loader"].dataset if logs["loader"] else None
        assert isinstance(ds, ImagesDataset), \
            "ActivationMapVisualizer: The loader is not an instance of torchlight.data.datasets.ImagesDataset"
        image, label, _ = ds.get_by_name(self.filename)

    @property
    def get_heatmap(self):
        return self.heatmap


class TTACallback(TestCallback):
    def __init__(self):
        """
        Test time augmentation callback
        """
        # TODO implement https://github.com/fastai/fastai/blob/master/fastai/learner.py#L242
        super().__init__()
