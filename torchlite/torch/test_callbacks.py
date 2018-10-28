"""
This module contains callbacks used during the test phase.
"""
from torchlite.common.test_callbacks import TestCallback
from torchlite.torch.datasets import ImageDataset


class ActivationMapVisualizerCallback(TestCallback):
    def __init__(self, filename):
        """
            Store an image with heatmap activations in a heatmaps list
            using the Grad_cam++ technique: https://arxiv.org/abs/1710.11063
            # TODO may combines with TensorboardVisualizer?

            /!\ This technique only works with image torchlite.data.datasets.ImagesDataset
        Args:
            filename (str): The file name that you want to visualize
        """
        super().__init__()
        self.filename = filename
        self.heatmap = None

    def on_test_end(self, logs=None):
        model = self.model
        ds = logs["loader"].dataset if logs["loader"] else None
        assert isinstance(ds, ImageDataset), \
            "ActivationMapVisualizer: The loader is not an instance of torchlite.data.datasets.ImagesDataset"
        image, label, _ = ds.get_by_name(self.filename)
        # TODO finish grad cam here https://github.com/adityac94/Grad_CAM_plus_plus/blob/master/misc/utils.py#L51

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
