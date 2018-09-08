from pathlib import Path
import tensorflow as tf


class Dataset:
    def __init__(self, logger, input_dir: Path):
        self.input_dir = input_dir
        self.logger = logger

    def get_dataset(self, labels_table=None):
        """
        Returns a tf.data.Dataset pipeline
        https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428

        Interesting optimization:
            https://stackoverflow.com/a/40985375/1334473
        Args:
            labels_table (tf.python.ops.lookup_ops.HashTable, None):
                If given this lookup table will be used to encode onehot labels
        Returns:

        """
        return None
