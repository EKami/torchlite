from torch.utils.data.dataloader import DataLoader
import tensorflow as tf


class DatasetWrapper:
    def __init__(self, dataset, steps=None, batch_size=None):
        self.batch_size = batch_size
        self.steps = steps
        self.dataset = dataset

    @classmethod
    def wrap_tf_dataset(cls, dataset: tf.data.Dataset, steps, batch_size):
        """
            A Dataset abstraction around tf.data.Dataset
        Args:
            batch_size (int): Just for indicative purposes, don't actually apply it to the passed dataset
            dataset (tf.data.Dataset): The tf.data.Dataset, yielding batches of data
            steps (int): Number of steps to iterate over the dataset before no data is left
        """
        return cls(dataset, steps, batch_size)

    @classmethod
    def wrap_torch_dataloader(cls, dataloader: DataLoader):
        """
        A Dataset abstraction around Pytorch Dataloaders
        Args:
            dataloader (torch.utils.data.DataLoader): A Pytorch Dataloader

        Returns:

        """
        return cls(dataloader)

    @property
    def get_batch_size(self):
        if isinstance(self.dataset, DataLoader):
            return self.dataset.batch_size
        else:
            return self.batch_size

    def __len__(self):
        if isinstance(self.dataset, DataLoader):
            return len(self.dataset)
        else:
            return self.steps

    def __iter__(self):
        return self.dataset.__iter__()

    def __next__(self):
        return self.dataset.__next__()


