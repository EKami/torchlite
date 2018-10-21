import os
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import RepeatedKFold
import cv2


class Dataset:
    def __init__(self, logger, input_dir: Path, batch_size: int,
                 train_val_split: float, mode, resize_imgs, num_process):
        self.resize_imgs = resize_imgs if resize_imgs is not None else (-1, -1)
        self.mode = mode
        self.num_process = num_process
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.input_dir = input_dir
        self.logger = logger

    @classmethod
    def construct_for_training(cls, logger, input_dir: Path, batch_size: int = 32,
                               train_val_split: float = 0.2, resize_imgs=None, num_process=os.cpu_count()):
        return cls(logger, input_dir, batch_size, train_val_split, "train", resize_imgs, num_process)

    def _get_image_pth(self, id, filter_color="green"):
        return str(self.input_dir) + "/" + (id + "_" + filter_color + ".png")

    @staticmethod
    def extract_img(input_dir: tf.Tensor, id: tf.Tensor, filter_color: tf.Tensor, resize: tuple):
        """
        Quote from the competition notes:
            All image samples are represented by four filters (stored as individual files),
            the protein of interest (green) plus three cellular landmarks: nucleus (blue),
            microtubules (red), endoplasmic reticulum (yellow). The green filter should
            hence be used to predict the label, and the other filters are used as references.
        Args:
            resize (tuple, None): Resize size or None to not resize
            input_dir (Path): The input dir
            id (tf.Tensor): The file id
            filter_color (tf.Tensor): red/green/blue/yellow Default: green

        Returns:
            np.array: The image
        """
        id = id.decode("UTF-8")
        input_dir = input_dir.decode("UTF-8")
        filter_color = filter_color.decode("UTF-8")
        file = Path(input_dir) / (id + "_" + filter_color + ".png")
        image = cv2.imread(str(file.resolve()))
        if resize[0] != -1:
            image = cv2.resize(image, dsize=(resize[1], resize[0]), interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def _get_tensors(df, unique_lbl):
        def yield_tensors():
            for id, row in df.iterrows():
                labels_indexes = [int(label) for label in row[0].split()]
                labels = np.zeros(shape=(len(unique_lbl),))
                for idx in labels_indexes:
                    labels[idx] = 1
                yield tf.constant(id), tf.constant(labels)

        return yield_tensors

    @staticmethod
    def _get_train_val_split(df, train_val_split):
        # Add repeated K-fold
        # splitter = RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)
        val_size = np.ceil(len(df) * train_val_split).astype(np.int32)
        train_df = df.iloc[:-val_size]
        val_df = df[-val_size:]
        return train_df, val_df

    @staticmethod
    def _get_ds_pipeline(df, input_dir, mode, unique_lbl, batch_size, num_process, resize_imgs):
        ds = tf.data.Dataset.from_generator(Dataset._get_tensors(df, unique_lbl),
                                            output_types=(tf.string, tf.int32),
                                            output_shapes=(tf.TensorShape(()),
                                                           tf.TensorShape(len(unique_lbl))
                                                           )
                                            )

        step_input_dir = str(input_dir / mode)
        # This could actually result in poor performances due to the GIL
        ds = ds.map(lambda file_id, labels: (file_id, tf.py_func(Dataset.extract_img,
                                                                 inp=[step_input_dir, file_id, "green", resize_imgs],
                                                                 Tout=tf.uint8), labels))
        ds = ds.map(lambda file_id, img, labels: (file_id,  # Normalize
                                                  tf.cast(img, tf.float32) * (1. / 255),
                                                  tf.cast(labels, tf.float32)))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(1)
        return ds

    def _read_data(self):
        train_df = pd.read_csv(self.input_dir / "train.csv", index_col="Id")
        unique_lbl = list(set([int(label) for labels in train_df["Target"].iteritems()
                               for label in labels[1].split()]))
        return train_df, unique_lbl

    def get_dataset(self):
        """
        Returns a tf.data.Dataset pipeline
        https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428

        Interesting optimization:
            https://stackoverflow.com/a/40985375/1334473
        Returns:
            tf.Dataset, int, Union[tf.Dataset, None], int
        """
        train_df, unique_lbls = self._read_data()
        if self.train_val_split > 0:
            df_train, df_val = Dataset._get_train_val_split(train_df, self.train_val_split)
            train_steps = len(df_train) // self.batch_size
            val_steps = len(df_val) // self.batch_size
            train_ds = self._get_ds_pipeline(df_train, self.input_dir, self.mode,
                                             unique_lbls, self.batch_size, self.num_process,
                                             self.resize_imgs)
            val_ds = self._get_ds_pipeline(df_val, self.input_dir, self.mode,
                                           unique_lbls, self.batch_size, self.num_process,
                                           self.resize_imgs)
        else:
            train_ds = self._get_ds_pipeline(train_df, self.input_dir, self.mode,
                                             unique_lbls, self.batch_size, self.num_process,
                                             self.resize_imgs)
            val_ds = None
            train_steps = len(train_df) // self.batch_size
            val_steps = None

        return train_ds, train_steps, val_ds, val_steps, unique_lbls
