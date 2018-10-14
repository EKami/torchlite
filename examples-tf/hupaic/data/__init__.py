import os
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from typing import Union
import cv2


class Dataset:
    def __init__(self, logger, input_dir: Path, batch_size: int = 32,
                 train_val_split: float = 0.2, num_process=os.cpu_count()):
        self.num_process = num_process
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.input_dir = input_dir
        self.logger = logger

    @staticmethod
    def extract_img(input_dir, id: Union[tf.Tensor, str], filter_color="green"):
        """
        Quote from the competition notes:
            All image samples are represented by four filters (stored as individual files),
            the protein of interest (green) plus three cellular landmarks: nucleus (blue),
            microtubules (red), endoplasmic reticulum (yellow). The green filter should
            hence be used to predict the label, and the other filters are used as references.
        Args:
            id (str): The file id
            filter_color (str): red/green/blue/yellow Default: green

        Returns:

        """
        if type(id) == tf.Tensor:
            id = id.decode("UTF-8")
        file = input_dir / (id + "_" + filter_color + ".png")
        image = cv2.imread(str(file.resolve()))
        return image

    def _get_tensors(self, df, unique_lbl):
        def yield_tensors():
            for id, row in df.iterrows():
                labels_indexes = [int(label) for label in row[0].split()]
                labels = np.zeros(shape=(len(unique_lbl),))
                for idx in labels_indexes:
                    labels[idx] = 1
                yield tf.constant(id), tf.constant(labels)

        return yield_tensors

    def _get_train_val_split(self, df):
        val_size = np.ceil(len(df) * self.train_val_split).astype(np.int32)
        train_df = df.iloc[:-val_size]
        val_df = df[-val_size:]
        return train_df, val_df

    def _get_ds_pipeline(self, df, unique_lbl):
        ds = tf.data.Dataset.from_generator(self._get_tensors(df, unique_lbl),
                                            output_types=(tf.string, tf.int32),
                                            output_shapes=(tf.TensorShape(()),
                                                           tf.TensorShape(len(unique_lbl))
                                                           )
                                            )

        # This could actually result in poor performances due to the GIL
        ds = ds.map(lambda file_id, labels:
                    (file_id, tf.py_func(self.input_dir, self.extract_img,
                                         inp=[file_id], Tout=tf.uint8), labels),
                    num_parallel_calls=self.num_process)
        ds = ds.map(lambda file_id, img, labels: (file_id,  # Normalize
                                                  tf.cast(img, tf.float32) * (1. / 255), labels))
        ds = ds.batch(self.batch_size)
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
            tf.Dataset
        """
        train_df, unique_lbl = self._read_data()
        if self.train_val_split > 0:
            df_train, df_val = self._get_train_val_split(train_df)
            train_ds = self._get_ds_pipeline(df_train, unique_lbl)
            val_ds = self._get_ds_pipeline(df_val, unique_lbl)
        else:
            train_ds = self._get_ds_pipeline(train_df, unique_lbl)
            val_ds = None
        return train_ds, val_ds
