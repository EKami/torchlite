import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf


class Dataset:
    def __init__(self, logger, input_dir: Path, batch_size: int = 32, train_val_split: float = 0.2):
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.input_dir = input_dir
        self.logger = logger

    def _get_tensors(self, df):
        def yield_tensors():
            for i, row in df.iterrows():
                d = 0
                rle_mask = rle_mask.split(" ")
                yield (tf.constant(patient["patientId"]),
                       tf.constant([patient["x"], patient["y"], patient["width"], patient["height"]]),
                       tf.constant(lung_class), tf.constant(patient["Target"], dtype=bool))

        return yield_tensors

    def _extract_img(self, id, bbox_pos):
        im = None
        return tf.constant(im, dtype=tf.uint8)

    def _get_train_val_split(self, df):
        val_size = np.ceil(len(df) * self.train_val_split).astype(np.int32)
        train_df = df.iloc[:-val_size]
        val_df = df[-val_size:]
        return train_df, val_df

    def _get_ds_pipeline(self, df):
        ds = tf.data.Dataset.from_generator(self._get_tensors(df),
                                            output_types=(tf.string, tf.int32, tf.bool, tf.bool),
                                            output_shapes=(tf.TensorShape(()),
                                                           tf.TensorShape((4,)),
                                                           tf.TensorShape(()),
                                                           tf.TensorShape(()))
                                            )
        ds = ds.map(lambda patient_id, bbox_pos, lung_class, target:
                    (patient_id, tf.py_func(self._extract_img, inp=[patient_id, bbox_pos],
                                            Tout=tf.uint8), lung_class, target),
                    num_parallel_calls=None)
        ds = ds.map(lambda patient_id, img, lung_class, target: (patient_id,  # Normalize
                                                                 tf.cast(img, tf.float32) *
                                                                 (1. / 255), lung_class, target))
        ds = ds.batch(self.batch_size)
        return ds

    def _read_data(self):
        train_df = pd.read_csv(self.input_dir / "train.csv", index_col="id")
        depths_df = pd.read_csv(self.input_dir / "depths.csv", index_col="id")
        train_df = train_df.merge(depths_df, on="id")
        test_df = depths_df[~depths_df.index.isin(train_df.index)]
        return train_df, test_df

    def get_dataset(self):
        """
        Returns a tf.data.Dataset pipeline
        https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428

        Interesting optimization:
            https://stackoverflow.com/a/40985375/1334473
        Returns:
            tf.Dataset
        """
        train_df, test_df = self._read_data()
        if self.train_val_split > 0:
            df_train, df_val = self._get_train_val_split(train_df)
            train_ds = self._get_ds_pipeline(df_train)
            val_ds = self._get_ds_pipeline(df_val)
        else:
            train_ds = self._get_ds_pipeline(train_df)
            val_ds = None
        return train_ds, val_ds
