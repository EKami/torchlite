import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
import pydicom


class Dataset:
    def __init__(self, logger, input_dir: Path, batch_size: int = 32, train_val_split: float = 0.2):
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.input_dir = input_dir
        self.logger = logger

    def _get_tensors(self, df, lung_class_count):
        def yield_tensors():
            for i, patient in df.iterrows():
                lung_class = np.zeros((len(lung_class_count),), dtype=np.bool)
                lung_class[patient["class"]] = 1
                yield (tf.constant(patient["patientId"]),
                       tf.constant([patient["x"], patient["y"], patient["width"], patient["height"]]),
                       tf.constant(lung_class), tf.constant(patient["Target"], dtype=bool))

        return yield_tensors

    def read_training_data(self):
        labels_train_df = pd.read_csv(self.input_dir / "stage_1_train_labels.csv", )
        class_info_train_df = pd.read_csv(self.input_dir / "stage_1_detailed_class_info.csv")
        df = pd.merge(labels_train_df, class_info_train_df, on="patientId")
        df = df.fillna(-1)
        df = df.astype(dtype={'patientId': str, 'x': np.int32, 'y': np.int32, 'width': np.int32,
                              'height': np.int32, 'Target': np.int32, 'class': 'category'})
        df["class"], lung_class_ltable = pd.factorize(df["class"], sort=True)
        return df, lung_class_ltable

    def _extract_img(self, patient_id, bbox_pos):
        patient_id = patient_id.decode("utf-8")
        d = pydicom.read_file(str((self.input_dir / (patient_id + ".dcm")).resolve()))

        # Greyscale image (appends 1 dim to array)
        im = d.pixel_array[:, :, None].shape
        return tf.constant(im, dtype=tf.uint8)

    def _get_train_val_split(self, df):
        val_size = np.ceil(len(df) * self.train_val_split).astype(np.int32)
        train_df = df.iloc[:-val_size]
        val_df = df[-val_size:]
        return train_df, val_df

    def _get_ds_pipeline(self, df, lung_class_ltable):
        ds = tf.data.Dataset.from_generator(self._get_tensors(df, lung_class_ltable),
                                            output_types=(tf.string, tf.int32, tf.bool, tf.bool),
                                            output_shapes=(tf.TensorShape(()),
                                                           tf.TensorShape((4,)),
                                                           tf.TensorShape((len(lung_class_ltable),)),
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

    def get_dataset(self):
        """
        Returns a tf.data.Dataset pipeline
        https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428

        Interesting optimization:
            https://stackoverflow.com/a/40985375/1334473
        Returns:
            tf.Dataset
        """
        df, lung_class_ltable = self.read_training_data()
        if self.train_val_split > 0:
            df_train, df_val = self._get_train_val_split(df)
            train_ds = self._get_ds_pipeline(df_train, lung_class_ltable)
            val_ds = self._get_ds_pipeline(df_val, lung_class_ltable)
        else:
            train_ds = self._get_ds_pipeline(df, lung_class_ltable)
            val_ds = None
        return train_ds, val_ds
