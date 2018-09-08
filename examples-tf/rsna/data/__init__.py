import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf


class Dataset:
    def __init__(self, logger, input_dir: Path):
        self.input_dir = input_dir
        self.logger = logger

    def _get_tensors(self, df, lung_class_count):
        def yield_tensors():
            for i, patient in df.iterrows():
                lung_class = np.zeros((lung_class_count,), dtype=np.bool)
                lung_class[patient["class"]] = 1
                yield (tf.constant(patient["patientId"]), tf.constant(patient["x"]),
                       tf.constant(patient["y"]), tf.constant(patient["width"]),
                       tf.constant(patient["height"]), tf.constant(lung_class),
                       tf.constant(patient["Target"]))

        return yield_tensors

    def _read_training_data(self):
        labels_train_df = pd.read_csv(self.input_dir / "stage_1_train_labels.csv", )
        class_info_train_df = pd.read_csv(self.input_dir / "stage_1_detailed_class_info.csv")
        df = pd.merge(labels_train_df, class_info_train_df, on="patientId")
        df = df.fillna(-1)
        df = df.astype(dtype={'patientId': str, 'x': np.int32, 'y': np.int32, 'width': np.int32,
                              'height': np.int32, 'Target': np.int32, 'class': 'category'})
        df["class"] = pd.factorize(df["class"], sort=True)[0] + 1
        lung_class_count = len(df["class"].unique())
        return df, lung_class_count

    def _extract_bbox(self, patient_id, pos_x, pos_y, width, height):

        pass

    def get_dataset(self):
        """
        Returns a tf.data.Dataset pipeline
        https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428

        Interesting optimization:
            https://stackoverflow.com/a/40985375/1334473
        Returns:
            tf.Dataset
        """
        df, lung_class_count = self._read_training_data()
        ds = tf.data.Dataset.from_generator(self._get_tensors(df, lung_class_count),
                                            output_types=(tf.string, tf.int32, tf.int32, tf.int32, tf.int32,
                                                          tf.bool, tf.bool),
                                            output_shapes=(tf.TensorShape(()),
                                                           tf.TensorShape(()),
                                                           tf.TensorShape(()),
                                                           tf.TensorShape(()),
                                                           tf.TensorShape(()),
                                                           tf.TensorShape((lung_class_count,)),
                                                           tf.TensorShape(()))
                                            )
        ds = ds.map(lambda patient_id, pos_x, pos_y, width, height, lung_class, target:
                    (patient_id, tf.py_func(self._extract_bbox, inp=[patient_id, pos_x, pos_y, width, height],
                                            Tout=tf.uint8), lung_class, target),
                    num_parallel_calls=None)
        return ds
