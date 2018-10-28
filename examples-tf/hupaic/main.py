"""
This file is the main file to run the hupaic project which can be found @
https://www.kaggle.com/c/human-protein-atlas-image-classification
"""
import torchlite

torchlite.set_backend(torchlite.TF)

import sys
import os
import zipfile
from pathlib import Path
import tensorflow as tf
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.api_client import ApiClient
import logging

from hupaic.data import Dataset
from hupaic.models import SimpleCnn
from torchlite.learner import Learner
from torchlite.common.data.datasets import DatasetWrapper
from torchlite.tf.learner.cores import ClassifierCore
from torchlite.tf.metrics import FBetaScore

# TODO remove on TF 2.0
# Enable eager execution
config = tf.ConfigProto(inter_op_parallelism_threads=8,
                        intra_op_parallelism_threads=8,
                        log_device_placement=True)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config)
tf.logging.set_verbosity(tf.logging.DEBUG)
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))


def getLogger():
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger


def retrieve_dataset():
    zip_files = ["train.zip", "test.zip"]
    out_dir = script_dir / ".." / "input" / "human-protein-atlas-image-classification"
    if not out_dir.exists():
        os.mkdir(out_dir)
        api = KaggleApi(ApiClient())
        api.authenticate()
        api.competition_download_files("human-protein-atlas-image-classification", out_dir, force=True, quiet=False)
        print("Extracting files...")
        for file in zip_files:
            pth = out_dir / file.split(".")[0]
            if not pth.exists():
                os.mkdir(pth)
            zip_ref = zipfile.ZipFile(out_dir / file, 'r')
            zip_ref.extractall(pth)
            zip_ref.close()
            os.remove(out_dir / file)
        print("Dataset downloaded!")
    else:
        print("Dataset already present in input dir, skipping...")
    return out_dir


def main():
    logger = getLogger()
    batch_size = 2
    epochs = 1
    metrics = [FBetaScore(logger, beta=1, average="macro", threshold=0.5)]
    callbacks = []

    # TODO the threshold on fbeta score should be calculated when the training is completely over
    # First retrieve the dataset (https://github.com/Kaggle/kaggle-api#api-credentials)
    ds_dir = retrieve_dataset()

    # Add resize otherwise the images are too big to fit in GPU memory
    ds = Dataset.construct_for_training(logger, ds_dir, batch_size, resize_imgs=(224, 224))
    train_ds, train_steps, val_ds, val_steps, unique_lbls = ds.get_dataset()
    model = SimpleCnn(logger, num_classes=len(unique_lbls))
    core = ClassifierCore(model,
                          loss_function=tf.keras.losses.binary_crossentropy,
                          optimizer=tf.train.GradientDescentOptimizer(0.0001),
                          input_index=1)
    learner = Learner(logger, core)
    learner.train(epochs, metrics, DatasetWrapper.wrap_tf_dataset(train_ds, train_steps, batch_size),
                  DatasetWrapper.wrap_tf_dataset(val_ds, val_steps, batch_size), callbacks)
    logger.info("Done!")


if __name__ == "__main__":
    main()
