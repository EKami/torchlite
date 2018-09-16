"""
This file is the main file to run the tsg project which can be found @
https://www.kaggle.com/c/tgs-salt-identification-challenge
"""
import torchlite
torchlite.set_backend("tensorflow")

import sys
import os
import zipfile
from pathlib import Path
import tensorflow as tf
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.api_client import ApiClient
import logging

from tsg.data import Dataset
from tsg.models.cores import RsnaCore
from tsg.models.simple_cnn import SimpleCnn
from torchlite.learner import Learner


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
    out_dir = script_dir / ".." / "input" / "tgs-salt-identification-challenge"
    if not out_dir.exists():
        os.mkdir(out_dir)
        api = KaggleApi(ApiClient())
        api.authenticate()
        api.competition_download_files("tgs-salt-identification-challenge", out_dir, force=True, quiet=False)
        print("Extracting files...")
        for file in zip_files:
            zip_ref = zipfile.ZipFile(out_dir / file, 'r')
            zip_ref.extractall(out_dir)
            zip_ref.close()
            os.remove(out_dir / file)
        print("Dataset downloaded!")
    else:
        print("Dataset already present in input dir, skipping...")
    return out_dir


def main():
    batch_size = 32
    epochs = 1
    num_classes = 2

    logger = getLogger()

    # First retrieve the dataset (https://github.com/Kaggle/kaggle-api#api-credentials)
    ds_dir = retrieve_dataset()

    ds = Dataset(logger, ds_dir, batch_size)
    train_ds, val_ds = ds.get_dataset()
    core = RsnaCore()
    model = SimpleCnn(logger, num_classes)

    for batch in train_ds.make_one_shot_iterator():
        d = 0
    learner = Learner(logger, core)

    print("Done!")


if __name__ == "__main__":
    main()
