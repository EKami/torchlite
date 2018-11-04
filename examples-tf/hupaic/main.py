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
import argparse

from hupaic.data import Dataset
from hupaic import models
from torchlite.common.learner import Learner
from torchlite.common.data.datasets import DatasetWrapper
from torchlite.tf.learner.cores import ClassifierCore
from torchlite.tf.metrics import FBetaScore
from torchlite.tf.train_callbacks import ModelSaverCallback

# TODO remove on TF 2.0
# Enable eager execution
config = tf.ConfigProto(inter_op_parallelism_threads=os.cpu_count(),
                        intra_op_parallelism_threads=os.cpu_count(),
                        log_device_placement=True)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config)
tf.logging.set_verbosity(tf.logging.DEBUG)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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


def retrieve_dataset(input_dir):
    input_dir = Path(input_dir)
    zip_files = ["train.zip", "test.zip"]
    if not input_dir.exists():
        os.makedirs(input_dir)
        api = KaggleApi(ApiClient())
        api.authenticate()
        api.competition_download_files("human-protein-atlas-image-classification",
                                       input_dir, force=True, quiet=False)
        print("Extracting files...")
        for file in zip_files:
            pth = input_dir / file.split(".")[0]
            if not pth.exists():
                os.mkdir(pth)
            zip_ref = zipfile.ZipFile(input_dir / file, 'r')
            zip_ref.extractall(pth)
            zip_ref.close()
            os.remove(input_dir / file)
        print("Dataset downloaded!")
    else:
        print("Dataset already present in input dir, skipping...")
    return input_dir


def train(batch_size, epochs, resize_imgs, input_dir, output_dir, model_name):
    input_shape = (resize_imgs, resize_imgs)
    logger = getLogger()
    # Resize to half the original size
    metrics = [FBetaScore(logger, beta=1, average="macro", threshold=0.5)]
    callbacks = [ModelSaverCallback(logger, output_dir)]

    # TODO the threshold on fbeta score should be calculated when the training is completely over
    # First retrieve the dataset (https://github.com/Kaggle/kaggle-api#api-credentials)
    input_dir = retrieve_dataset(input_dir)

    ds = Dataset.construct_for_training(logger, input_dir, batch_size, resize_imgs=(*input_shape, 3))
    train_ds, train_steps, val_ds, val_steps, unique_lbls = ds.get_dataset()
    model_class_ = getattr(models, model_name)
    model = model_class_(logger, num_classes=len(unique_lbls), input_shape=input_shape)
    core = ClassifierCore(model,
                          loss_function=tf.keras.losses.binary_crossentropy,
                          optimizer=tf.train.GradientDescentOptimizer(0.0001),
                          input_index=1)
    learner = Learner(logger, core)
    learner.train(epochs, metrics, DatasetWrapper.wrap_tf_dataset(train_ds, train_steps, batch_size),
                  DatasetWrapper.wrap_tf_dataset(val_ds, val_steps, batch_size), callbacks)
    logger.info("Done!")


def eval():
    pass


def main(args):
    if args.mode == "train":
        train(args.batch_size, args.epochs, args.resize, args.input, args.output, args.model)
    else:
        eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the neural network in train/eval mode')
    subs = parser.add_subparsers(dest='mode')

    train_parser = subs.add_parser('train', help='Use this script in train mode')
    eval_parser = subs.add_parser('eval', help='Use this script in evaluation mode')

    # Train mode
    train_parser.add_argument("--input", default=str(script_dir / ".." / "input" /
                                                     "human-protein-atlas-image-classification"),
                              help="The folder containing the input data "
                                   "(defaults to ../input/human-protein-atlas-image-classification/)", type=str)
    train_parser.add_argument("--output", default=str(script_dir / ".." / "output" /
                                                      "human-protein-atlas-image-classification"),
                              help="The output folder where the NN weights and results will be saved "
                                   "(defaults to ../output/human-protein-atlas-image-classification/)", type=str)
    train_parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
    train_parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    train_parser.add_argument('--model', default=models.SimpleCnn.__name__, type=str,
                              help='The model name to use for training')
    train_parser.add_argument('--resize', default=256, type=int,
                              help='Training images size. '
                                   'A single number e.g: 128 will result in a 128x128 resize')

    # Eval mode
    eval_parser.add_argument("--input", default=str(script_dir / ".." / "output" /
                                                    "human-protein-atlas-image-classification"),
                             help="The input folder where the model and the trained weights are located. "
                                  "This folder will also serve as an output folder to store the results. "
                                  "It's typically the output folder of the train mode and set to it by default.",
                             type=str)
    eval_parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    eval_parser.add_argument('--model', default=models.SimpleCnn.__name__, type=str,
                             help='The model name to use for inference')

    kwargs = parser.parse_args()
    main(kwargs)
