import os
from datetime import datetime
import json
from pathlib import Path
from logging import Logger
from typing import Union
from torchlite.common.train_callbacks import TrainCallback


class ModelSaverCallback(TrainCallback):
    def __init__(self, logger: Logger, weights_pth: Union[str, Path]):
        """
        Save the Keras models of the learner at each epoch
        Args:
            logger (Logger): A python logger
            weights_pth (Path): Path to the directory where to save the model weights
        """
        super().__init__()
        self.weights_pth = Path(weights_pth)
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists(self.weights_pth):
            os.makedirs(self.weights_pth)
        for model_name, model in logs["models"].items():
            meta = {"date": str(datetime.now()), "name": model_name, "last_epoch": str(epoch)}
            model.save_weights(str((self.weights_pth / (model_name + "-epoch_" + str(epoch))).resolve()))
            model.save_weights(str((self.weights_pth / model_name).resolve()))
            meta_pth = (self.weights_pth / (model_name + "_meta.json")).resolve()
            with open(meta_pth, 'w') as outfile:
                json.dump(meta, outfile)
