import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn import tools
from nn.metrics import MetricsList
from .callbacks import CallbackList, TQDM


class Classifier:
    def __init__(self, net, use_cuda=torch.cuda.is_available()):
        self.net = net
        self.use_cuda = use_cuda
        self.epoch_counter = 0

    def restore_model(self, model_path):
        """
            Restore a model parameters from the one given in argument
        Args:
            model_path (str): The path to the model to restore

        """
        self.net.load_state_dict(torch.load(model_path))

    def _validate_epoch(self, valid_loader, criterion, metrics_list, callback_list):

        it_count = len(valid_loader)
        batch_size = valid_loader.batch_size
        losses = tools.AverageMeter()
        for ind, (inputs, targets) in enumerate(valid_loader):
            callback_list.on_batch_begin(ind, logs={"step": "validation",
                                                    "batch_size": batch_size})
            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Volatile because we are in pure inference mode
            # http://pytorch.org/docs/master/notes/autograd.html#volatile
            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True)

            # forward
            logits = self.net(inputs)

            loss = criterion(logits, targets)
            losses.update(loss.data[0], batch_size)
            logs = metrics_list(targets, logits)
            callback_list.on_batch_end(ind, logs={"step": "validation",
                                                  "metrics": logs,
                                                  "iterations_count": it_count})

        return losses.debias_loss, metrics_list

    def _train_epoch(self, train_loader, optimizer, criterion, metrics_list, callback_list):
        # Total training files count / batch_size
        batch_size = train_loader.batch_size
        it_count = len(train_loader)
        losses = tools.AverageMeter()
        for ind, (inputs, targets) in enumerate(train_loader):
            callback_list.on_batch_begin(ind, logs={"step": "training",
                                                    "batch_size": batch_size})
            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            # forward
            logits = self.net.forward(inputs)

            # backward + optimize
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.data[0], batch_size)
            logs = metrics_list(targets, logits)

            callback_list.on_batch_end(ind, logs={"step": "training",
                                                  "loss": loss.data[0],
                                                  "metrics": logs,
                                                  "iterations_count": it_count})
        return losses.debias_loss, metrics_list

    def _run_epoch(self, train_loader, valid_loader, optimizer, loss, metrics, callback_list):

        # switch to train mode
        self.net.train()

        # Run a train pass on the current epoch
        callback_list.on_epoch_begin(self.epoch_counter, {"step": "training", 'epoch_count': self.epoch_counter})
        train_loss, train_metrics = self._train_epoch(train_loader, optimizer, loss, MetricsList(metrics),
                                                      callback_list)

        callback_list.on_epoch_end(self.epoch_counter, {"step": "training",
                                                        'train_loss': train_loss,
                                                        'train_metrics': train_metrics})
        # switch to evaluate mode
        self.net.eval()

        # Run the validation pass
        callback_list.on_epoch_begin(self.epoch_counter, {"step": "validation", 'epoch_count': self.epoch_counter})
        val_loss, val_metrics = None, None
        if valid_loader:
            val_loss, val_metrics = self._validate_epoch(valid_loader, loss, MetricsList(metrics), callback_list)

        callback_list.on_epoch_end(self.epoch_counter, {'step': 'validation',
                                                        'train_loss': train_loss,
                                                        'train_metrics': train_metrics,
                                                        'val_loss': val_loss,
                                                        'val_metrics': val_metrics})

    def train(self, optimizer, loss, metrics, epochs,
              train_loader: DataLoader, valid_loader: DataLoader = None, callbacks=None):
        """
            Trains the neural net
        Args:
            optimizer (Optimizer): The optimizer function
            loss (function): The objective function.
            metrics (list, None): Metrics to be evaluated by the model
                        during training and testing.
                        Typically you will use `metrics=['accuracy']`.
            epochs (int): number of epochs
            train_loader (DataLoader): The Dataloader for training
            valid_loader (DataLoader, optional): The Dataloader for validation
            callbacks (list, None): List of callbacks functions to call at each epoch
        """
        if self.use_cuda:
            self.net.cuda()

        if not callbacks:
            callbacks = []
        callbacks.append(TQDM())
        train_loader_len = len(train_loader)
        if valid_loader:
            val_loader_len = len(valid_loader)
        else:
            val_loader_len = None

        callback_list = CallbackList(callbacks)
        callback_list.set_model(self.net)
        callback_list.on_train_begin({'total_epochs': epochs,
                                      'epoch_count': self.epoch_counter,
                                      'train_loader_len': train_loader_len,
                                      'val_loader_len': val_loader_len})

        for epoch in range(epochs):
            self._run_epoch(train_loader, valid_loader, optimizer, loss, metrics, callback_list)
            self.epoch_counter += 1

    def predict(self, test_loader, callbacks=None):
        """
            Launch the prediction on the given loader and pass
            each predictions to the given callbacks.
        Args:
            test_loader (DataLoader): The loader containing the test dataset
            callbacks (list): List of callbacks functions to call at prediction pass
        """
        # Switch to evaluation mode
        self.net.eval()

        it_count = len(test_loader)

        with tqdm(total=it_count, desc="Classifying") as pbar:
            for ind, (images, files_name) in enumerate(test_loader):
                if self.use_cuda:
                    images = images.cuda()

                images = Variable(images, volatile=True)

                # forward
                logits = self.net(images)
                probs = F.sigmoid(logits)
                probs = probs.data.cpu().numpy()

                # If there are callback call their __call__ method and pass in some arguments
                if callbacks:
                    for cb in callbacks:
                        cb(net=self.net,
                           probs=probs,
                           files_name=files_name
                           )

                pbar.update(1)
