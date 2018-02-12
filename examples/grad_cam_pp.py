"""
This script implements the grad-cam++ method to look inside
a CNN and see a heatmap of where the network is looking to do
its predictions. Link to the paper: https://arxiv.org/abs/1710.11063

The presented code uses a pretrained VGG model on a cat/dog classification
task as a demo of using Grad-cam++ in that context.

Resources:
 - https://github.com/fastai/fastai/blob/master/courses/dl1/lesson7-CAM.ipynb
 - https://github.com/adityac94/Grad_CAM_plus_plus
 - http://www.gitxiv.com/posts/JradqzFyKi4pHebB4/grad-cam-generalized-gradient-based-visual-explanations-for
"""
import torchvision.transforms as transforms
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
import torchlight.data.fetcher as fetcher
from torchlight.shortcuts import ImageClassifierShortcut
from torchlight.nn.learners.learner import Learner
from torchlight.nn.learners.cores import ClassifierCore
from torchlight.nn.metrics import CategoricalAccuracy
from torchlight.nn.test_callbacks import ActivationMapVisualizerCallback
import matplotlib.pyplot as plt
import numpy as np
import torchlight.nn.tools as tools


def show_test_image(test_image_name, shortcut, y_mapping, y_pred):
    image_mat, _, image_index = shortcut.get_test_loader.dataset.get_by_name(test_image_name)
    image_mat = tools.to_np(image_mat)
    plt.imshow(image_mat)
    plt.title("Predicted: " + str(y_mapping[np.argmax(y_pred[image_index])]))
    plt.show()


def main():
    batch_size = 512
    epochs = 3
    root_dir = "/tmp"
    fetcher.WebFetcher.download_dataset("https://s3-eu-west-1.amazonaws.com/torchlight-datasets/dogscats.zip",
                                        root_dir, True)
    root_dir = "/tmp/dogscats"
    root_dir = Path(root_dir)
    train_folder = root_dir / "train"
    val_folder = root_dir / "valid"
    test_folder = root_dir / "test"
    test_image_name = "12500.jpg"

    # Image augmentation/transformations
    transformations = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),
                                          ])
    shortcut = ImageClassifierShortcut.from_paths(train_folder=train_folder.absolute(),
                                                  val_folder=val_folder.absolute(),
                                                  test_folder=test_folder.absolute(),
                                                  transforms=transformations,
                                                  batch_size=batch_size)

    net = shortcut.get_resnet_model()
    # Don't optimize the frozen layers parameters of resnet
    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)
    loss = F.nll_loss
    metrics = [CategoricalAccuracy()]
    grad_cam_callback = ActivationMapVisualizerCallback(test_image_name)  # TODO finish grad_cam++ here

    learner = Learner(ClassifierCore(net, optimizer, loss))
    learner.train(epochs, metrics, shortcut.get_train_loader, None)

    y_mapping = shortcut.get_y_mapping
    y_pred = learner.predict(shortcut.get_test_loader, callbacks=[grad_cam_callback])
    heatmap = grad_cam_callback.get_heatmap
    show_test_image(test_image_name, shortcut, y_mapping, y_pred)


if __name__ == "__main__":
    main()
