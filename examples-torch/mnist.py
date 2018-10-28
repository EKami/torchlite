import torchlite
torchlite.set_backend("torch")

from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchlite.learner import Learner
from torchlite.torch.learner.cores import ClassifierCore
from torchlite.torch.metrics import CategoricalAccuracy
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.adaptive_max_pool2d(self.conv2_drop(self.conv2(x)), 4))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def main():
    batch_size = 128
    epochs = 20
    mnist_train_data = datasets.MNIST('/tmp/data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]))
    train_loader = DataLoader(mnist_train_data, batch_size, shuffle=True, num_workers=os.cpu_count())

    mnist_test_data = datasets.MNIST('/tmp/data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_loader = DataLoader(mnist_test_data, batch_size, shuffle=False, num_workers=os.cpu_count())

    net = Net()
    optimizer = optim.RMSprop(net.parameters(), lr=1e-3)
    loss = F.nll_loss

    learner = Learner(ClassifierCore(net, optimizer, loss))
    metrics = [CategoricalAccuracy()]

    learner.train(epochs, metrics, train_loader, test_loader, callbacks=None)

    # y_pred = learner.predict(test_loader)
    # y_true = test_loader.dataset.test_labels
    #
    # print("Test accuracy: {}%".format(CategoricalAccuracy()('test', y_pred, y_true)))


if __name__ == "__main__":
    main()
