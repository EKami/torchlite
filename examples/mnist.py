from torchvision import datasets, transforms
from torchlight.nn.learner import Learner
from torchlight.nn.metrics import CategoricalAccuracy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


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
    train_loader = DataLoader(mnist_train_data, batch_size, shuffle=True)

    mnist_test_data = datasets.MNIST('/tmp/data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_loader = DataLoader(mnist_test_data, batch_size, shuffle=False)

    net = Net()
    learner = Learner(net)

    optimizer = optim.RMSprop(net.parameters(), lr=1e-3)
    loss = F.nll_loss
    metrics = [CategoricalAccuracy()]

    learner.train(optimizer, loss, metrics, epochs, train_loader, None, callbacks=None)

    y_pred = learner.predict(test_loader)
    y_true = test_loader.dataset.test_labels

    print(f"Test accuracy: {CategoricalAccuracy()(y_true, y_pred)}%")


if __name__ == "__main__":
    main()
