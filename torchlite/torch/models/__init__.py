import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlite.torch.tools import tensor_tools


class FinetunedConvModel(nn.Module):

    def __init__(self, base_model_head, output_layer):
        """
        A convolutional neural net model used for categorical classification
        Args:
            base_model_head (list): The list of pretrained layers which will
                be added on top of this model like Resnet or Vgg.
            output_layer (nn.Module): The output layer (usually softmax or sigmoid)
            E.g:
                resnet = torchvision.models.resnet34(pretrained=True)
                # Take the head of resnet up until AdaptiveAvgPool2d
                resnet_head = tools.children(resnet)[:-2]
                net = FinetunedConvModel(resnet_head)
        """
        super().__init__()
        self.base_model_head = nn.Sequential(*FinetunedModelTools.freeze(base_model_head))

        # Fine tuning
        self.conv1 = nn.Conv2d(512, 2, 3, padding=1)
        self.adp1 = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()
        self.out = output_layer

    def forward(self, input):
        x = self.base_model_head(input)
        x = self.conv1(x)
        x = self.adp1(x)
        x = self.flatten(x)
        x = self.out(x)
        return x


class TabularModel(nn.Module):
    def __init__(self, embedding_sizes, n_continuous, emb_drop, output_sizes, hidden_sizes,
                 hidden_dropouts, y_range=None, use_bn=False):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c, s in embedding_sizes])
        for emb in self.embs:
            emb_init(emb)
        n_emb = sum(em.embedding_dim for em in self.embs)

        hidden_sizes = [n_emb + n_continuous] + hidden_sizes
        self.linears = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
                                      for i in range(len(hidden_sizes) - 1)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(sz) for sz in hidden_sizes[1:]])

        for o in self.linears:
            nn.init.kaiming_normal_(o.weight.data)
        self.outp = nn.Linear(hidden_sizes[-1], output_sizes)
        nn.init.kaiming_normal_(self.outp.weight.data)

        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in hidden_dropouts])
        self.bn = nn.BatchNorm1d(n_continuous)
        self.use_bn, self.y_range = use_bn, y_range

    def forward(self, x_cat, x_cont):
        x = [em(x_cat[:, i]) for i, em in enumerate(self.embs)]
        x = torch.cat(x, 1)
        x2 = self.bn(x_cont)
        x = self.emb_drop(x)
        x = torch.cat([x, x2], 1)
        for l, d, b in zip(self.linears, self.drops, self.batch_norms):
            x = F.relu(l(x))
            if self.use_bn:
                x = b(x)
            x = d(x)
        x = self.outp(x)
        if self.y_range:
            x = F.sigmoid(x)
            x = x * (self.y_range[1] - self.y_range[0])
            x = x + self.y_range[0]
        return x


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class FinetunedModelTools:
    @staticmethod
    def _get_layer_groups(layers):
        return tensor_tools.children(layers)

    @staticmethod
    def _set_trainable_attr(m, b):
        m.trainable = b
        for p in m.parameters():
            p.requires_grad = b

    @staticmethod
    def _apply_leaf(layer, func):
        c = tensor_tools.children(layer)
        if isinstance(layer, nn.Module):
            func(layer)
        if len(c) > 0:
            for l in c:
                FinetunedModelTools._apply_leaf(l, func)

    @staticmethod
    def set_trainable(leaf, trainable):
        FinetunedModelTools._apply_leaf(leaf, lambda m: FinetunedModelTools._set_trainable_attr(m, trainable))

    @staticmethod
    def freeze_to(layers, index):
        """
        Freeze all but the layers up until index.
        Make all layers untrainable (i.e. frozen) up to the index layer.

        Args:
            layers (list): The layers to freeze
            index (int): The index on which to freeze up to

        Returns:

        """
        c = FinetunedModelTools._get_layer_groups(layers)
        for l in c:
            FinetunedModelTools.set_trainable(l, False)
        for l in c[index:]:
            FinetunedModelTools.set_trainable(l, True)
        return layers

    @staticmethod
    def freeze(layers):
        """
        Freeze all but the very last layer.
        Make all layers untrainable (i.e. frozen) except for the last layer.

        Args:
            layers (list): The layers to freeze

        Returns:
            model: The passed model
        """
        return FinetunedModelTools.freeze_to(layers, -1)


def emb_init(x):
    x = x.weight.data
    sc = 2 / (x.size(1) + 1)
    x.uniform_(-sc, sc)
