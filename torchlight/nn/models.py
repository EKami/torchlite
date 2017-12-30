import torch
import torch.functional as F
import torch.nn as nn
from utils import tools


def emb_init(x):
    x = x.weight.data
    sc = 2 / (x.size(1) + 1)
    x.uniform_(-sc, sc)


class MixedInputModel(nn.Module):
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
            nn.init.kaiming_normal(o.weight.data)
        self.outp = nn.Linear(hidden_sizes[-1], output_sizes)
        nn.init.kaiming_normal(self.outp.weight.data)

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


class BasicModel:
    def __init__(self, model, name='unnamed'):
        self.model, self.name = model, name

    def get_layer_groups(self, do_fc=False):
        return tools.children(self.model)


class StructuredModel(BasicModel):
    def get_layer_groups(self, do_fc=False):
        m = self.model
        return [m.embs, tools.children(m.lins) + tools.children(m.bns), m.outp]
