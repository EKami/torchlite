import torch
import torch.functional as F
import torch.nn as nn
from utils import tools


def emb_init(x):
    x = x.weight.data
    sc = 2 / (x.size(1) + 1)
    x.uniform_(-sc, sc)


class MixedInputModel(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops,
                 y_range=None, use_bn=False):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_szs])
        for emb in self.embs:
            emb_init(emb)
        n_emb = sum(e.embedding_dim for e in self.embs)

        szs = [n_emb + n_cont] + szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i + 1]) for i in range(len(szs) - 1)])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]])
        for o in self.lins:
            nn.init.kaiming_normal(o.weight.data)
        self.outp = nn.Linear(szs[-1], out_sz)
        nn.init.kaiming_normal(self.outp.weight.data)

        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn, self.y_range = use_bn, y_range

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        x = torch.cat(x, 1)
        x2 = self.bn(x_cont)
        x = self.emb_drop(x)
        x = torch.cat([x, x2], 1)
        for l, d, b in zip(self.lins, self.drops, self.bns):
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
