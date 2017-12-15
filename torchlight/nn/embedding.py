import torch.nn as nn
import numpy as np


class Embeddings(nn.Module):
    def __init__(self, features):
        """
        Creates multiples embedding with a preconfigured length
        /!\ Embedding takes as input indexes, not onehot labels
        https://youtu.be/XJ_waZlJU8g?t=1h5m23s
        Args:
            features (dict): Categorical features with their values.
                Ex: {'temp': ['Warm', 'Cold', 'Cold'], 'state': ['Texas', 'New york', 'New york'], ...}
                Each feature should have the same number of values as all the others.
        """

        # To register multiple embeddings at once use nn.ModuleList: https://youtu.be/sHcLkfRrgoQ?t=55m5s
        emb = []
        for k, v in features:
            _, card = np.unique(v, return_counts=True)
            card += 1
            # TODO let the user define the maximum embedding size: https://youtu.be/5_xFdhfUnvQ?t=36m14s
            emb_sz = min(50, (card + 1) // 2)
            e = nn.Embedding(card, emb_sz)
            e.weight.data.uniform_(0, 0.05)
            emb.append(e)

    def forward(self):
        pass
