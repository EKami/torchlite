import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, num_embedding, vars):
        """
        Creates multiples embedding with a preconfigured length
        :param num_embedding:
        """
        pass

    def forward(self):
        # Calculate the cardinality of each categorical feature (Rossman notebook)
        cat_sz = [(c, len(joined_samp[c].cat.categories) + 1) for c in cat_vars]
        # Calculate the embedding dimension
        emb_szs = [(c, min(50, (c + 1) // 2)) for _, c in cat_sz]
        # https://youtu.be/J99NV9Cr75I?t=48m52s
        e = nn.Embedding(, emb_szs)
        e.weight.data.uniform_(-0.01, 0.01)