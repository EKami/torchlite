import torch
import torch.nn as nn


class GroupNorm(nn.Module):
    def __init__(self, c_num, group_num=16, eps=1e-10):
        """
        The groupnorm layer from https://arxiv.org/abs/1803.08494
        Args:
            c_num (int): Number of input channels
            group_num (int): Number of group by which to divide the channels
            eps (float): Epsilon
        """
        super(GroupNorm, self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, self.group_num, -1)

        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)

        x = (x - mean) / (std + self.eps)
        x = x.view(batch_size, channels, height, width)

        return x * self.gamma + self.beta
