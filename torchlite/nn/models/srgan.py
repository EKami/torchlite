import torch
from torchlite.nn.models.models import Flatten
import torch.nn as nn
import torch.nn.functional as F
import math


class Generator(nn.Module):
    def __init__(self, scale_factor, res_blocks_count=16):
        """
        Generator for SRGAN
        Args:
            scale_factor (int): The new scale for the resulting image (x2, x4...)
            res_blocks_count (int): Number of residual blocks, the less there is,
            the faster the inference time will be but the network will capture
            less information.
        """
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU()
        )
        self.res_blocks = []
        for i in range(res_blocks_count):
            self.res_blocks.append(ResidualBlock(64))

        self.res_blocks = nn.Sequential(*self.res_blocks)
        self.block_x1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.block_x2 = nn.Sequential(*[UpsampleBLock(64, 2) for _ in range(upsample_block_num)])
        self.block_x3 = nn.Conv2d(64, 3, kernel_size=1, padding=0)

    def forward(self, x):
        block1 = self.block1(x)
        res_blocks = self.res_blocks(block1)
        block_x1 = self.block_x1(res_blocks)
        block_x2 = self.block_x2(block1 + block_x1)  # ElementWise sum

        # TODO causes a memory leak on CPU
        block_x3 = self.block_x3(block_x2)

        return F.tanh(block_x3)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual  # ElementWise sum


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1024, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            nn.Conv2d(1024, 2048, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),

            nn.Conv2d(2048, 1024, kernel_size=(1, 1), stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=1, padding=1),
            nn.BatchNorm2d(512),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=(1, 1), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
        )

        in_size = self.infer_lin_size(input_shape)

        self.out_block = nn.Sequential(
            Flatten(),
            nn.Linear(in_size, 1),
            nn.Sigmoid(),
        )

    def infer_lin_size(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        size = self.block1(input).data.view(bs, -1).size(1)
        return size

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = F.leaky_relu(block1 + block2, 0.2)
        out = self.out_block(block3)
        return out
