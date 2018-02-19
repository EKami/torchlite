from torchlight.nn.models.models import Flatten
import torch.nn as nn
import torch.nn.functional as F
import math


class Generator(nn.Module):
    def __init__(self, scale_factor, res_blocks_count=16):
        """
        Generator for SRPGAN
        Args:
            scale_factor (int): The new scale for the resulting image (x2, x4...)
            res_blocks_count (int): Number of residual blocks, the less there is,
            the faster the inference time will be but the network will capture
            less information.
        """
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=2, padding=4),
            nn.LeakyReLU()
        )
        self.res_blocks = []
        for i in range(res_blocks_count):
            self.res_blocks.append(ResidualBlock(64))

        self.res_blocks = nn.Sequential(*self.res_blocks)
        self.block_x1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(64)
        )
        self.block_x2 = nn.Sequential(*[UpsampleBLock(64, 2) for _ in range(upsample_block_num)])
        self.block_x3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        block1 = self.block1(x)
        res_blocks = self.res_blocks(block1)
        block_x1 = self.block_x1(res_blocks)

        # Upsample
        block_x2 = self.block_x2(block1 + block_x1)  # ElementWise sum
        block_x3 = self.block_x3(block_x2)

        return (F.tanh(block_x3) + 1) / 2


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.lrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
        self.in2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.in1(residual)
        residual = self.lrelu(residual)
        residual = self.conv2(residual)
        residual = self.in2(residual)

        return x + residual  # ElementWise sum


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.lrelu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            Flatten(),
        )

    def forward(self, x):
        x = self.net(x)
        x = nn.Linear(x.size(), -1)()
        x = F.sigmoid(x)
        return x
