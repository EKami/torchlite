import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchlite.torch.models import Flatten


class Generator(nn.Module):
    def __init__(self, scale_factor, res_blocks_count=4):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = [ResidualBlock(64) for _ in range(res_blocks_count)]
        self.res_blocks = nn.Sequential(*self.res_blocks)

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )

        up_blocks = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        self.up_blocks = nn.Sequential(*up_blocks)
        self.block4 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        block_x1 = self.block1(x)
        res_blocks = self.res_blocks(block_x1)
        block_x3 = self.block3(res_blocks)
        up_blocks = self.up_blocks(block_x1 + block_x3)
        block_x4 = self.block4(up_blocks)

        return (F.tanh(block_x4) + 1) / 2


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.in1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.in2(residual)

        return x + residual


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
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.block7 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=(1, 1), stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.block8 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=1, padding=1)

        self.block9 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=(1, 1), stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.block10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )

        self.block11 = nn.Conv2d(128, 512, kernel_size=(3, 3), stride=1, padding=1)

        in_size = self.infer_lin_size(input_shape)

        self.out_block = nn.Sequential(
            Flatten(),
            nn.Linear(in_size, 1),
            nn.Sigmoid(),
        )

    def infer_lin_size(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        model = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            self.block6,
            self.block7,
            self.block8,
            self.block9,
            self.block10,
            self.block11,
        )
        size = model(input).data.view(bs, -1).size(1)
        return size

    def forward(self, x):
        feature_maps = []

        x = self.block1(x)
        feature_maps.append(x)

        x = self.block2(x)
        feature_maps.append(x)

        x = self.block3(x)
        feature_maps.append(x)

        x = self.block4(x)
        feature_maps.append(x)

        x = self.block5(x)
        feature_maps.append(x)

        x = self.block6(x)
        feature_maps.append(x)

        x = self.block7(x)
        feature_maps.append(x)

        block8 = self.block8(x)
        # feature_maps.append(block8)

        x = self.block9(block8)
        feature_maps.append(x)

        x = self.block10(x)
        feature_maps.append(x)

        block11 = self.block11(x)
        # feature_maps.append(block11)

        final_block = F.leaky_relu(block8 + block11, 0.2)
        feature_maps.append(final_block)

        out = self.out_block(final_block)
        return out, feature_maps


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
