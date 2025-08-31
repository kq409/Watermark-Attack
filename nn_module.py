import torch
from torch import nn, cat
from torch.nn import Conv2d, BatchNorm2d, ReLU, ConvTranspose2d

#
# class CA(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
#         self.bn16 = BatchNorm2d(num_features=16)
#
#         self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
#         self.bn32 = BatchNorm2d(num_features=32)
#
#         self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
#         self.bn64 = BatchNorm2d(num_features=64)
#
#         self.conv4 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
#         self.bn128 = BatchNorm2d(num_features=128)
#
#         self.conv5 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
#         self.bn256 = BatchNorm2d(num_features=256)
#
#         self.conv6 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2)
#
#         self.conv7 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)
#         self.bn512 = BatchNorm2d(num_features=512)
#
#         self.conv8 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2)
#
#         self.deconv9 = ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=5, padding=2, stride=1)
#
#         self.concate1 = cat()
#
#         self.deconv10 =ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=5, padding=2, stride=1)

class ConvBNReLU(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels= in_channels, out_channels=out_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=5),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return x

class Network(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Network, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down1 = ConvBNReLU(n_channels, 16)
        self.down2 = ConvBNReLU(16, 32)
        self.down3 = ConvBNReLU(32, 64)
        self.down4 = ConvBNReLU(64, 128)
        self.down5 = ConvBNReLU(128, 256)
        self.down6 = ConvBNReLU(256, 256)
        self.down7 = ConvBNReLU(256, 512)
        self.down8 = ConvBNReLU(512, 512)

        self.up1 = Up(512, 512)
        self.up2 = Up(1024, 256)
        self.up3 = Up(512, 256)
        self.up4 = Up(512, 128)
        self.up5 = Up(256, 64)
        self.up6 = Up(128, 32)
        self.up7 = Up(64, 16)

        self.deconv = nn.ConvTranspose2d(32, 3, kernel_size=5)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)

        x = self.up1(x8, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)

        x = self.deconv(x)
        x = self.sigmoid(x)
        x = self.relu(x)
        return x

network = Network(3, 1)
input = torch.ones((1, 3, 256, 256))
output = network(input)
print(output)
# print(network)
