import torch
from torch import nn


class ConvBNReLU(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x1 = self.Conv(x)
        # print(x.size())
        return x1

class UpDrop(nn.Module):

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(inplace=False)
        )

    def forward(self, x1, x2):
        x3 = self.up(x1)
        x = torch.cat([x2, x3], dim=1)
        # print(x.shape)
        return x

class Up(nn.Module):

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x1, x2):
        x3 = self.up(x1)
        x = torch.cat([x2, x3], dim=1)
        # print(x.shape)
        return x

class UpTanh(nn.Module):

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.up(x)
        # print(x.shape)
        return x1

class model_UD(nn.Module):
    def __init__(self, n_channels):
        super(model_UD, self).__init__()
        self.n_channels = n_channels

        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)

        self.down1 = ConvBNReLU(64, 128)
        self.down2 = ConvBNReLU(128, 256)
        self.down3 = ConvBNReLU(256, 512)
        self.down4 = ConvBNReLU(512, 512)
        self.down5 = ConvBNReLU(512, 512)
        self.down6 = ConvBNReLU(512, 512)
        self.down7 = ConvBNReLU(512, 512)

        self.up1 = UpDrop(512, 512)
        self.up2 = UpDrop(1024, 512)
        self.up3 = UpDrop(1024, 512)
        self.up4 = Up(1024, 512)
        self.up5 = Up(1024, 256)
        self.up6 = Up(512, 128)
        self.up7 = Up(256, 64)

        self.up8 = UpTanh(128, 3)


    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)


        # print("UP")
        xout1 = self.up1(x8, x7)
        xout2 = self.up2(xout1, x6)
        xout3 = self.up3(xout2, x5)
        xout4 = self.up4(xout3, x4)
        xout5 = self.up5(xout4, x3)
        xout6 = self.up6(xout5, x2)
        xout7 = self.up7(xout6, x1)

        x = self.up8(xout7)

        return x

# network = model_UD(3)
# input_t = torch.ones((1, 3, 512, 512))
# output = network(input_t)
# print(output)
# print(output.shape)