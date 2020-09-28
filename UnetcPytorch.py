import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import WarpSTPytorch as WarpST

def conv_bn_relu(in_channels, out_channels, kernel_size=3, strides=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def downsampling():
    return nn.MaxPool2d(2)

def upsampling(in_channels, out_channels, kernel_size=2, strides=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace= True)
    )

class Net (nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #DownSampling
        self.conv1 = conv_bn_relu(2, 64)
        self.conv2 = conv_bn_relu(64, 128)
        self.conv3 = conv_bn_relu(128, 256)
        self.down = downsampling()

        #UpSampling
        self.up_4 = upsampling(256, 128)
        self.conv4 =conv_bn_relu(256, 128)

        self.up_5 = upsampling(128, 64)
        self.conv5 = conv_bn_relu(128, 64)

        self.conv6 = nn.Conv2d(64, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a = 0, mode= 'fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)

        p1 = self.down(x1)
        x2 = self.conv2(p1)

        p2 = self.down(x2)
        x3 = self.conv3(p2)

        p4 = self.up_4(x3)
        x4 = torch.cat([p4, x2], dim=1)
        x4 = self.conv4(x4)

        p5 = self.up_5(x4)
        x5 = torch.cat([p5, x1], dim=1)
        x5 = self.conv5(p5)

        x6 = self.conv6(x5)

        return x6

class STN(object):
  def __init__(self, config, is_train):
    self.config = config
    self.is_train = is_train
    self.net = Net()
    print(self.net)
    self.optimizer = optim.SGD(self.net.parameters(), self.config.lr)

  def fit(self, batch_x, batch_y):
    batch_x = torch.from_numpy(batch_x)
    batch_y = torch.from_numpy(batch_y)

    xy = torch.cat([batch_x, batch_y], 3)
    xy = xy.permute(0, 3, 1, 2)

    if self.is_train:

        self.optimizer.zero_grad()

        # vector map & moved image
        v = self.net(xy)
        z = WarpST(batch_x, v, self.config.im_size)
        z = z.permute(0, 2, 3, 1)
        loss = ncc(batch_y, z)
        loss.backward()
        self.optimizer.step()

        return z, loss.item()