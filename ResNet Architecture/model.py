import torch.nn as nn
from torch.nn import Module

class ResNet(Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.convlayer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=1, stride=2)
        self.batchnorm = nn.BatchNorm2d(64)
        self.Relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.network_body  = nn.Sequential(
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            nn.AvgPool2d(kernel_size=(10, 10)),
            nn.Flatten(),
            nn.Linear(512, 2),
            nn.Sigmoid()      
        )

    def forward(self, input):
        output = self.convlayer(input)
        output = self.batchnorm(output)
        output = self.Relu(output)
        output = self.pool(output)
        output = self.network_body(output)
        return output


class ResBlock(nn.Module):
    def __init__(self, in_ch, o_ch, stride):
        super(ResBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, o_ch, kernel_size = 3, padding = 1, stride=stride),
            nn.BatchNorm2d(o_ch),
            nn.ReLU()  
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(o_ch, o_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(o_ch)
        )

        self.Relu = nn.ReLU()
        self.down_sampling = nn.Sequential(
            nn.Conv2d(in_ch, o_ch, kernel_size=1, stride=stride),
            nn.BatchNorm2d(o_ch)
        )

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        x = self.down_sampling(input)
        output += x
        output = self.Relu(output)

        return output