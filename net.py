import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d \
            (in_planes, in_planes, kernel_size=3, stride=stride,
             padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d \
            (in_planes, out_planes, kernel_size=1,
             stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2),
           512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=7, in_channels=1):
        super(MobileNet, self).__init__()
        self.name = 'MobileNetV1'
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.fn = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fn(out)
        return out


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=7, in_channels=1):
        super(MobileNetV2, self).__init__()
        self.name = 'MobileNetV2'
        self.stage = nn.Sequential(nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU6(inplace=True))
        self.model = models.MobileNetV2(num_classes)

    def forward(self, x):
        x = self.stage(x)
        y = self.model(x)
        return y


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=7, in_channels=1):
        super(MobileNetV3Small, self).__init__()
        self.name = 'MobileNetV3Small'
        self.stage = nn.Sequential(nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU6(inplace=True))
        self.model = nn.Sequential(*list(models.mobilenet_v3_small().children())[:-1])
        self.fn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        out = self.stage(x)
        out = self.model(out)
        out = self.fn(out)
        return out


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=7, in_channels=1):
        super(MobileNetV3Large, self).__init__()
        self.name = 'MobileNetV3Large'
        self.stage = nn.Sequential(nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU6(inplace=True))
        self.model = nn.Sequential(*list(models.mobilenet_v3_large().children())[:-1])
        self.fn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=960, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        out = self.stage(x)
        out = self.model(out)
        out = self.fn(out)
        return out
