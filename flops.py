from thop import profile, clever_format
from net import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small, Vgg19, ResNet50
import torch

if __name__ == '__main__':
    net = ResNet50()
    x = torch.randn(1, 1, 48, 48)
    flops, params = profile(net, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" % flops)
    print("params: %s" % params)
