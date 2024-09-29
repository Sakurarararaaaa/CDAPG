from networks import DIYResNet18
from networks import CD_MyResNet18
import torch.nn as nn
import argparse


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        print("planes:---------------", planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def ResNet18_Port(dataset, mode, block, num_classes, **kwargs):
    # resnet18_model = models.resnet18(weights=None)
    resnet18_model = DIYResNet18('resnet18', block=block)
    conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
    resnet18_model.encoder[0] = conv1
    resnet18_model.encoder[8] = nn.Sequential()
    final_model = CD_MyResNet18(BasicBlock, resnet18_model, num_classes, **kwargs)
    print("final_model\n", final_model)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--TrainChoise', default='MLP_MyResNet18_BYOL', type=str,
                        help='ResNet18 | ResNet18_ResNet18 | ResNet18_ResNet18_IRMLP| MyResNet18 | MLP_MyResNet18_IRMLP | '
                             'MLP_MLP | ResNet18_MLP | MyResNet18_BYOL | MLP_MyResNet18_BYOL | ResNet18_MyResNet18_IRMLP'
                             'MLP_MyResNet18 | MLP_ResNet18 | ResNet18_MyResNet18')
    parser.add_argument('--BlockChoise', default='BasicBlock', type=str,
                        help='BasicBlock | SEBasicBlock | BasicBlock_LeakyReLU | '
                             'SEBasicBlock_LeakyReLU | BasicBlock_GELU')
    args = parser.parse_args()

    model = ResNet18_Port(dataset=args.dataset, mode=args.TrainChoise, block=args.BlockChoise,num_classes=args.num_classes)
