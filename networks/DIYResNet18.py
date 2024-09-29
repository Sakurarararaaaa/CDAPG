import torchvision.models as models
import torch
from torch import nn

class DIYResNet18(torch.nn.Module):
    def __init__(self, model, block):
        super(DIYResNet18, self).__init__()
        print("block:", block)
        if model == 'resnet18':
            if block == 'BasicBlock':
                resnet = models.resnet18(weights=None)
            else:
                resnet = models.resnet18()#weights=None

        elif model == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, mlp_hidden_size=512, projection_size=128)


    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)

class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)