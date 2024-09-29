import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor = 4, dropout = 0.):
    return nn.Sequential(
        nn.Linear(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(image_size, channels, patch_size, dim, depth, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, nn.Sequential(
                Rearrange('b n c -> b c n'),
                FeedForward(num_patches, expansion_factor, dropout),
                Rearrange('b c n -> b n c'),
            )),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Rearrange('b n c -> b c n'),
        nn.AdaptiveAvgPool1d(1),
        Rearrange('b c () -> b c'),
        # nn.Linear(dim, num_classes)
    )

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out

class CD_MyResNet18(nn.Module):
    def __init__(self, block, resnet18_model, num_classes, **kwargs):
        super(CD_MyResNet18, self).__init__()
        # self.relu = nn.ReLU(inplace=False)
        # print("fix-resnet18_model\n", resnet18_model)
        self.relu = nn.ReLU(inplace=False)
        self.backbone = nn.Sequential(*list(resnet18_model.children())[:-1])
        self.att_conv1 = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=3, padding=1, bias=False)
        self.att_bn1 = nn.BatchNorm2d(512 * block.expansion)
        self.att_conv2 = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=3, padding=1, bias=False)
        self.att_bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.att_conv3 = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=3, padding=1, bias=False)
        self.att_bn3 = nn.BatchNorm2d(512 * block.expansion)
        self.att_conv4 = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=3, padding=1, bias=False)
        self.att_bn4 = nn.BatchNorm2d(512 * block.expansion)
        # in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        self.MLPMixer = MLPMixer(in_channels=1, image_size=256, patch_size=16, num_classes=3,
                                 dim=512, depth=1, token_dim=256, channel_dim=2048)
        self.norm2 = LayerNorm(512 * 2, eps=1e-6, data_format="channels_first")
        # self.conv_norm = nn.LayerNorm(512*2, eps=1e-6)   # final norm layer
        # self.conv_norm = nn.BatchNorm2d(512 * 2)
        self.residual = IRMLP(512 + 512, 512)

        self.avgpool = nn.AvgPool2d(14, stride=1)#16
        # self.weight = nn.Parameter(torch.ones(2))
        self.fc = nn.Linear(512, num_classes)
        # self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, x):#, atts
        global final_score_saliency_map
        input = x
        ax = self.backbone(x)
        ex = ax
        input_gray = torch.mean(input, dim=1, keepdim=True)
        input_resized = F.interpolate(input_gray, (14, 14), mode='bilinear')

        fe = ax.clone()
        org = fe.clone()
        a1, a2, a3, a4 = fe.size()
        fe = fe.view(a1, a2, -1) # fe.shape torch.Size([64, 512, 196])

        fe -= fe.min(2, keepdim=True)[0]
        fe /= fe.max(2, keepdim=True)[0]
        fe = fe.view(a1, a2, a3, a4)

        fe[torch.isnan(fe)] = 1
        fe[(org == 0)] = 0 # fe.shape torch.Size([64, 512, 16, 16])

        new_fe = fe * input_resized # new_fe.shape torch.Size([64, 512, 16, 16])

        # FIN
        ax = self.att_conv1(new_fe)
        ax = self.att_bn1(ax)
        ax = self.relu(ax)
        ax = self.att_conv2(ax)
        ax = self.att_bn2(ax)
        ax = self.relu(ax)
        ax = self.att_conv3(ax)
        ax = self.att_bn3(ax)
        ax = self.relu(ax)
        ax = self.att_conv4(ax)
        ax = self.att_bn4(ax)
        ax = self.relu(ax)

        ax = self.avgpool(ax) # ax.shape torch.Size([64, 512, 1, 1])

        w = F.softmax(ax.view(ax.size(0), -1), dim=1) # w.shape torch.Size([64, 512])
        b, c, u, v = fe.size() # c=512
        score_saliency_map = torch.zeros((b, 1, u, v)).cuda() # torch.Size([64, 1, 16, 16])

        for i in range(c):
            saliency_map = torch.unsqueeze(ex[:, i, :, :], 1) # saliency_map.shape torch.Size([64, 1, 16, 16])

            score = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(w[:, i], 1), 1), 1)
            # score.shape torch.Size([64, 1, 1, 1])

            final_score_saliency_map = score_saliency_map + (score * saliency_map) # torch.Size([64, 1, 16, 16])

        final_score_saliency_map = F.relu(final_score_saliency_map) # torch.Size([64, 1, 16, 16])
        # final_score_saliency_map = F.leaky_relu(final_score_saliency_map) # torch.Size([64, 1, 16, 16])

        org = final_score_saliency_map.clone()
        a1, a2, a3, a4 = final_score_saliency_map.size()
        final_score_saliency_map = final_score_saliency_map.view(a1, a2, -1)
        # final_score_saliency_map.shape torch.Size([64, 1, 196])

        final_score_saliency_map = final_score_saliency_map - (final_score_saliency_map.min(2, keepdim=True)[0])
        final_score_saliency_map = final_score_saliency_map / (final_score_saliency_map.max(2, keepdim=True)[0])
        final_score_saliency_map = (final_score_saliency_map.view(a1, a2, a3, a4))

        final_score_saliency_map[torch.isnan(final_score_saliency_map)] = org[torch.isnan(final_score_saliency_map)]

        att = final_score_saliency_map # att.shape torch.Size([64, 1, 16, 16])

        # attention mechanism
        rx = att * ex # rx.shape torch.Size([64, 512, 16, 16])
        rx = rx + ex # torch.Size([64, 512, 16, 16])

        # classifier
        rx = self.avgpool(rx) # torch.Size([64, 512, 1, 1])
        rx = rx.view(rx.size(0), -1) # torch.Size([64, 512])
        out = self.fc(rx)

        return out, att