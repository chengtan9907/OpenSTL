import torch
import torch.nn as nn
import torch.nn.functional as F


backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(
            tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(
            tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(tenFlow.device)
        # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border',
                                           align_corners=True)

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.bernoulli(x)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PReLU(out_planes)
    )
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class Routing(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Routing, self).__init__()
        self.routing = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.routing(x)


class MVFB(nn.Module):
    def __init__(self, in_planes, num_feature):
        super(MVFB, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, num_feature//2, 3, 2, 1),
            conv(num_feature//2, num_feature, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
        )
        self.conv_sq = conv(num_feature, num_feature//4)

        self.conv1 = nn.Sequential(
            conv(in_planes, 8, 3, 2, 1),
        )
        self.convblock1 = nn.Sequential(
            conv(8, 8),
        )
        self.lastconv = nn.ConvTranspose2d(num_feature//4 + 8, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        x0 = x
        flow0 = flow
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
        x = torch.cat((x, flow), 1)
        x1 = self.conv0(x)
        x2 = self.conv_sq(self.convblock(x1) + x1)
        x2 = F.interpolate(x2, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        x3 = self.conv1(torch.cat((x0,flow0), 1))
        x4 = self.convblock1(x3)
        tmp = self.lastconv(torch.cat((x2, x4), dim=1))
        flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        return flow, mask
