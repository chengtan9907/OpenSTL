# refer to the code from VAN, Thanks!
# https://github.com/Visual-Attention-Network/VAN-Classification

import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  # 1x1
        self.dwconv = DWConv(hidden_features)                  # CFF: Convlutional feed-forward network
        self.act = act_layer()                                 # GELU
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1) # 1x1
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):      # Large Kernel Attention
    def __init__(self, dim, kernel_size, dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2*dim, 1)

        reduction = 16
        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
            nn.Sigmoid()
        )

        # GATE
        self.conv2_0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv2_spatial = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv2_1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        
        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x


class SpatialAttention(nn.Module):
    def __init__(self, d_model, kernel_size=21):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        self.spatial_gating_unit = AttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class GASubBlock(nn.Module):
    def __init__(self, dim, kernel_size=21, mlp_ratio=4., drop=0., drop_path=0.1, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x