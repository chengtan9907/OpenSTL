# refer to the code from MogaNet, Thanks!
# https://github.com/Westlake-AI/MogaNet/blob/main/models/moganet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation in MogaNet."""

    def __init__(self, embed_dims, mlp_hidden_dims, kernel_size=3, act_layer=nn.GELU, ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()
        self.embed_dims = embed_dims
        self.mlp_hidden_dims = mlp_hidden_dims

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=self.mlp_hidden_dims, kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.mlp_hidden_dims, out_channels=self.mlp_hidden_dims, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=True, groups=self.mlp_hidden_dims)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(
            in_channels=mlp_hidden_dims, out_channels=embed_dims, kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.mlp_hidden_dims, out_channels=1, kernel_size=1)
        self.sigma = nn.Parameter(
            1e-5 * torch.ones((1, mlp_hidden_dims, 1, 1)), requires_grad=True)
        self.decompose_act = act_layer()

    def feat_decompose(self, x):
        x = x + self.sigma * (x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel in MogaNet."""

    def __init__(self, embed_dims, dw_dilation=[1, 2, 3], channel_split=[1, 3, 4]):
        super(MultiOrderDWConv, self).__init__()
        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims, stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1, out_channels=self.embed_dims_1, kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1, stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2, out_channels=self.embed_dims_2, kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2, stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation in MogaNet."""

    def __init__(self, embed_dims, attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4], attn_shortcut=True):
        super(MultiOrderGatedAggregation, self).__init__()
        self.embed_dims = embed_dims
        self.attn_shortcut = attn_shortcut
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims, dw_dilation=attn_dw_dilation, channel_split=attn_channel_split)
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()
        # decompose
        self.sigma = nn.Parameter(1e-5 * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma * (x - x_d)
        x = self.act_value(x)
        return x

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        if self.attn_shortcut:
            x = x + shortcut
        return x
