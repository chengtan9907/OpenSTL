import math
import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_
from timm.models.convnext import ConvNeXtBlock
from timm.models.mlp_mixer import MixerBlock
from timm.models.swin_transformer import SwinTransformerBlock, window_partition, window_reverse
from timm.models.vision_transformer import Block as ViTBlock

from .layers import (HorBlock, ChannelAggregationFFN, MultiOrderGatedAggregation,
                     PoolFormerBlock, CBlock, SABlock, MixMlp, VANBlock)


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels % groups != 0:
            groups=1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class gInception_ST(nn.Module):
    """A IncepU block for SimVP"""

    def __init__(self, C_in, C_hid, C_out, incep_ker = [3,5,7,11], groups = 8):        
        super(gInception_ST, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)

        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(
                C_hid, C_out, kernel_size=ker, stride=1,
                padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class AttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2*dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        
        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x


class SpatialAttention(nn.Module):
    """A Spatial Attention block for SimVP"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        self.spatial_gating_unit = AttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class GASubBlock(nn.Module):
    """A GABlock (gSTA) for SimVP"""

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class ConvMixerSubBlock(nn.Module):
    """A block of ConvMixer."""

    def __init__(self, dim, kernel_size=9, activation=nn.GELU):
        super().__init__()
        # spatial mixing
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same")
        self.act_1 = activation()
        self.norm_1 = nn.BatchNorm2d(dim)
        # channel mixing
        self.conv_pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.act_2 = activation()
        self.norm_2 = nn.BatchNorm2d(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def forward(self, x):
        x = x + self.norm_1(self.act_1(self.conv_dw(x)))
        x = self.norm_2(self.act_2(self.conv_pw(x)))
        return x


class ConvNeXtSubBlock(ConvNeXtBlock):
    """A block of ConvNeXt."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim, mlp_ratio=mlp_ratio,
                         drop_path=drop_path, ls_init_value=1e-6, conv_mlp=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma'}

    def forward(self, x):
        x = x + self.drop_path(
            self.gamma.reshape(1, -1, 1, 1) * self.mlp(self.norm(self.conv_dw(x))))
        return x


class HorNetSubBlock(HorBlock):
    """A block of HorNet."""

    def __init__(self, dim, mlp_ratio=4., drop_path=0.1, init_value=1e-6):
        super().__init__(dim, mlp_ratio=mlp_ratio, drop_path=drop_path, init_value=init_value)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma1', 'gamma2'}

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


class MLPMixerSubBlock(MixerBlock):
    """A block of MLP-Mixer."""

    def __init__(self, dim, input_resolution=None, mlp_ratio=4., drop=0., drop_path=0.1):
        seq_len = input_resolution[0] * input_resolution[1]
        super().__init__(dim, seq_len=seq_len,
                         mlp_ratio=(0.5, mlp_ratio), drop_path=drop_path, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


class MogaSubBlock(nn.Module):
    """A block of MogaNet."""

    def __init__(self, embed_dims, mlp_ratio=4., drop_rate=0., drop_path_rate=0., init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4]):
        super(MogaSubBlock, self).__init__()
        self.out_channels = embed_dims
        # spatial attention
        self.norm1 = nn.BatchNorm2d(embed_dims)
        self.attn = MultiOrderGatedAggregation(
            embed_dims, attn_dw_dilation=attn_dw_dilation, attn_channel_split=attn_channel_split)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # channel MLP
        self.norm2 = nn.BatchNorm2d(embed_dims)
        mlp_hidden_dims = int(embed_dims * mlp_ratio)
        self.mlp = ChannelAggregationFFN(
            embed_dims=embed_dims, mlp_hidden_dims=mlp_hidden_dims, ffn_drop=drop_rate)
        # init layer scale
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2', 'sigma'}

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class PoolFormerSubBlock(PoolFormerBlock):
    """A block of PoolFormer."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim, pool_size=3, mlp_ratio=mlp_ratio, drop_path=drop_path,
                         drop=drop, init_value=1e-5)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SwinSubBlock(SwinTransformerBlock):
    """A block of Swin Transformer."""

    def __init__(self, dim, input_resolution=None, layer_i=0, mlp_ratio=4., drop=0., drop_path=0.1):
        window_size = 7 if input_resolution[0] % 7 == 0 else max(4, input_resolution[0] // 16)
        window_size = min(8, window_size)
        shift_size = 0 if (layer_i % 2 == 0) else window_size // 2
        super().__init__(dim, input_resolution, num_heads=8, window_size=window_size,
                         shift_size=shift_size, mlp_ratio=mlp_ratio,
                         drop_path=drop_path, drop=drop, qkv_bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


def UniformerSubBlock(embed_dims, mlp_ratio=4., drop=0., drop_path=0.,
                      init_value=1e-6, block_type='Conv'):
    """Build a block of Uniformer."""

    assert block_type in ['Conv', 'MHSA']
    if block_type == 'Conv':
        return CBlock(dim=embed_dims, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
    else:
        return SABlock(dim=embed_dims, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=True,
                       drop=drop, drop_path=drop_path, init_value=init_value)


class VANSubBlock(VANBlock):
    """A block of VAN."""

    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., init_value=1e-2, act_layer=nn.GELU):
        super().__init__(dim=dim, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path,
                         init_value=init_value, act_layer=act_layer)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class ViTSubBlock(ViTBlock):
    """A block of Vision Transformer."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim=dim, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=True,
                         drop=drop, drop_path=drop_path, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)
