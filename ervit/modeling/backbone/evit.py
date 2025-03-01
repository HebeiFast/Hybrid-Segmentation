# Copyright (c) ByteDance Inc. All rights reserved.
from functools import partial

import torch
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from .utils import merge_pre_bn
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import surrogate

NORM_EPS = 1e-5


class ConvBNReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = neuron.LIFNode(tau=2.,
                                  decay_input=True,
                                  v_threshold=1.,
                                  v_reset=0.,
                                  surrogate_function=surrogate.Sigmoid(),
                                  detach_reset=True,
                                  step_mode='m',
                                  backend='cupy')

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.flatten(0, 1)
        x = self.conv(x)
        x = self.norm(x)
        x = x.reshape(T, B, x.shape[1], x.shape[2], x.shape[3]).contiguous()
        x = self.act(x)
        x = x.permute(1, 0, 2, 3, 4)
        return x



def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.norm(self.conv(self.avgpool(x)))
        return x


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """
    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = neuron.LIFNode(tau=2.,
                                  decay_input=True,
                                  v_threshold=1.,
                                  v_reset=0.,
                                  surrogate_function=surrogate.Sigmoid(),
                                  detach_reset=True,
                                  step_mode='m',
                                  backend='cupy')

        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x, T, B):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = out.reshape(T, B, out.shape[1], out.shape[2], out.shape[3]).contiguous()
        out = self.act(out)
        out = out.flatten(0, 1)
        out = self.projection(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        # hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        hidden_dim = _make_divisible(in_features * mlp_ratio, 16)
        # hidden_dim = _make_divisible(in_features * mlp_ratio, 8)
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = neuron.LIFNode(tau=2.,
                                  decay_input=True,
                                  v_threshold=1.,
                                  v_reset=0.,
                                  surrogate_function=surrogate.Sigmoid(),
                                  detach_reset=True,
                                  step_mode='m',
                                  backend='cupy')
        self.conv2 = nn.Conv2d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x, T, B):
        x = self.conv1(x)
        x = x.reshape(T, B, x.shape[1], x.shape[2], x.shape[3]).contiguous()
        x = self.act(x)
        x = x.flatten(0, 1)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class Spiking_NCB(nn.Module):
    """
    Next Convolution Block
    """
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0,
                 drop=0, head_dim=32, mlp_ratio=3):
        super(Spiking_NCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim)
        self.attention_path_dropout = DropPath(path_dropout)

        self.norm = norm_layer(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        self.mlp_path_dropout = DropPath(path_dropout)
        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.flatten(0, 1)
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mhca(x, T, B))
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out, T, B))
        x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3]).contiguous()

        return x


class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention
    """
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        # self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        # self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(tau=2.,
                                    decay_input=True,
                                    v_threshold=1.,
                                    v_reset=0.,
                                    surrogate_function=surrogate.Sigmoid(),
                                    detach_reset=True,
                                    step_mode='m',
                                    backend='cupy')
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(tau=2.,
                                    decay_input=True,
                                    v_threshold=1.,
                                    v_reset=0.,
                                    surrogate_function=surrogate.Sigmoid(),
                                    detach_reset=True,
                                    step_mode='m',
                                    backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(tau=2.,
                                    decay_input=True,
                                    v_threshold=1.,
                                    v_reset=0.,
                                    surrogate_function=surrogate.Sigmoid(),
                                    detach_reset=True,
                                    step_mode='m',
                                    backend='cupy')
        self.attn_lif = neuron.LIFNode(tau=2.,
                                       decay_input=True,
                                       v_threshold=1.,
                                       v_reset=0.,
                                       surrogate_function=surrogate.Sigmoid(),
                                       detach_reset=True,
                                       step_mode='m',
                                       backend='cupy')

        self.proj_conv = nn.Conv1d(dim, self.out_dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(self.out_dim)
        # self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def forward(self, x):
        B,T,C,H,W = x.shape

        x = x.flatten(3)
        B, T, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(B, T, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(B, T, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(B, T, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # x = k.transpose(-2,-1) @ v
        # x = (q @ x) * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.attn_drop(attn)
        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_bn(self.proj_conv(x))
        x = self.proj_drop(x).reshape(B*T, C, H, W).contiguous()
        return x


class Spiking_NTB(nn.Module):
    """
    Next Transformer Block
    """
    def __init__(
            self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
            mlp_ratio=2, head_dim=32, mix_block_ratio=0.75, attn_drop=0, drop=0,
    ):
        super(Spiking_NTB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        norm_func = partial(nn.BatchNorm2d, eps=NORM_EPS)

        # self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 16)
        # self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 8)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(self.mhsa_out_channels, head_dim=head_dim, sr_ratio=sr_ratio,
                             attn_drop=attn_drop, proj_drop=drop)
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2 = norm_func(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_path_dropout = DropPath(path_dropout)

        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.e_mhsa.merge_bn(self.norm1)
            self.mlp.merge_bn(self.norm2)
            self.is_bn_merged = True

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.flatten(0, 1)
        x = self.patch_embed(x)

        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm1(x)
        else:
            out = x

        out = out.reshape(B, T, x.shape[1], x.shape[2], x.shape[3]).contiguous()
        # out = rearrange(out, "b c h w -> b (h w) c")  # b n c
        out = self.mhsa_path_dropout(self.e_mhsa(out))
        x = x + out
        # x = x + rearrange(out, "b (h w) c -> b c h w", h=H)

        out = self.projection(x)
        out = out + self.mhca_path_dropout(self.mhca(out, T, B))
        x = torch.cat([x, out], dim=1)

        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out, T, B))
        x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3]).contiguous()

        return x


class Spiking_EViT(nn.Module):
    def __init__(self, stem_chs, depths, path_dropout, attn_drop=0, drop=0, num_classes=1000,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75,
                 use_checkpoint=False, resume='', with_extra_norm=True, frozen_stages=-1,
                 norm_eval=False, norm_cfg=None,
                 ):
        super(Spiking_EViT, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.frozen_stages = frozen_stages
        self.with_extra_norm = with_extra_norm
        self.norm_eval = norm_eval

        self.stage_out_channels = [[32] * (depths[0]),
                                   [32] * (depths[1] - 1) + [64],
                                   [64, 64, 128] * (depths[2] // 3),
                                   [128] * (depths[3] - 1) + [128]]

        self.num_features = [32, 64, 128, 128]

        self.stage_block_types = [[Spiking_NCB] * depths[0],
                                  [Spiking_NCB] * (depths[1] - 1) + [Spiking_NTB],
                                  [Spiking_NCB, Spiking_NCB, Spiking_NTB] * (depths[2] // 3),
                                  [Spiking_NCB] * (depths[3] - 1) + [Spiking_NTB]]


        # Next Hybrid Strategy
        self.event_stem = nn.Sequential(
            ConvBNReLU(5, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )

        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is Spiking_NCB:
                    layer = Spiking_NCB(input_channel, output_channel, stride=stride, path_dropout=dpr[idx + block_id],
                                drop=drop, head_dim=head_dim)
                    features.append(layer)
                elif block_type is Spiking_NTB:
                    layer = Spiking_NTB(input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
                                sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio,
                                attn_drop=attn_drop, drop=drop)
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)

        self.extra_norm_list = None
        if with_extra_norm:
            self.extra_norm_list = []
            for stage_id in range(len(self.stage_out_channels)):
                self.extra_norm_list.append(nn.BatchNorm2d(
                    self.stage_out_channels[stage_id][-1], eps=NORM_EPS))
            self.extra_norm_list = nn.Sequential(*self.extra_norm_list)

        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)

        self.stage_out_idx = [sum(depths[:idx + 1]) - 1 for idx in range(len(depths))]

