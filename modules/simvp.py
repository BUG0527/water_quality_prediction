"""
SimVPv2 模型组件 —— 基于原作者开源代码精确复现
参考: https://github.com/buuuuuuug/SimVPv2 (CVPR 2022)

核心设计: 纯 2D CNN 视频预测
- Encoder: ConvSC 堆叠, GroupNorm + SiLU, 交替 stride-2 下采样
- Translator (Mid_IncepNet): 将 (B,T,C,H,W) → (B,T*C,H,W) 作为 2D 通道处理
  使用多尺度 Inception (3,5,7,11 kernel) + GroupConv + LeakyReLU + 编码-解码 + 跳跃连接
- Decoder: 镜像 Encoder, PixelShuffle 上采样, 跳跃连接
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ============================================================================
# timm 兼容层（避免外部依赖）
# ============================================================================

def _trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """截断正态分布初始化 (替代 timm trunc_normal_)"""
    with torch.no_grad():
        l = (a - mean) / std
        u = (b - mean) / std
        tensor.uniform_(max(0.0, (math.erf(l / math.sqrt(2.0)) + 1.0) / 2.0),
                        min(1.0, (math.erf(u / math.sqrt(2.0)) + 1.0) / 2.0))
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(a, b)


def _to_2tuple(x):
    return (x, x) if isinstance(x, int) else x


class DropPath(nn.Module):
    """DropPath (Stochastic Depth) — 替代 timm DropPath"""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ============================================================================
# SimVPv2 基础构建块 (simvp_model.py)
# ============================================================================

class BasicConv2d(nn.Module):
    """基础卷积块: Conv2d + GroupNorm(2, Cout) + SiLU，可选 PixelShuffle 上采样"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, upsampling=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size,
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
        if isinstance(m, nn.Conv2d):
            _trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    """空间卷积块: 支持下采样(stride=2) 或 上采样(PixelShuffle)"""
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False,
                 upsampling=False, act_norm=True, is_3d=False):
        super(ConvSC, self).__init__()
        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding, act_norm=act_norm)

    def forward(self, x):
        return self.conv(x)


class GroupConv2d(nn.Module):
    """分组卷积: GroupConv → GroupNorm → LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


def sampling_generator(N, reverse=False):
    """生成交替的 [False, True, False, True, ...] 下采样模式"""
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    else:
        return samplings[:N]


# ============================================================================
# Inception 翻译器 (gInception_ST + Mid_IncepNet)
# ============================================================================

class gInception_ST(nn.Module):
    """多尺度空间 Inception 块: 1×1 conv → parallel{不同 kernel 的 GroupConv} → sum"""
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super(gInception_ST, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)

        layers = []
        for ker in incep_ker:
            layers.append(
                GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1,
                           padding=ker // 2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class Mid_IncepNet(nn.Module):
    """
    SimVPv2 核心翻译器 (IncepU 变体):
    - 将 (B, T, C, H, W) reshape 为 (B, T*C, H, W) 作为 2D 通道
    - Encoder-Decoder 结构 + 跳跃连接
    - 每层使用 gInception_ST (多尺度 GroupConv Inception)
    """
    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3, 5, 7, 11], groups=8, **kwargs):
        super(Mid_IncepNet, self).__init__()
        self.N2 = N2

        # Encoder
        enc_layers = [gInception_ST(channel_in, channel_hid // 2, channel_hid,
                                     incep_ker=incep_ker, groups=groups)]
        for i in range(1, N2 - 1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid // 2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
            gInception_ST(channel_hid, channel_hid // 2, channel_hid,
                          incep_ker=incep_ker, groups=groups))

        # Decoder (with skip connections: input channels = 2 * channel_hid)
        dec_layers = [
            gInception_ST(channel_hid, channel_hid // 2, channel_hid,
                          incep_ker=incep_ker, groups=groups)]
        for i in range(1, N2 - 1):
            dec_layers.append(
                gInception_ST(2 * channel_hid, channel_hid // 2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
            gInception_ST(2 * channel_hid, channel_hid // 2, channel_in,
                          incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)  # 时间×通道 → 2D 通道

        # Encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2 - 1:
                skips.append(z)

        # Decoder (with skip connections)
        z = self.dec[0](z)
        for i in range(1, self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


# ============================================================================
# SimVPv2 Encoder / Decoder
# ============================================================================

class Encoder(nn.Module):
    """SimVPv2 空间编码器: 交替 stride 下采样"""
    def __init__(self, C_in, C_hid, N_S, spatio_kernel=3):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):  # (B*T, C, H, W)
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """SimVPv2 空间解码器: 交替 PixelShuffle 上采样 + 跳跃连接"""
    def __init__(self, C_hid, C_out, N_S, spatio_kernel=3):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s) for s in samplings[:-1]],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)  # 跳跃连接
        Y = self.readout(Y)
        return Y


# ============================================================================
# SimVP_Model (完整模型，供参考和对比)
# ============================================================================

class SimVP_Model(nn.Module):
    """
    标准 SimVPv2 模型 (单站点视频预测)

    输入: (B, T, C, H, W)
    输出: (B, T, C, H, W)  (相同帧数，预测 future 时需切片)
    """
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 model_type='IncepU', spatio_kernel_enc=3, spatio_kernel_dec=3,
                 **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape
        self.T = T
        self.C = C
        self.hid_S = hid_S
        self.N_S = N_S

        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec)

        if model_type == 'IncepU':
            self.hid = Mid_IncepNet(T * hid_S, hid_T, N_T)
        else:
            # GANet fallback
            self.hid = Mid_GANet(T * hid_S, hid_T, N_T)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y


# ============================================================================
# GANet 翻译器 (备选)
# ============================================================================

class Mlp(nn.Module):
    """ConvFFN: 1×1 conv → DWConv → GELU → 1×1 conv"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    """3×3 Depthwise Convolution"""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class AttentionModule(nn.Module):
    """Large Kernel Attention (LKA)"""
    def __init__(self, dim, kernel_size=21, dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p,
                                       groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2 * dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x


class SpatialAttention(nn.Module):
    """空间注意力: 1×1 → GELU → LKA → 1×1 + 残差"""
    def __init__(self, d_model, kernel_size=21):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x


class GASubBlock(nn.Module):
    """gSTA 子块: SpatialAttention + ConvFFN (Mlp), LayerScale, DropPath"""
    def __init__(self, dim, kernel_size=21, mlp_ratio=4., drop=0., drop_path=0.1,
                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        return x


class GABlock(nn.Module):
    """gSTA 块包装器 (可选通道变换)"""
    def __init__(self, in_channels, out_channels, mlp_ratio=8., drop=0.0,
                 drop_path=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = GASubBlock(in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        if in_channels != out_channels:
            self.reduction = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class Mid_GANet(nn.Module):
    """gSTA 翻译器: 堆叠 GABlock, 保持 (B, T*C, H, W) 维度"""
    def __init__(self, channel_in, channel_hid, N2, mlp_ratio=4., drop=0.0,
                 drop_path=0.1):
        super().__init__()
        self.N2 = N2
        enc_layers = [GABlock(channel_in, channel_hid, mlp_ratio=mlp_ratio,
                               drop=drop, drop_path=drop_path)]
        for i in range(1, N2 - 1):
            enc_layers.append(GABlock(channel_hid, channel_hid, mlp_ratio=mlp_ratio,
                                       drop=drop, drop_path=drop_path))
        enc_layers.append(GABlock(channel_hid, channel_in, mlp_ratio=mlp_ratio,
                                   drop=drop, drop_path=drop_path))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
        y = z.reshape(B, T, C, H, W)
        return y


# ============================================================================
# 空间尺寸计算工具
# ============================================================================

def compute_spatial_dims(input_size: int, N_S: int) -> int:
    """计算 N_S 层交替 stride 后的空间尺寸"""
    samplings = sampling_generator(N_S)
    size = input_size
    for s in samplings:
        if s:  # stride=2
            size = (size - 1) // 2 + 1
    return size


# ============================================================================
# 维度测试
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SimVPv2 (Original) Dimension Test")
    print("=" * 60)

    B, T, C, H, W = 4, 12, 6, 24, 24
    hid_S, hid_T, N_S, N_T = 64, 512, 4, 8

    print(f"Input: ({B}, {T}, {C}, {H}, {W})")
    Hp = compute_spatial_dims(H, N_S)
    Wp = compute_spatial_dims(W, N_S)
    print(f"After Encoder (N_S={N_S}): ({B}*{T}, {hid_S}, {Hp}, {Wp})")

    model = SimVP_Model(in_shape=(T, C, H, W), hid_S=hid_S, hid_T=hid_T,
                        N_S=N_S, N_T=N_T, model_type='IncepU')

    x = torch.randn(B, T, C, H, W)
    with torch.no_grad():
        y = model(x)

    print(f"Output: {y.shape} (expected: {B, T, C, H, W})")
    assert y.shape == (B, T, C, H, W), f"Shape mismatch: {y.shape}"

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print("\n[OK] SimVPv2 dimension tests passed")
