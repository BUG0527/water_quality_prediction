"""
SimVPv2 + TGGC 融合模型 —— 水质预测系统

基于原始开源代码精确复现:
- SimVPv2: https://github.com/buuuuuuug/SimVPv2 (CVPR 2022)
- TGGC:    https://github.com/KimMeen/TGGC (ICLR 2023)

完整数据流:
  Input:  (B, T_in, N_stations, C, H, W)    — 5站点GAF图像序列
    │
    ├─ Stage 1: SimVPv2 Encoder (5站点参数共享)
    │     for s in stations:
    │       enc(x_s) → feat_s (B*T_in, hid_S, H', W') + skip_s
    │       GAP(feat_s) → vec_s (B, T_in, hid_S)
    │     stack(vec) → (B, T_in, N, hid_S)
    │
    ├─ Stage 2: TGGC 图频谱-时序处理
    │     feat_coupled, attn = TGGC(feat_all)
    │     → (B, T_in, N, hid_S), (N, N)
    │
    ├─ Stage 3: 特征融合 (残差广播注入)
    │     feat_s_enhanced = feat_s + broadcast(proj(feat_coupled[:,:,s]))
    │
    ├─ Stage 4: SimVPv2 Mid_IncepNet Translator (参数共享)
    │     (B, T_in*hid_S, H', W') → 2D Inception → (B, T_in*hid_S, H', W')
    │
    └─ Stage 5: SimVPv2 Decoder (参数共享)
          dec(feat, skip) → (B, T_in, C, H, W) → slice → (B, T_out, C, H, W)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from modules.simvp import (
    Encoder, Decoder, Mid_IncepNet, Mid_GANet,
    compute_spatial_dims,
)
from modules.tggc import TGGC


class FeatureFusion(nn.Module):
    """
    特征融合层: TGGC 输出向量 → 广播回空间维度 → 残差注入

    与原始 SimVPv2 兼容，仅在 Encoder 和 Translator 之间插入。
    """

    def __init__(self, feat_dim: int, fusion_type: str = "residual"):
        super().__init__()
        self.feat_dim = feat_dim
        self.fusion_type = fusion_type
        self.proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, feat_map, coupled_vec):
        """
        Args:
            feat_map:    (B, T, D, H', W')  原始空间特征
            coupled_vec: (B, T, D)           TGGC 融合后的站点向量
        Returns:
            (B, T, D, H', W')  增强后的特征
        """
        B, T, D, Hp, Wp = feat_map.shape
        coupled_proj = self.proj(coupled_vec)
        coupled_broadcast = coupled_proj[:, :, :, None, None].expand(-1, -1, -1, Hp, Wp)

        if self.fusion_type == "residual":
            return feat_map + coupled_broadcast
        elif self.fusion_type == "gate":
            gate = torch.sigmoid(coupled_broadcast)
            return feat_map * (1.0 + gate)
        else:
            return feat_map + coupled_broadcast


class WaterQualityPredictor(nn.Module):
    """
    SimVPv2 + TGGC 水质预测融合模型

    Args:
        in_shape:       (C, H, W) 单帧图像形状
        hid_S:          Encoder/Decoder 隐藏通道数
        hid_T:          Translator 隐藏通道数
        N_S:            Encoder/Decoder 层数
        N_T:            Translator 层数
        T_in:           输入帧数
        T_out:          输出帧数 (≤ T_in)
        num_stations:   站点数量
        model_type:     Translator 类型 ('IncepU' / 'gSTA')
        tggc_order:     TGGC 图多项式阶数
        tggc_layers:    TGGC StockBlock 堆叠层数
        tggc_gconv:     图卷积类型 ('gegen' / 'cheby' / 'jacobi')
    """

    def __init__(
        self,
        in_shape=(6, 24, 24),
        hid_S=64,
        hid_T=512,
        N_S=4,
        N_T=8,
        T_in=12,
        T_out=6,
        num_stations=5,
        model_type='IncepU',
        tggc_order=4,
        tggc_layers=2,
        tggc_gconv='gegen',
        tggc_multi_layer=5,
        tggc_modes=5,
        fusion_type='residual',
    ):
        super().__init__()
        C, H, W = in_shape
        self.C = C
        self.H = H
        self.W = W
        self.hid_S = hid_S
        self.T_in = T_in
        self.T_out = T_out
        self.N_S = N_S
        self.N_T = N_T
        self.num_stations = num_stations

        # 计算编码后的空间尺寸
        self.H_prime = compute_spatial_dims(H, N_S)
        self.W_prime = compute_spatial_dims(W, N_S)

        # ---- SimVPv2 Encoder (5站点共享) ----
        self.encoder = Encoder(C, hid_S, N_S, spatio_kernel=3)

        # ---- TGGC 空间耦合 ----
        self.tggc = TGGC(
            N=num_stations, T=T_in, D=hid_S,
            order=tggc_order, gconv=tggc_gconv,
            layers=tggc_layers, multi_layer=tggc_multi_layer,
            modes=tggc_modes,
        )

        # ---- 特征融合 ----
        self.fusion = FeatureFusion(hid_S, fusion_type=fusion_type)

        # ---- SimVPv2 Translator (5站点共享) ----
        # Mid_IncepNet 输入 channel = T_in * hid_S (合并时间和通道维)
        translator_in_channels = T_in * hid_S
        if model_type == 'IncepU':
            self.translator = Mid_IncepNet(
                translator_in_channels, hid_T, N_T,
                incep_ker=[3, 5, 7, 11], groups=8,
            )
        else:
            self.translator = Mid_GANet(
                translator_in_channels, hid_T, N_T,
                mlp_ratio=8., drop=0.0, drop_path=0.1,
            )

        # ---- SimVPv2 Decoder (5站点共享) ----
        self.decoder = Decoder(hid_S, C, N_S, spatio_kernel=3)

    def _encode_station(self, x_s):
        """
        对单个站点编码。

        Args:
            x_s: (B, T_in, C, H, W)
        Returns:
            feat_map: (B, T_in, hid_S, H', W')
            skip:     (B*T_in, hid_S, H_enc1, W_enc1)  跳跃连接
            feat_vec: (B, T_in, hid_S)  GAP 向量
        """
        B, T, C, H, W = x_s.shape
        x_flat = x_s.reshape(B * T, C, H, W)

        embed, skip = self.encoder(x_flat)  # (B*T, hid_S, H', W'), (B*T, hid_S, H1, W1)

        _, _, Hp, Wp = embed.shape
        feat_map = embed.reshape(B, T, self.hid_S, Hp, Wp)

        # GAP
        feat_vec = F.adaptive_avg_pool2d(embed, 1).squeeze(-1).squeeze(-1)
        feat_vec = feat_vec.reshape(B, T, self.hid_S)

        return feat_map, skip, feat_vec

    def _translate_decode_station(self, feat_map, skip):
        """
        对增强后的单站点特征进行翻译 + 解码。

        Args:
            feat_map: (B, T_in, hid_S, H', W')
            skip:     (B*T_in, hid_S, H1, W1)
        Returns:
            (B, T_out, C, H, W)
        """
        B, T, D, Hp, Wp = feat_map.shape

        # Translator: (B, T, D, H, W) → (B, T*D, H, W) → → (B, T*D, H, W)
        x_trans = feat_map.reshape(B, T * D, Hp, Wp)
        # 添加时间维: → (B, 1, T*D, H, W) → Translator 内部会 reshape 回 T 维
        x_trans = x_trans.unsqueeze(1).expand(-1, 1, -1, Hp, Wp)
        # Mid_IncepNet expects (B, Tt, Ct, H, W) where Tt*Ct = translator channels
        # 我们需要: reshape (B, T*D, H, W) → 译者处理 → (B, T*D, H, W)
        # 直接调 internal: (B, T*D, H, W) 全在 channel 维

        # 这里的 Trick: Mid_IncepNet 期望输入 (B, T', C', H, W)
        # T'*C' = T * D. 令 T'=T, C'=D:
        trans_in = feat_map  # (B, T, D, H', W') naturally!
        trans_out = self.translator(trans_in)  # (B, T, D, H', W')

        # Reshape for Decoder: (B*T, D, H', W')
        dec_in = trans_out.reshape(B * T, D, Hp, Wp)

        # Decoder
        dec_out = self.decoder(dec_in, skip)  # (B*T, C, H, W)

        # Reshape + Slice
        pred = dec_out.reshape(B, T, self.C, self.H, self.W)

        if self.T_out < T:
            pred = pred[:, :self.T_out]

        return pred

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, T_in, N, C, H, W)
            return_attention: 是否返回 TGGC 注意力矩阵 (用于可解释性)
        Returns:
            dict:
              'pred':   (B, T_out, N, C, H, W)
              'attn':   (N, N) or None — 站点间邻接权重
        """
        B, T_in, N, C, H, W = x.shape
        assert N == self.num_stations
        assert T_in == self.T_in

        # ================================================================
        # Stage 1: Encoder (5站点共享权重)
        # ================================================================
        feat_maps = []  # [(B, T, D, H', W'), ...]
        skips = []      # [(B*T, D, H1, W1), ...]
        feat_vecs = []  # [(B, T, D), ...]

        for s in range(N):
            fmap, skip, fvec = self._encode_station(x[:, :, s])
            feat_maps.append(fmap)
            skips.append(skip)
            feat_vecs.append(fvec)

        # Stack: (B, T_in, N, D)
        feat_all = torch.stack(feat_vecs, dim=2)

        # ================================================================
        # Stage 2: TGGC 空间耦合
        # ================================================================
        feat_coupled, attn = self.tggc(feat_all, return_attention=return_attention)

        # ================================================================
        # Stage 3: 特征融合 (残差广播)
        # ================================================================
        enhanced_maps = []
        for s in range(N):
            coupled_s = feat_coupled[:, :, s, :]  # (B, T, D)
            enhanced = self.fusion(feat_maps[s], coupled_s)
            enhanced_maps.append(enhanced)

        # ================================================================
        # Stage 4 + 5: Translator + Decoder (5站点共享权重)
        # ================================================================
        preds = []
        for s in range(N):
            pred_s = self._translate_decode_station(enhanced_maps[s], skips[s])
            preds.append(pred_s)

        # Stack: (B, T_out, N, C, H, W)
        pred = torch.stack(preds, dim=2)

        return {'pred': pred, 'attn': attn}


# ============================================================================
# 维度测试
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  WaterQualityPredictor (Real Architectures)")
    print("=" * 60)

    B, T_in, N, C, H, W = 2, 12, 5, 6, 24, 24
    T_out = 6

    print(f"Input:  ({B}, {T_in}, {N}, {C}, {H}, {W})")
    print(f"Output: ({B}, {T_out}, {N}, {C}, {H}, {W})")

    model = WaterQualityPredictor(
        in_shape=(C, H, W),
        hid_S=64,     # Encoder/Decoder 通道
        hid_T=512,    # Translator 通道
        N_S=4,        # Encoder 层数
        N_T=8,        # Translator 层数
        T_in=T_in,
        T_out=T_out,
        num_stations=N,
        model_type='IncepU',
        tggc_order=4,
        tggc_layers=2,
    )

    x = torch.randn(B, T_in, N, C, H, W)
    with torch.no_grad():
        output = model(x, return_attention=True)

    pred = output['pred']
    attn = output['attn']

    print(f"\npred: {list(pred.shape)} (expected: [{B}, {T_out}, {N}, {C}, {H}, {W}])")
    print(f"attn: {list(attn.shape)} (expected: [{N}, {N}])")

    assert pred.shape == (B, T_out, N, C, H, W), f"Shape mismatch: {pred.shape}"
    assert attn.shape == (N, N), f"Attention shape: {attn.shape}"

    # 参数量统计
    enc_p = sum(p.numel() for p in model.encoder.parameters())
    tggc_p = sum(p.numel() for p in model.tggc.parameters())
    trans_p = sum(p.numel() for p in model.translator.parameters())
    dec_p = sum(p.numel() for p in model.decoder.parameters())
    fus_p = sum(p.numel() for p in model.fusion.parameters())
    total = enc_p + tggc_p + trans_p + dec_p + fus_p

    print(f"\nParameters:")
    print(f"  Encoder:    {enc_p:,}")
    print(f"  TGGC:       {tggc_p:,}")
    print(f"  Translator: {trans_p:,}")
    print(f"  Decoder:    {dec_p:,}")
    print(f"  Fusion:     {fus_p:,}")
    print(f"  Total:      {total:,}")

    print("\n[OK] WaterQualityPredictor dimension tests passed")
