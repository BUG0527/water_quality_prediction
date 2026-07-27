"""
TGGC (Temporal Graph-Guided Convolutional) 模块 —— 基于原作者代码复现
参考: https://github.com/KimMeen/TGGC (ICLR 2023)

核心机制:
1. 隐式图学习: GRU + Self-Attention → 邻接矩阵
2. 频谱图卷积: Gegenbauer/Chebyshev 多项式展开 → mul_L (order, N, N)
3. FFT 频域滤波: FourierBlock 沿时间轴做频谱选择
4. 序列分解: moving_avg → trend + residual
5. Stacked Blocks: forecast/backcast 残差学习

适配: 扩展支持多维节点特征 (B, T, N, D) → (B, T, N, D)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# 频率模式选择
# ============================================================================

def get_frequency_modes(seq_len, modes=4, mode_select_method='random'):
    """选择 FFT 频率分量 (低频优先或随机抽样)"""
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


# ============================================================================
# 移动平均 / 序列分解 (原作者代码)
# ============================================================================

class moving_avg(nn.Module):
    """1D 移动平均 (用于趋势-残差分解)"""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (B, T, D)
        front = x[:, 0:1, :].repeat(
            1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(
            1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """单尺度序列分解: residual = x - moving_avg(x)"""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """多尺度序列分解: 多个 kernel 加权组合"""
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg_out = func(x)
            moving_mean.append(moving_avg_out.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(
            moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


# ============================================================================
# FourierBlock: FFT 频域滤波 (原作者代码)
# ============================================================================

class FourierBlock(nn.Module):
    """
    频域滤波块: 沿时间维做 FFT → 选择频率分量 → 可学习频谱权重 → IFFT
    用于捕捉全局时序依赖
    """
    def __init__(self, node, in_channels, out_channels, seq_len,
                 modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        self.index = get_frequency_modes(seq_len, modes=modes,
                                          mode_select_method=mode_select_method)
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(node, in_channels, out_channels,
                                     len(self.index), dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q):
        B, D, N, L = q.shape
        x = q.permute(0, 2, 1, 3)  # (B, N, D, L)
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, N, D, L//2+1) complex
        out_ft = torch.zeros(B, N, D, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            if i < x_ft.shape[-1]:
                out_ft[:, :, :, wi] = self.compl_mul1d(
                    x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        output = torch.fft.irfft(out_ft, n=x.size(-1)).permute(0, 2, 1, 3)
        return output, None


# ============================================================================
# StockBlockLayer: 图频谱 + FFT + 序列分解 (原作者代码, 适配多维特征)
# ============================================================================

class StockBlockLayer(nn.Module):
    """
    TGGC 核心块: 图频谱卷积 → FFT 频域滤波 → 序列分解 → forecast/backcast

    适配: 支持 D 维节点特征, 使用 order 阶图多项式
    """

    def __init__(self, time_step, unit, multi_layer, Fourier_option,
                 order=4, non_linear='linear', modes=4, activation='softmax',
                 stack_cnt=0, node_dim=256):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.node_dim = node_dim
        self.order = order
        self.non_linear = non_linear
        self.modes = modes
        self.activation = activation

        # 图卷积权重: (1, order, D, time_step, multi * time_step)
        self.weight = nn.Parameter(
            torch.randn(1, order, node_dim, time_step, self.multi * time_step) * 0.01)

        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.dropout = nn.Dropout(0.2)

        if Fourier_option == 'FB':
            self.Fourier = FourierBlock(
                node=self.unit, in_channels=order, out_channels=order,
                seq_len=time_step, modes=self.modes, mode_select_method='random')

        moving_avg_list = [2]
        if isinstance(moving_avg_list, list):
            self.decomp1 = series_decomp_multi(moving_avg_list)
            self.decomp2 = series_decomp_multi(moving_avg_list)
        else:
            self.decomp1 = series_decomp(moving_avg_list)
            self.decomp2 = series_decomp(moving_avg_list)

    def spe_seq_cell(self, input):
        """频谱序列处理: FourierBlock → 序列分解"""
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input_flat = input.view(batch_size, -1, node_cnt, time_step)
        new_x, _ = self.Fourier(input_flat)
        x = input_flat + self.dropout(new_x)
        xt = x.permute(0, 2, 3, 1).reshape(batch_size * node_cnt, time_step, -1)
        x_s, x_t = self.decomp1(xt)
        x = x_s + x_t
        x = x.reshape(batch_size, node_cnt, time_step, -1).permute(0, 3, 1, 2)
        return x

    def forward(self, x, mul_L):
        """
        x:     (B, D, N, T) — D 维节点特征
        mul_L: (order, N, N) — 图多项式
        Returns: forecast, backcast_source, x_back
        """
        # 图频谱变换: mul_L @ x  (order, N, N) @ (B, D, N, T) → (B, order, D, N, T)
        x = x.unsqueeze(1)  # (B, 1, D, N, T)
        mul_L_exp = mul_L.unsqueeze(0).unsqueeze(2)  # (1, order, 1, N, N)
        gfted = torch.einsum('bonm,bidmt->biont', mul_L_exp, x)  # (B, order, D, N, T)

        # 可学习频谱权重
        gfted = torch.einsum('biont,oidtm->bionm', gfted, self.weight)
        # → (B, order, D, N, multi*T)

        # FFT + 序列分解
        gfted = gfted.reshape(-1, self.order, self.node_dim, self.unit, self.time_step * self.multi)
        igfted = self.spe_seq_cell(gfted).unsqueeze(2)  # (B', order, 1, D, N, T)
        igfted = igfted.reshape(-1, self.order, self.node_dim, self.unit, self.time_step)

        # 求和所有 order 层
        igfted = torch.sum(igfted, dim=1)  # (B, D, N, T)

        # Reshape for forecast/backcast
        igfted_flat = igfted.permute(0, 2, 3, 1).reshape(-1, self.time_step * self.node_dim)

        # Forecast / Backcast
        if self.non_linear == 'linear':
            forecast_source = self.forecast(igfted_flat)
        elif self.non_linear == 'relu':
            forecast_source = torch.relu(self.forecast(igfted_flat))

        forecast = self.forecast_result(forecast_source)

        x_flat = x.squeeze(1).permute(0, 2, 3, 1).reshape(-1, self.time_step * self.node_dim)
        backcast_short = self.backcast_short_cut(x_flat)
        backcast_source = torch.sigmoid(backcast_short - self.backcast(igfted_flat))
        x_back = torch.sigmoid(self.backcast(igfted_flat))

        # Reshape outputs
        forecast = forecast.reshape(-1, self.unit, self.time_step, self.node_dim).permute(0, 3, 1, 2)
        backcast_source = backcast_source.reshape(-1, self.unit, self.time_step, self.node_dim).permute(0, 3, 1, 2)
        x_back = x_back.reshape(-1, self.unit, self.time_step, self.node_dim).permute(0, 3, 1, 2)

        return forecast, backcast_source, x_back


# ============================================================================
# TGGC 完整模块
# ============================================================================

class TGGC(nn.Module):
    """
    Temporal Graph-Guided Convolutional Network

    适配版本: 处理多维节点特征 (B, T, N, D) → (B, T, N, D)

    核心流程:
    1. 自注意力学习邻接矩阵
    2. Gegenbauer 多项式展开图拉普拉斯
    3. 堆叠 StockBlockLayer 做图频谱 + 时序处理
    4. 残差融合输出

    Args:
        N:          节点数 (站点数)
        T:          时间步
        D:          特征维度
        order:      图多项式阶数 (默认 4)
        gconv:      图卷积类型 ('gegen' / 'cheby' / 'jacobi')
        layers:     StockBlock 堆叠层数
        multi_layer: 时序多尺度因子
        modes:      FFT 频率分量数
        coe_a:      Gegenbauer 参数 α
        coe_b:      Jacobi 参数 β
    """

    def __init__(self, N=5, T=12, D=256, order=4, gconv='gegen',
                 layers=2, multi_layer=5, modes=5, coe_a=1.2, coe_b=1.0,
                 dropout_rate=0.4, leaky_rate=0.02, non_linear='linear',
                 Fouropt='FB', attention_set='linear', activation='softmax'):
        super(TGGC, self).__init__()
        self.unit = N
        self.time_step = T
        self.node_dim = D
        self.order = order
        self.gconv = gconv
        self.layers = layers
        self.coe_a = coe_a
        self.coe_b = coe_b

        # ---- 图学习: Self-Attention ----
        self.weight_key = nn.Parameter(torch.zeros(N, 1))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(N, 1))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(leaky_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

        # ---- GRU 编码 ----
        self.GRU = nn.GRU(self.time_step, self.unit)

        # ---- StockBlock 堆叠 ----
        self.stock_block = nn.ModuleList()
        for i in range(self.layers):
            self.stock_block.append(
                StockBlockLayer(self.time_step, self.unit, multi_layer,
                                Fourier_option=Fouropt, order=order,
                                non_linear=non_linear, modes=modes,
                                activation=activation, stack_cnt=i,
                                node_dim=D))

        # ---- 输出投影 ----
        self.fc_out = nn.Sequential(
            nn.Linear(int(self.time_step * D), int(self.time_step * D)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step * D), int(self.time_step * D)),
        )
        self.fc_back = nn.Sequential(
            nn.Linear(int(self.time_step * D), int(self.time_step * D)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step * D), int(self.time_step * D)),
        )

    # ========================================================================
    # 图拉普拉斯 & 多项式
    # ========================================================================

    def get_laplacian(self, graph, normalize=True):
        """对称归一化拉普拉斯: L = I - D^(-1/2) A D^(-1/2)"""
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - \
                torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def Cheb_polynomial(self, laplacian):
        """Chebyshev 多项式: T_0=I, T_1=L, T_k=2L·T_{k-1}-T_{k-2}"""
        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.ones([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = 2 * torch.matmul(laplacian, second_laplacian) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat(
            [first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def Gegen_coe(self, k, a=1, l=-1.0, r=1.0):
        return [2 * (k + a - 1), k + 2 * (a - 1)]

    def Gegen_polynomial(self, laplacian, a=1, order=4):
        """Gegenbauer 多项式 (推荐, 效果最好)"""
        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.ones([1, N, N], device=laplacian.device, dtype=torch.float)

        if order == 1:
            return first_laplacian

        second_laplacian = 2 * a * laplacian
        if order == 2:
            return torch.cat([first_laplacian, second_laplacian], dim=0)

        Theta_2 = self.Gegen_coe(2, a)
        third_laplacian = 1.0 / 2 * (torch.matmul((Theta_2[0] * laplacian), second_laplacian)
                                       - Theta_2[1] * first_laplacian)
        if order == 3:
            return torch.cat([first_laplacian, second_laplacian, third_laplacian], dim=0)

        Theta_3 = self.Gegen_coe(3, a)
        forth_laplacian = 1.0 / 3 * (torch.matmul((Theta_3[0] * laplacian), third_laplacian)
                                       - Theta_3[1] * second_laplacian)
        return torch.cat([first_laplacian, second_laplacian, third_laplacian,
                           forth_laplacian], dim=0)

    def Jacobi_polynomial(self, laplacian, a=1, b=1):
        """Jacobi 多项式"""
        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.ones([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = (a - b) / 2 + (a + b + 2) / 2 * laplacian

        def Jacobi_coe(k, a, b):
            t0 = (2*k+a+b) * (2*k+a+b-1) / (2*k*(k+a+b))
            t1 = (2*k+a+b-1) * (a*a-b*b) / (2*k*(k+a+b)*(2*k+a+b-2))
            t2 = (k+a-1) * (k+b-1) * (2*k+a+b) / (k*(k+a+b)*(2*k+a+b-2))
            return [t0, t1, t2]

        T2 = Jacobi_coe(2, a, b)
        third = torch.matmul((T2[0]*laplacian+T2[1]), second_laplacian) + T2[2]*first_laplacian
        T3 = Jacobi_coe(3, a, b)
        forth = torch.matmul((T3[0]*laplacian+T3[1]), third) + T3[2]*second_laplacian
        return torch.cat([first_laplacian, second_laplacian, third, forth], dim=0)

    # ========================================================================
    # 隐式图学习
    # ========================================================================

    def latent_correlation_layer(self, x):
        """
        自注意力学习节点间相关关系。

        x: (B, T, N, D) → 聚合 D 维后学习 (N, N) 邻接矩阵
        Returns: mul_L (order, N, N), attention (N, N)
        """
        B, T, N, D = x.shape

        # 聚合特征维度: (B, T, N, D) → (B, T, N)
        x_agg = x.mean(dim=-1)

        # GRU 编码: (T, B, N) → (T, B, N)
        input_gru = x_agg.permute(1, 0, 2).contiguous()  # (T, B, N)
        out, _ = self.GRU(input_gru)
        out = out.permute(1, 0, 2).contiguous()  # (B, T, N)

        # 自注意力
        attention = self.self_graph_attention(out)  # (B, N, N)
        attention = torch.mean(attention, dim=0)  # (N, N)

        # 对称化 + 归一化
        degree = torch.sum(attention, dim=1)
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(
            diagonal_degree_hat,
            torch.matmul(degree_l - attention, diagonal_degree_hat))

        # 多项式展开
        if self.gconv == 'cheby':
            mul_L = self.Cheb_polynomial(laplacian)
        elif self.gconv == 'jacobi':
            mul_L = self.Jacobi_polynomial(laplacian, self.coe_a, self.coe_b)
        elif self.gconv == 'gegen':
            mul_L = self.Gegen_polynomial(laplacian, self.coe_a, r=self.order)
        else:
            mul_L = self.Cheb_polynomial(laplacian)

        return mul_L, attention

    def self_graph_attention(self, input):
        """单头自注意力计算"""
        input = input.permute(0, 2, 1).contiguous()  # (B, N, T)
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)    # (B, N, 1)
        query = torch.matmul(input, self.weight_query) # (B, N, 1)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + \
               query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    # ========================================================================
    # Forward
    # ========================================================================

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, T, N, D) — 站点特征 (已通过 GAP)
            return_attention: 是否返回注意力权重
        Returns:
            feat_coupled: (B, T, N, D) — 融合空间耦合信息的特征
            attention: (N, N) or None — 站点间邻接矩阵
        """
        B, T, N, D = x.shape

        # ---- 1. 图学习 ----
        mul_L, attention = self.latent_correlation_layer(x)  # mul_L: (order, N, N)

        # ---- 2. 重塑输入 ----
        # (B, T, N, D) → (B, D, N, T)
        X = x.permute(0, 3, 2, 1).contiguous()

        # ---- 3. Stacked Block 处理 ----
        result = []
        for stack_i in range(self.layers):
            forecast, X, _ = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)

        # 求和所有层的 forecast: each (B, D, N, T)
        forecast = torch.stack(result).sum(0)  # (B, D, N, T)

        # ---- 4. 输出投影 ----
        forecast_flat = forecast.permute(0, 2, 3, 1).reshape(B * N, T * D)
        coupled_flat = self.fc_out(forecast_flat)
        coupled = coupled_flat.reshape(B, N, T, D).permute(0, 2, 1, 3)

        # ---- 5. 残差连接 ----
        feat_coupled = x + 0.1 * coupled

        if return_attention:
            return feat_coupled, attention
        return feat_coupled, None


# ============================================================================
# 维度测试
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TGGC (Original Adapted) Dimension Test")
    print("=" * 60)

    B, T, N, D = 4, 12, 5, 256
    print(f"Input: ({B}, {T}, {N}, {D})")

    tggc = TGGC(N=N, T=T, D=D, order=4, gconv='gegen', layers=2,
                multi_layer=5, modes=5)

    x = torch.randn(B, T, N, D)
    coupled, attn = tggc(x, return_attention=True)

    print(f"feat_coupled: {coupled.shape} (expected: {B, T, N, D})")
    print(f"attention:    {attn.shape} (expected: {N, N})")
    assert coupled.shape == (B, T, N, D), f"Shape mismatch: {coupled.shape}"
    assert attn.shape == (N, N), f"Attention shape: {attn.shape}"

    params = sum(p.numel() for p in tggc.parameters())
    print(f"Parameters: {params:,}")
    print("\n[OK] TGGC dimension tests passed")
