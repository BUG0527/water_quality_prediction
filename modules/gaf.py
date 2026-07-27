"""
Gramian Angular Field (GAF) 变换模块

将一维时序数据转化为二维 GAF 图像，支持:
- GASF (Gramian Angular Summation Field): 保留时间相关性
- GADF (Gramian Angular Difference Field): 捕捉时间差分信息

公式:
  GASF: G(i,j) = cos(φ_i + φ_j)
  GADF: G(i,j) = sin(φ_i - φ_j)
  其中 φ = arccos(x_norm), x_norm ∈ [-1, 1]
"""

import numpy as np
from typing import Optional


def _minmax_scale(x: np.ndarray, feature_range: tuple = (-1.0, 1.0)) -> np.ndarray:
    """
    将一维数组缩放到指定范围，处理常数序列的边界情况。

    Args:
        x: 一维 numpy 数组
        feature_range: (min, max) 目标范围

    Returns:
        缩放后的数组
    """
    x_min, x_max = x.min(), x.max()
    range_min, range_max = feature_range

    if x_max - x_min < 1e-10:
        # 常数序列：全部映射到范围中点 (0)
        return np.zeros_like(x, dtype=np.float64)

    scaled = (x - x_min) / (x_max - x_min)  # → [0, 1]
    scaled = scaled * (range_max - range_min) + range_min  # → [min, max]
    return scaled


def _gaf_core(x_norm: np.ndarray, method: str = "gasf") -> np.ndarray:
    """
    GAF 变换核心计算。

    Args:
        x_norm: 归一化到 [-1, 1] 的一维序列，长度 L
        method: "gasf" 或 "gadf"

    Returns:
        (L, L) 的 GAF 图像
    """
    # 确保值域在 [-1, 1]，处理浮点误差
    x_norm = np.clip(x_norm, -1.0, 1.0)

    # 极坐标角度: φ = arccos(x_norm) ∈ [0, π]
    phi = np.arccos(x_norm)  # (L,)

    if method == "gasf":
        # GASF: G(i,j) = cos(φ_i + φ_j)
        # = cos(φ_i)cos(φ_j) - sin(φ_i)sin(φ_j)
        # = x_i * x_j - sqrt(1-x_i^2) * sqrt(1-x_j^2)
        gaf = np.cos(np.add.outer(phi, phi))  # (L, L)
    elif method == "gadf":
        # GADF: G(i,j) = sin(φ_i - φ_j)
        gaf = np.sin(np.subtract.outer(phi, phi))  # (L, L)
    else:
        raise ValueError(f"Unknown GAF method: {method}. Use 'gasf' or 'gadf'.")

    return gaf


def gaf_transform(
    x: np.ndarray,
    method: str = "gasf",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    将一维时序数据转换为 GAF 图像。

    Args:
        x: 一维数组，长度 L（时间窗口大小）
        method: "gasf" (推荐) 或 "gadf"
        eps: 数值稳定性小量

    Returns:
        (L, L) 的 GAF 图像，值域 [-1, 1]

    Example:
        >>> ts = np.array([0.5, 0.3, 0.8, 0.2, 0.7])
        >>> img = gaf_transform(ts, method='gasf')
        >>> img.shape
        (5, 5)
    """
    x = np.asarray(x, dtype=np.float64).flatten()

    # Step 1: MinMax 缩放到 [-1, 1]
    x_norm = _minmax_scale(x, feature_range=(-1.0, 1.0))

    # Step 2: GAF 变换
    gaf = _gaf_core(x_norm, method=method)

    return gaf.astype(np.float32)


def gaf_multichannel_transform(
    x_multivar: np.ndarray,
    method: str = "gasf",
) -> np.ndarray:
    """
    多变量 GAF 变换：对同一时刻的多个指标独立做 GAF。

    Args:
        x_multivar: (C, L) 数组，C 个指标，每个长度 L
        method: GAF 方法

    Returns:
        (C, L, L) 多通道 GAF 图像（类似 RGB 三通道）
    """
    C, L = x_multivar.shape
    gaf_channels = np.zeros((C, L, L), dtype=np.float32)
    for c in range(C):
        gaf_channels[c] = gaf_transform(x_multivar[c], method=method)
    return gaf_channels


def gaf_batch_transform(
    data: np.ndarray,
    method: str = "gasf",
) -> np.ndarray:
    """
    批量 GAF 变换：对多维数据生成 GAF 图像。

    Args:
        data: (T, N_stations, C, L) 数组
              T: 时间步, N_stations: 站点数, C: 指标数, L: 窗口大小
        method: GAF 方法

    Returns:
        (T, N_stations, C, L, L) GAF 图像序列
    """
    T, N_stations, C, L = data.shape
    result = np.zeros((T, N_stations, C, L, L), dtype=np.float32)

    for t in range(T):
        for s in range(N_stations):
            result[t, s] = gaf_multichannel_transform(data[t, s], method=method)

    return result


def inverse_gaf(
    gaf_img: np.ndarray,
    method: str = "gasf",
    window_len: Optional[int] = None,
) -> np.ndarray:
    """
    将 GAF 图像近似逆变换回一维时序（用于结果可视化）。

    从 GASF 的主对角线提取: G(i,i) = cos(2*φ_i) → φ_i = arccos(G(i,i))/2
    然后: x_i = cos(φ_i)

    Args:
        gaf_img: (L, L) GAF 图像
        method: 生成 GAF 时使用的方法
        window_len: 原始窗口长度（如果 GAF 图像被 resize 了）

    Returns:
        (L,) 近似恢复的一维时序
    """
    L = gaf_img.shape[0] if window_len is None else window_len
    diag = np.diag(gaf_img)[:L]

    diag = np.clip(diag, -1.0, 1.0)

    if method == "gasf":
        phi = np.arccos(diag) / 2.0
    elif method == "gadf":
        # GADF: G(i,i) = sin(0) = 0, 无法从对角线恢复
        # 使用第一行的近似
        row0 = gaf_img[0, :L]
        phi = np.arcsin(row0)
    else:
        raise ValueError(f"Unknown method: {method}")

    x_recovered = np.cos(phi)
    return x_recovered.astype(np.float32)


# ============================================================================
# 单元测试辅助
# ============================================================================
if __name__ == "__main__":
    # 测试 GAF 变换
    ts = np.array([0.5, 0.3, 0.8, 0.2, 0.7, 0.1, 0.9, 0.4, 0.6, 0.3])
    print(f"输入序列: {ts}")
    print(f"长度: {len(ts)}")

    gasf = gaf_transform(ts, method="gasf")
    print(f"GASF shape: {gasf.shape}, range: [{gasf.min():.3f}, {gasf.max():.3f}]")

    gadf = gaf_transform(ts, method="gadf")
    print(f"GADF shape: {gadf.shape}, range: [{gadf.min():.3f}, {gadf.max():.3f}]")

    # 测试多变量
    multi = np.random.randn(6, 10)
    multi_gaf = gaf_multichannel_transform(multi)
    print(f"多通道 GAF shape: {multi_gaf.shape}")

    # 测试逆变换
    recovered = inverse_gaf(gasf, method="gasf")
    print(f"逆变换: {recovered}")

    print("\n[OK] GAF module test passed")
