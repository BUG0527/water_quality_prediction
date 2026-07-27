"""
评估指标计算

支持:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- SSIM (Structural Similarity Index Measure) — 使用 1 通道滑动窗口
- PSNR (Peak Signal-to-Noise Ratio)

对 GAF 图像 (值域 [-1, 1]) 和原始水质数据均可计算。
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算 MSE。"""
    return F.mse_loss(pred, target).item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算 MAE。"""
    return F.l1_loss(pred, target).item()


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算 RMSE。"""
    return math.sqrt(F.mse_loss(pred, target).item())


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算 PSNR。

    GAF 图像值域 [-1, 1] → max_val = 2.0
    """
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return float("inf")
    max_val = target.max() - target.min()
    return (20.0 * math.log10(max_val.item())) - (10.0 * math.log10(mse.item()))


def _gaussian_kernel(
    window_size: int, sigma: float, channels: int = 1
) -> torch.Tensor:
    """生成 1D 高斯核 → 2D 外积。"""
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g /= g.sum()
    g_2d = g[:, None] * g[None, :]
    return g_2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_single_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """
    单通道 SSIM 计算。

    Args:
        pred:   (B, 1, H, W)
        target: (B, 1, H, W)
    Returns:
        (B,) SSIM 值
    """
    kernel = _gaussian_kernel(window_size, 1.5, 1).to(pred.device)

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=1)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred**2, kernel, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(target**2, kernel, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=1) - mu12

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2.0 * mu12 + C1) * (2.0 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean(dim=[1, 2, 3])


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算多通道 SSIM 的平均值。

    Args:
        pred:   (B, C, H, W) 或 (N, B, C, H, W)
        target: 同 shape

    Returns:
        平均 SSIM 值
    """
    if pred.dim() == 5:
        # (N, B, C, H, W) → 展平
        pred = pred.reshape(-1, *pred.shape[-3:])
        target = target.reshape(-1, *target.shape[-3:])

    B, C = pred.shape[:2]
    ssim_vals = []
    for c in range(C):
        ssim_c = _ssim_single_channel(
            pred[:, c:c + 1],
            target[:, c:c + 1],
        )
        ssim_vals.append(ssim_c.mean())
    return torch.stack(ssim_vals).mean().item()


# ============================================================================
# 综合指标
# ============================================================================

def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    prefix: str = "",
) -> Dict[str, float]:
    """
    计算所有常用指标。

    Args:
        pred:   (B, T_out, N, C, H, W) 或 (N_samples, C, H, W)
        target: 同 shape
        prefix: 指标名前缀 (如 "val/", "test/")

    Returns:
        {name: value} 字典
    """
    # 确保在 CPU 上计算
    pred = pred.detach().cpu()
    target = target.detach().cpu()

    # 对于 6D 数据，展平 batch, time, station 维度
    if pred.dim() == 6:
        pred = pred.reshape(-1, pred.shape[-3], pred.shape[-2], pred.shape[-1])
        target = target.reshape(-1, target.shape[-3], target.shape[-2], target.shape[-1])

    metrics = {
        f"{prefix}mse": compute_mse(pred, target),
        f"{prefix}mae": compute_mae(pred, target),
        f"{prefix}rmse": compute_rmse(pred, target),
    }

    # SSIM 和 PSNR 只在图像维度足够时计算
    if pred.shape[-1] >= 7:
        metrics[f"{prefix}ssim"] = compute_ssim(pred, target)
        metrics[f"{prefix}psnr"] = compute_psnr(pred, target)

    return metrics


def compute_per_station_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    station_names: list = None,
) -> Dict[str, Dict[str, float]]:
    """
    分站点计算指标。

    Args:
        pred:  (B, T_out, N, C, H, W)
        target: 同 shape
        station_names: 站点名称列表

    Returns:
        {station_name: {metric_name: value}}
    """
    N = pred.shape[2]
    if station_names is None:
        station_names = [f"Station_{i+1}" for i in range(N)]

    result = {}
    for s in range(N):
        pred_s = pred[:, :, s]
        target_s = target[:, :, s]
        result[station_names[s]] = compute_all_metrics(
            pred_s, target_s, prefix=""
        )
    return result


def compute_per_indicator_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    indicator_names: list = None,
) -> Dict[str, Dict[str, float]]:
    """
    分指标计算误差。

    Args:
        pred:   (N_samples, C, H, W)
        target: 同 shape
        indicator_names: 指标名称列表

    Returns:
        {indicator_name: {metric_name: value}}
    """
    C = pred.shape[-3]
    if indicator_names is None:
        indicator_names = [f"Indicator_{i+1}" for i in range(C)]

    result = {}
    for c in range(C):
        pred_c = pred[:, c:c + 1]
        target_c = target[:, c:c + 1]
        result[indicator_names[c]] = compute_all_metrics(
            pred_c, target_c, prefix=""
        )
    return result


# ============================================================================
# 测试
# ============================================================================
if __name__ == "__main__":
    B, T_out, N, C, H, W = 4, 6, 5, 6, 24, 24
    pred = torch.randn(B, T_out, N, C, H, W)
    target = torch.randn(B, T_out, N, C, H, W)

    metrics = compute_all_metrics(pred, target, prefix="test/")
    print("综合指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    per_station = compute_per_station_metrics(pred, target)
    print("\n分站点 MSE:")
    for station, m in per_station.items():
        print(f"  {station}: {m['mse']:.6f}")

    print("\n[OK] Metrics module tests passed")
