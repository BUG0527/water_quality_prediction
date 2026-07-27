"""
可视化工具

提供以下图表:
- GAF 图像对比 (预测 vs 真实)
- α 权重热力图 (站点间影响)
- 训练 Loss 曲线
- 逐帧预测对比
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，适合服务器
import matplotlib.pyplot as plt
from typing import Optional, List

# ============================================================================
# 全局样式设置
# ============================================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "image.cmap": "RdYlBu_r",
})

# 支持中文字体（如果可用）
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def plot_gaf_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    indicator_names: Optional[List[str]] = None,
    station_name: str = "Station_1",
    time_idx: int = 0,
    save_path: Optional[str] = None,
):
    """
    绘制预测 vs 真实 GAF 图像对比图。

    Args:
        pred:   (C, H, W) 预测的 GAF 图像
        target: (C, H, W) 真实的 GAF 图像
        indicator_names: 指标名称列表
        station_name:    站点名
        time_idx:        时间索引
        save_path:       保存路径 (None = 显示)
    """
    C = pred.shape[0]
    if indicator_names is None:
        indicator_names = [f"Ind_{i+1}" for i in range(C)]

    fig, axes = plt.subplots(3, C, figsize=(3 * C, 9))

    for c in range(C):
        # 预测
        im0 = axes[0, c].imshow(pred[c], vmin=-1, vmax=1, aspect="auto")
        axes[0, c].set_title(f"{indicator_names[c]} (Pred)")
        axes[0, c].axis("off")

        # 真实
        im1 = axes[1, c].imshow(target[c], vmin=-1, vmax=1, aspect="auto")
        axes[1, c].set_title(f"{indicator_names[c]} (GT)")
        axes[1, c].axis("off")

        # 残差
        im2 = axes[2, c].imshow(pred[c] - target[c], aspect="auto",
                                cmap="coolwarm")
        axes[2, c].set_title(f"{indicator_names[c]} (Error)")
        axes[2, c].axis("off")

    plt.suptitle(f"{station_name} | t={time_idx} | GAF 预测对比",
                 fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_alpha_heatmap(
    alpha: np.ndarray,
    station_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    绘制 TGGC 的 α 权重热力图 —— 展示站点间 K 阶频谱影响。

    Args:
        alpha: (N, N, K) 影响权重矩阵
        station_names: 站点名称
        save_path: 保存路径
    """
    N, _, K = alpha.shape
    if station_names is None:
        station_names = [f"S{i+1}" for i in range(N)]

    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4))
    if K == 1:
        axes = [axes]

    for k in range(K):
        im = axes[k].imshow(alpha[:, :, k], cmap="YlOrRd", aspect="auto")
        axes[k].set_title(f"阶数 k={k}")
        axes[k].set_xticks(range(N))
        axes[k].set_xticklabels(station_names, rotation=45)
        axes[k].set_yticks(range(N))
        axes[k].set_yticklabels(station_names)
        plt.colorbar(im, ax=axes[k], shrink=0.8)

    fig.suptitle("TGGC 站点间频谱影响权重 α(N,N,K)", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
):
    """
    绘制训练/验证 loss 曲线。

    Args:
        train_losses: 各 epoch 训练 loss
        val_losses:   各 epoch 验证 loss
        save_path:    保存路径
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, label="Train Loss", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val Loss", color="#FF5722", linewidth=1.5)

    # 标注最佳 epoch
    if val_losses:
        best_epoch = np.argmin(val_losses) + 1
        best_loss = val_losses[best_epoch - 1]
        ax.scatter([best_epoch], [best_loss], color="red", s=80, zorder=5)
        ax.annotate(
            f"Best: epoch={best_epoch}\nloss={best_loss:.6f}",
            xy=(best_epoch, best_loss),
            xytext=(best_epoch + 2, best_loss * 1.1),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=9,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("训练过程 Loss 曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_temporal_prediction(
    pred_series: np.ndarray,
    target_series: np.ndarray,
    indicator_name: str = "PH",
    station_name: str = "Station_1",
    save_path: Optional[str] = None,
):
    """
    绘制单指标的时间序列预测对比 (从 GAF 对角线提取)。

    Args:
        pred_series:   (T_out,) 预测序列
        target_series: (T_out,) 真实序列
        indicator_name: 指标名
        station_name:  站点名
        save_path:     保存路径
    """
    T = len(pred_series)
    times = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, target_series, "o-", label="Ground Truth",
            color="#4CAF50", linewidth=2, markersize=6)
    ax.plot(times, pred_series, "s--", label="Prediction",
            color="#FF5722", linewidth=2, markersize=6)
    ax.fill_between(times, target_series, pred_series, alpha=0.2, color="gray")

    ax.set_xlabel("Future Time Step")
    ax.set_ylabel(f"{indicator_name} (normalized)")
    ax.set_title(f"{station_name} — {indicator_name} 预测对比")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_per_station_comparison(
    per_station_metrics: dict,
    metric_name: str = "mse",
    save_path: Optional[str] = None,
):
    """
    绘制各站点指标对比柱状图。

    Args:
        per_station_metrics: {station_name: {metric: value, ...}}
        metric_name: 要绘制的指标名
        save_path:  保存路径
    """
    stations = list(per_station_metrics.keys())
    values = [per_station_metrics[s][metric_name] for s in stations]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(stations)))
    bars = ax.bar(stations, values, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_xlabel("Station")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f"各站点 {metric_name.upper()} 对比")
    ax.grid(True, alpha=0.3, axis="y")

    # 数值标注
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"{val:.4f}", ha="center", fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ============================================================================
# 测试
# ============================================================================
if __name__ == "__main__":
    print("测试可视化函数...")

    # 模拟 GAF 数据
    C, H, W = 6, 24, 24
    pred = np.random.randn(C, H, W) * 0.5
    target = np.random.randn(C, H, W) * 0.5

    ind_names = ["PH", "DO", "COND", "TURB", "NH3N", "COD"]
    plot_gaf_comparison(pred, target, indicator_names=ind_names,
                        save_path="results/test_gaf_comparison.png")

    # 模拟 alpha
    alpha = np.random.rand(5, 5, 3)
    plot_alpha_heatmap(alpha, save_path="results/test_alpha_heatmap.png")

    # 模拟 loss
    train_loss = np.linspace(0.5, 0.05, 50) + np.random.randn(50) * 0.02
    val_loss = np.linspace(0.5, 0.08, 50) + np.random.randn(50) * 0.02
    plot_loss_curve(train_loss.tolist(), val_loss.tolist(),
                    save_path="results/test_loss_curve.png")

    print("\n[OK] Visualization module tests passed")
    print("  图表已保存至 results/ 目录")
