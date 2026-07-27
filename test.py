"""
测试/预测脚本

功能:
- 加载训练好的最佳模型
- 对测试集进行预测
- 计算并展示各项指标（分站点、分指标）
- 生成可视化图表:
  - GAF 图像预测对比
  - TGGC α 权重热力图
  - 时序预测对比（从 GAF 逆变换）
  - 各站点指标对比柱状图
"""

import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Dict, Optional, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CHECKPOINT_DIR,
    RESULT_DIR,
    PROCESSED_DATA_DIR,
    CHECKPOINT_FILENAME,
    TOTAL_STATIONS,
    INDICATORS,
    INDICATOR_NAMES,
    STATION_NAMES,
    GAF_WINDOW_SIZE,
    GAF_IMAGE_SIZE,
    GAF_METHOD,
    T_IN,
    T_OUT,
    SIMVP_HID_S,
    SIMVP_HID_T,
    SIMVP_N_S,
    SIMVP_N_T,
    SIMVP_MODEL_TYPE,
    TGGC_ORDER,
    TGGC_LAYERS,
    TGGC_GCONV,
    TGGC_MULTI_LAYER,
    TGGC_MODES,
    FUSION_TYPE,
    BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
)
from model import WaterQualityPredictor
from data_preprocessing import WaterQualityDataset, create_dataloaders
from utils.metrics import (
    compute_all_metrics,
    compute_per_station_metrics,
    compute_per_indicator_metrics,
)
from utils.visualization import (
    plot_gaf_comparison,
    plot_alpha_heatmap,
    plot_loss_curve,
    plot_temporal_prediction,
    plot_per_station_comparison,
)
from modules.gaf import inverse_gaf


def load_best_model() -> WaterQualityPredictor:
    """加载训练好的最佳模型。"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

    if not os.path.exists(checkpoint_path):
        # 尝试其他可能的 checkpoint 文件
        candidates = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
        if candidates:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, candidates[0])
            print(f"使用 checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(
                f"未找到任何 checkpoint 文件。请先运行 train.py 进行训练。\n"
                f"查找目录: {CHECKPOINT_DIR}"
            )

    print(f"加载模型: {checkpoint_path}")
    model = WaterQualityPredictor(
        in_shape=(INDICATORS, GAF_IMAGE_SIZE, GAF_IMAGE_SIZE),
        hid_S=SIMVP_HID_S,
        hid_T=SIMVP_HID_T,
        N_S=SIMVP_N_S,
        N_T=SIMVP_N_T,
        T_in=T_IN,
        T_out=T_OUT,
        num_stations=TOTAL_STATIONS,
        model_type=SIMVP_MODEL_TYPE,
        tggc_order=TGGC_ORDER,
        tggc_layers=TGGC_LAYERS,
        tggc_gconv=TGGC_GCONV,
        tggc_multi_layer=TGGC_MULTI_LAYER,
        tggc_modes=TGGC_MODES,
        fusion_type=FUSION_TYPE,
    )
    model = model.to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    epoch = checkpoint.get("epoch", "unknown")
    best_score = checkpoint.get("best_score", "N/A")
    print(f"  训练 epoch: {epoch}")
    print(f"  最佳 score: {best_score}")

    return model


def load_test_data() -> DataLoader:
    """加载测试集。"""
    dataset_path = os.path.join(PROCESSED_DATA_DIR, "dataset.pt")
    data = torch.load(dataset_path, map_location="cpu")
    test_ds = WaterQualityDataset(data["test_X"], data["test_Y"])
    print(f"测试集: {len(test_ds)} 样本")

    loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )
    return loader, data


@torch.no_grad()
def run_prediction(
    model: WaterQualityPredictor,
    dataloader: DataLoader,
    num_batches: Optional[int] = None,
) -> Dict:
    """
    在测试集上运行预测。

    Returns:
        {
            "preds":  (N, T_out, N_stations, C, H, W),
            "targets": (N, T_out, N_stations, C, H, W),
            "alpha":  (N_stations, N_stations, K) — 最后的 α 权重
        }
    """
    model.eval()
    all_preds = []
    all_targets = []
    alpha = None

    pbar = tqdm(dataloader, desc="预测")
    for i, (batch_x, batch_y) in enumerate(pbar):
        if num_batches and i >= num_batches:
            break

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        output = model(batch_x, return_alpha=True)

        all_preds.append(output["pred"].cpu())
        all_targets.append(batch_y.cpu())

        if output.get("alpha") is not None:
            alpha = output["alpha"].cpu().numpy()

    preds = torch.cat(all_preds, dim=0)       # (N, T_out, 5, 6, H, W)
    targets = torch.cat(all_targets, dim=0)

    return {"preds": preds, "targets": targets, "alpha": alpha}


def extract_time_series_from_gaf(
    gaf_images: np.ndarray,
    method: str = "gasf",
) -> np.ndarray:
    """
    从 GAF 图像提取时间序列（用于可视化原始指标趋势）。

    GAF 图像 (..., C, H, W) → (..., C, H) 通过对角线提取。

    Args:
        gaf_images: (..., C, H, W) GAF 图像
        method:     GAF 方法
    Returns:
        (..., C, H) 近似恢复的时序值
    """
    shape = gaf_images.shape
    C, H = shape[-3], shape[-2]
    flat = gaf_images.reshape(-1, C, H, H)
    N = flat.shape[0]

    series = np.zeros((N, C, H), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            series[n, c] = inverse_gaf(flat[n, c], method=method)

    out_shape = list(shape[:-1])  # remove W dim
    return series.reshape(out_shape)


def generate_visualizations(
    preds: np.ndarray,
    targets: np.ndarray,
    alpha: Optional[np.ndarray],
):
    """
    生成所有可视化图表。
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    B, T_out, N, C, H, W = preds.shape

    print("\n生成可视化图表...")

    # ---- 1. GAF 图像对比 ----
    print("  1/5 GAF 图像对比...")
    for s in range(min(N, 3)):  # 最多展示 3 个站点
        plot_gaf_comparison(
            pred=preds[0, 0, s],        # 第 1 个样本, 第 1 帧
            target=targets[0, 0, s],
            indicator_names=INDICATOR_NAMES,
            station_name=STATION_NAMES[s],
            time_idx=0,
            save_path=os.path.join(RESULT_DIR, f"gaf_comparison_{STATION_NAMES[s]}.png"),
        )

    # ---- 2. α 权重热力图 ----
    if alpha is not None:
        print("  2/5 α 权重热力图...")
        plot_alpha_heatmap(
            alpha=alpha,
            station_names=STATION_NAMES,
            save_path=os.path.join(RESULT_DIR, "alpha_heatmap.png"),
        )

    # ---- 3. 时序预测对比 ----
    print("  3/5 时序预测对比...")
    pred_series = extract_time_series_from_gaf(preds.numpy(), method=GAF_METHOD)
    target_series = extract_time_series_from_gaf(targets.numpy(), method=GAF_METHOD)

    for s in range(min(N, 3)):
        for c in range(min(C, 3)):  # 每个站点展示 3 个指标
            plot_temporal_prediction(
                pred_series=pred_series[0, :, s, c],
                target_series=target_series[0, :, s, c],
                indicator_name=INDICATOR_NAMES[c],
                station_name=STATION_NAMES[s],
                save_path=os.path.join(
                    RESULT_DIR,
                    f"temporal_{STATION_NAMES[s]}_{INDICATOR_NAMES[c]}.png",
                ),
            )

    # ---- 4. 各站点指标对比 ----
    print("  4/5 各站点指标对比...")
    per_station = compute_per_station_metrics(
        torch.from_numpy(preds), torch.from_numpy(targets),
        station_names=STATION_NAMES,
    )
    plot_per_station_comparison(
        per_station, metric_name="mse",
        save_path=os.path.join(RESULT_DIR, "per_station_mse.png"),
    )

    # ---- 5. Loss 曲线 ----
    history_path = os.path.join(RESULT_DIR, "training_history.json")
    if os.path.exists(history_path):
        print("  5/5 Loss 曲线...")
        with open(history_path) as f:
            history = json.load(f)
        plot_loss_curve(
            history["train_losses"],
            history["val_losses"],
            save_path=os.path.join(RESULT_DIR, "loss_curve.png"),
        )

    print(f"可视化图表已保存至: {RESULT_DIR}/")


def print_metrics_report(
    preds: torch.Tensor,
    targets: torch.Tensor,
):
    """
    打印完整的指标报告。
    """
    print(f"\n{'='*60}")
    print("  测试指标报告")
    print(f"{'='*60}")

    # 整体指标
    overall = compute_all_metrics(preds, targets, prefix="")
    print("\n【 整体指标 】")
    for k, v in overall.items():
        print(f"  {k.upper():8s}: {v:.6f}")

    # 分站点
    print("\n【 分站点 MSE 】")
    per_station = compute_per_station_metrics(
        preds, targets, station_names=STATION_NAMES,
    )
    for name, metrics in per_station.items():
        print(f"  {name:12s}: MSE={metrics['mse']:.6f}, "
              f"MAE={metrics['mae']:.6f}, "
              f"SSIM={metrics.get('ssim', 0):.4f}")

    # 分指标
    print("\n【 分指标 MSE (所有站点平均) 】")
    flat_preds = preds.reshape(-1, preds.shape[-3], preds.shape[-2], preds.shape[-1])
    flat_targets = targets.reshape(-1, targets.shape[-3], targets.shape[-2], targets.shape[-1])
    per_indicator = compute_per_indicator_metrics(
        flat_preds, flat_targets, indicator_names=INDICATOR_NAMES,
    )
    for name, metrics in per_indicator.items():
        print(f"  {name:8s}: MSE={metrics['mse']:.6f}, "
              f"MAE={metrics['mae']:.6f}")


def main():
    """测试主流程。"""
    print("=" * 60)
    print("  水质预测系统 — 测试评估")
    print("=" * 60)

    # ---- 加载模型 ----
    model = load_best_model()

    # ---- 加载数据 ----
    loader, raw_data = load_test_data()

    # ---- 预测 ----
    result = run_prediction(model, loader)
    preds = result["preds"]
    targets = result["targets"]
    alpha = result["alpha"]

    print(f"\n预测完成: {preds.shape}")

    # ---- 指标报告 ----
    print_metrics_report(preds, targets)

    # ---- 可视化 ----
    generate_visualizations(preds, targets, alpha)

    # ---- 保存结果 ----
    torch.save({
        "preds": preds,
        "targets": targets,
        "alpha": alpha,
    }, os.path.join(RESULT_DIR, "predictions.pt"))
    print(f"\n预测结果已保存: {os.path.join(RESULT_DIR, 'predictions.pt')}")

    print("\n[OK] Testing completed")


if __name__ == "__main__":
    main()
