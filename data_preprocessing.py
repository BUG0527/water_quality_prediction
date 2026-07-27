"""
数据预处理：时序数据 → GAF 图像 → 监督学习样本

流程:
1. 读取插值后的 CSV 数据
2. 按站点和指标独立进行标准化 (Z-score → MinMax[-1,1])
3. 滑动窗口 GAF 变换: 长度为 L 的窗口 → (L, L) GAF 图像
4. 构建监督样本: X(T_in) → Y(T_out)
5. 划分 train/val/test 并保存为 .pt 文件

GAF 变换说明:
- 每帧对应一个时间点的"快照"
- 该帧由 6 个通道的 GAF 图像堆叠而成（类似 RGB）
- 每个通道对应一个水质指标的 GAF 图像
- GAF 图像使用该时间点之前 L 个时间点的值构建
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional, Dict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INTERPOLATED_DATA_PATH,
    PROCESSED_DATA_DIR,
    INTERP_TIME_COLUMN,
    TOTAL_STATIONS,
    INDICATORS,
    INDICATOR_NAMES,
    STATION_NAMES,
    GAF_WINDOW_SIZE,
    GAF_METHOD,
    GAF_IMAGE_SIZE,
    T_IN,
    T_OUT,
    TOTAL_FRAMES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    NUM_WORKERS,
)
from modules.gaf import gaf_transform, gaf_batch_transform


# ============================================================================
# 数据集类
# ============================================================================

class WaterQualityDataset(Dataset):
    """
    水质 GAF 图像数据集。

    每个样本:
      - X: (T_in, 5, 6, H, W) 历史 GAF 序列
      - Y: (T_out, 5, 6, H, W) 未来 GAF 序列

    注意: 数据存储在 float32，模型训练时按需迁移到设备。
    """

    def __init__(
        self,
        samples_x: torch.Tensor,
        samples_y: torch.Tensor,
    ):
        """
        Args:
            samples_x: (N, T_in, 5, 6, H, W)
            samples_y: (N, T_out, 5, 6, H, W)
        """
        self.X = samples_x  # 已在内存中
        self.Y = samples_y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


# ============================================================================
# 标准化
# ============================================================================

def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    对每个站点×指标独立进行 Z-score 归一化，然后 MinMax 缩放到 [-1, 1]。

    Args:
        data: (T, N, C) 原始时间序列

    Returns:
        data_norm:  (T, N, C) 归一化后数据
        stats:      归一化参数 (用于逆变换)
    """
    T, N, C = data.shape
    data_norm = np.zeros_like(data, dtype=np.float64)
    stats = {}

    for s in range(N):
        for c in range(C):
            series = data[:, s, c]

            # Z-score
            mean = np.nanmean(series)
            std = np.nanstd(series)
            if std < 1e-8:
                std = 1.0
            zscored = (series - mean) / std

            # MinMax → [-1, 1]
            d_min = zscored.min()
            d_max = zscored.max()
            if d_max - d_min < 1e-8:
                data_norm[:, s, c] = 0.0
            else:
                data_norm[:, s, c] = 2.0 * (zscored - d_min) / (d_max - d_min) - 1.0

            stats[f"S{s+1}_{INDICATOR_NAMES[c]}"] = {
                "mean": mean,
                "std": std,
                "min": d_min,
                "max": d_max,
            }

    return data_norm, stats


def inverse_normalize(data_norm: np.ndarray, stats: Dict) -> np.ndarray:
    """
    逆标准化: [-1,1] → 原始值域。

    Args:
        data_norm: (..., N, C) 归一化数据
        stats:     normalize_data 返回的参数字典

    Returns:
        还原后的数据
    """
    data_norm = np.asarray(data_norm)
    shape = data_norm.shape
    data = np.zeros_like(data_norm, dtype=np.float64)

    for s in range(shape[-2]):
        for c in range(shape[-1]):
            key = f"S{s+1}_{INDICATOR_NAMES[c]}"
            s_dict = stats[key]
            zscored = (data_norm[..., s, c] + 1.0) / 2.0 * (s_dict["max"] - s_dict["min"]) + s_dict["min"]
            data[..., s, c] = zscored * s_dict["std"] + s_dict["mean"]

    return data


# ============================================================================
# 滑动窗口 GAF 变换
# ============================================================================

def build_gaf_sequence(
    data_norm: np.ndarray,
    window_size: int,
    method: str = "gasf",
) -> np.ndarray:
    """
    将 (T, N, C) 的时间序列转换为 (T - window_size + 1, N, C, H, W) 的 GAF 序列。

    对于每个时刻 t ∈ [window_size-1, T-1]:
      取 data[t-window_size+1 : t+1] → GAF 变换 → (C, window_size, window_size)

    Args:
        data_norm:   (T, N, C) 标准化后的时序
        window_size: GAF 窗口大小 L
        method:      GAF 方法

    Returns:
        gaf_seq: (T_gaf, N, C, L, L) GAF 图像序列
    """
    T, N, C = data_norm.shape
    T_gaf = T - window_size + 1

    if T_gaf <= 0:
        raise ValueError(f"时间步 T={T} 小于窗口大小 {window_size}，"
                         f"无法构建 GAF 序列。需要至少 {window_size} 个时间点。")

    gaf_seq = np.zeros((T_gaf, N, C, window_size, window_size), dtype=np.float32)

    print(f"构建 GAF 序列: 窗口={window_size}, 总 GAF 帧数={T_gaf}")
    for t in tqdm(range(T_gaf), desc="GAF 变换"):
        window = data_norm[t:t + window_size]  # (L, N, C)
        # 对每个站点
        for s in range(N):
            # 对每个指标独立做 GAF
            for c in range(C):
                gaf_seq[t, s, c] = gaf_transform(
                    window[:, s, c], method=method
                )

    return gaf_seq


def build_samples(
    gaf_seq: np.ndarray,
    T_in: int,
    T_out: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 GAF 序列构建监督学习样本。

    Args:
        gaf_seq: (T_gaf, N, C, H, W)
        T_in:    输入帧数
        T_out:   输出帧数

    Returns:
        X: (N_samples, T_in, N, C, H, W)
        Y: (N_samples, T_out, N, C, H, W)
    """
    T_gaf = gaf_seq.shape[0]
    total_frames = T_in + T_out
    N_samples = T_gaf - total_frames + 1

    if N_samples <= 0:
        raise ValueError(f"GAF 序列长度 {T_gaf} 不足，"
                         f"无法构建 {total_frames} 帧的样本。"
                         f"请提供更长的时间序列或减小 GAF 窗口。")

    X_list = []
    Y_list = []

    for i in range(N_samples):
        X_list.append(gaf_seq[i:i + T_in])          # (T_in, N, C, H, W)
        Y_list.append(gaf_seq[i + T_in:i + total_frames])  # (T_out, N, C, H, W)

    X = np.stack(X_list, axis=0)  # (N_samples, T_in, N, C, H, W)
    Y = np.stack(Y_list, axis=0)  # (N_samples, T_out, N, C, H, W)

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    return X, Y


# ============================================================================
# 数据划分
# ============================================================================

def split_dataset(
    X: torch.Tensor,
    Y: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[WaterQualityDataset, WaterQualityDataset, WaterQualityDataset]:
    """
    按时间顺序划分 train/val/test（不打乱，保持时序依赖）。

    注意: 使用时间顺序划分而非随机划分，以保持时序完整性。
    """
    N = len(X)
    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))

    indices = np.arange(N)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_ds = WaterQualityDataset(X[train_idx], Y[train_idx])
    val_ds = WaterQualityDataset(X[val_idx], Y[val_idx])
    test_ds = WaterQualityDataset(X[test_idx], Y[test_idx])

    print(f"\n数据划分 (时间顺序):")
    print(f"  Train: {len(train_ds)} 样本")
    print(f"  Val:   {len(val_ds)} 样本")
    print(f"  Test:  {len(test_ds)} 样本")

    return train_ds, val_ds, test_ds


# ============================================================================
# DataLoader 工厂
# ============================================================================

def create_dataloaders(
    train_ds: WaterQualityDataset,
    val_ds: WaterQualityDataset,
    test_ds: WaterQualityDataset,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Dict[str, DataLoader]:
    """
    创建 PyTorch DataLoader。

    Returns:
        {"train": dl, "val": dl, "test": dl}
    """
    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        ),
    }
    return loaders


# ============================================================================
# 主流程
# ============================================================================

def main():
    """数据预处理主流程。"""
    print("=" * 60)
    print("  数据预处理: 时序 → GAF → 样本")
    print("=" * 60)

    # ---- 1. 加载插值数据 ----
    if not os.path.exists(INTERPOLATED_DATA_PATH):
        print(f"错误: 插值数据不存在: {INTERPOLATED_DATA_PATH}")
        print("请先运行 Data_Interpolation.py")
        return

    print(f"\n加载插值数据: {INTERPOLATED_DATA_PATH}")
    df = pd.read_csv(INTERPOLATED_DATA_PATH)
    time_col = df.columns[0]
    time_index = pd.to_datetime(df[time_col])
    T = len(df)

    # 提取数据矩阵 (T, 5, 6)
    data = np.zeros((T, TOTAL_STATIONS, INDICATORS), dtype=np.float64)
    for s in range(TOTAL_STATIONS):
        for c, ind in enumerate(INDICATOR_NAMES):
            col = f"Station{s+1}_{ind}"
            data[:, s, c] = df[col].values.astype(np.float64)

    print(f"  数据形状: {data.shape}")
    print(f"  时间范围: {time_index.min()} ~ {time_index.max()}")

    # ---- 2. 标准化 ----
    data_norm, stats = normalize_data(data)
    print(f"\n标准化完成, 值域: [{data_norm.min():.3f}, {data_norm.max():.3f}]")

    # 保存归一化参数
    import json
    stats_path = os.path.join(PROCESSED_DATA_DIR, "norm_stats.json")
    def convert_stats(obj):
        result = {}
        for k, v in obj.items():
            result[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                         for kk, vv in v.items()}
        return result
    with open(stats_path, "w") as f:
        json.dump(convert_stats(stats), f, indent=2)
    print(f"归一化参数已保存: {stats_path}")

    # ---- 3. GAF 变换 ----
    gaf_seq = build_gaf_sequence(
        data_norm,
        window_size=GAF_WINDOW_SIZE,
        method=GAF_METHOD,
    )
    print(f"GAF 序列: {gaf_seq.shape}")
    print(f"  GAF 值域: [{gaf_seq.min():.3f}, {gaf_seq.max():.3f}]")
    print(f"  内存占用: {gaf_seq.nbytes / 1024 / 1024:.1f} MB")

    # ---- 4. 构建样本 ----
    X, Y = build_samples(gaf_seq, T_IN, T_OUT)
    print(f"\n监督样本:")
    print(f"  X: {list(X.shape)}  (N, T_in, N_stations, C, H, W)")
    print(f"  Y: {list(Y.shape)}  (N, T_out, N_stations, C, H, W)")
    print(f"  总内存: {(X.nbytes + Y.nbytes) / 1024 / 1024:.1f} MB")

    # ---- 5. 数据划分 ----
    train_ds, val_ds, test_ds = split_dataset(
        X, Y, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    # ---- 6. 保存 ----
    torch.save({
        "train_X": train_ds.X, "train_Y": train_ds.Y,
        "val_X": val_ds.X, "val_Y": val_ds.Y,
        "test_X": test_ds.X, "test_Y": test_ds.Y,
        "norm_stats": convert_stats(stats),
    }, os.path.join(PROCESSED_DATA_DIR, "dataset.pt"))
    print(f"\n数据集已保存: {os.path.join(PROCESSED_DATA_DIR, 'dataset.pt')}")

    # ---- 7. 创建 DataLoader 并测试 ----
    loaders = create_dataloaders(train_ds, val_ds, test_ds)
    for split, loader in loaders.items():
        batch_x, batch_y = next(iter(loader))
        print(f"\n{split} DataLoader:")
        print(f"  批大小: {batch_x.shape}")

    print("\n[OK] Data preprocessing completed")


if __name__ == "__main__":
    main()
