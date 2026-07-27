"""
PCHIP 分段三次埃尔米特插值 — 水质数据预处理

功能:
1. 读取单个 CSV 文件（含时间列 + 5站点×6指标列）
2. 检测并处理缺失值
3. 使用 PCHIP 插值填充缺失和加密时间分辨率
4. 输出插值后的 CSV 文件

PCHIP (Piecewise Cubic Hermite Interpolating Polynomial):
- 保单调性的三次插值
- 不会像样条插值那样在数据稀疏区域产生振荡
- 适合水质数据这类物理量
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from typing import Optional, Tuple
from tqdm import tqdm

# 添加项目根路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RAW_DATA_PATH,
    INTERPOLATED_DATA_PATH,
    INTERP_TIME_COLUMN,
    INTERP_FREQUENCY,
    TOTAL_STATIONS,
    INDICATORS,
    INDICATOR_NAMES,
    STATION_NAMES,
    get_csv_columns,
)


def generate_sample_data(
    output_path: str,
    n_days: int = 365,
    missing_ratio: float = 0.05,
    seed: int = 42,
):
    """
    生成模拟水质数据（用于代码测试）。

    数据特征:
    - 周期性（日变化 + 季节变化）
    - 指标间存在相关性
    - 添加随机噪声和缺失值

    Args:
        output_path:   输出 CSV 路径
        n_days:        时间点数（如 365 天）
        missing_ratio: 随机缺失比例
        seed:          随机种子
    """
    np.random.seed(seed)

    # 生成时间列
    time_index = pd.date_range(start="2024-01-01", periods=n_days, freq="D")

    data = {}
    data[INTERP_TIME_COLUMN] = time_index.strftime("%Y-%m-%d %H:%M:%S")

    # 各指标的基础值 + 周期模式
    base_params = {
        "PH":    (7.5, 0.5, 0.3),     # (均值, 年振幅, 日振幅)
        "DO":    (8.0, 2.0, 0.5),
        "COND":  (400, 100, 20),
        "TURB":  (10,  8,   3),
        "NH3N":  (0.5, 0.3, 0.1),
        "COD":   (4.0, 2.0, 0.5),
    }

    for station_idx in range(1, TOTAL_STATIONS + 1):
        s_offset = np.random.randn() * 0.2  # 站点偏移
        for ind_name in INDICATOR_NAMES:
            col_name = f"Station{station_idx}_{ind_name}"
            base, annual_amp, daily_amp = base_params[ind_name]

            # 年周期
            t = np.arange(n_days)
            annual = annual_amp * np.sin(2 * np.pi * t / 365 + s_offset)
            # 日周期（简化）
            daily = daily_amp * np.sin(2 * np.pi * t / 1 + s_offset)

            values = base + annual + daily + np.random.randn(n_days) * daily_amp * 0.5

            # 注入缺失值
            mask = np.random.random(n_days) < missing_ratio
            values[mask] = np.nan

            data[col_name] = values

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"模拟数据已生成: {output_path}")
    print(f"  时间点数: {n_days}")
    print(f"  站点数: {TOTAL_STATIONS}")
    print(f"  指标数: {INDICATORS}")
    print(f"  缺失比例: {missing_ratio:.1%}")
    return df


def load_data(path: str) -> Tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    """
    加载水质 CSV 文件。

    Returns:
        df:         原始 DataFrame
        time_index:  时间索引
        data_array:  (T, N_stations, C) numpy 数组
    """
    df = pd.read_csv(path)
    time_col = df.columns[0]
    time_index = pd.to_datetime(df[time_col])

    data_cols = get_csv_columns()
    missing_cols = [c for c in data_cols if c not in df.columns]

    if missing_cols:
        print(f"警告: CSV 中缺少 {len(missing_cols)} 列，将自动填充为 NaN")
        for c in missing_cols:
            df[c] = np.nan

    # 提取数据矩阵 (T, N, C)
    T = len(df)
    data_array = np.full((T, TOTAL_STATIONS, INDICATORS), np.nan, dtype=np.float64)

    for s in range(TOTAL_STATIONS):
        for c, ind in enumerate(INDICATOR_NAMES):
            col = f"Station{s+1}_{ind}"
            if col in df.columns:
                data_array[:, s, c] = df[col].values.astype(np.float64)

    return df, time_index, data_array


def pchip_interpolate(
    data_array: np.ndarray,
    time_index: pd.DatetimeIndex,
    freq: str = "1H",
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    使用 PCHIP 对每个站点×指标的时间序列独立进行插值。

    PCHIP 特点:
    - 分段三次多项式插值
    - 保持数据的单调性（不会出现过冲/下冲）
    - 一阶导数连续

    Args:
        data_array: (T_original, N, C) 含 NaN 的原始数据
        time_index: 原始时间索引
        freq:       目标频率 (如 "1H", "30T", "D")

    Returns:
        new_time: 插值后的时间索引
        interp_data: (T_new, N, C) 插值后数据
    """
    # 将时间转为数值（相对于起点的秒数）
    time_numeric = (time_index - time_index[0]).total_seconds().values

    # 生成新的均匀时间网格
    new_time_index = pd.date_range(
        start=time_index.min(),
        end=time_index.max(),
        freq=freq,
    )
    new_time_numeric = (new_time_index - time_index[0]).total_seconds().values

    T_new = len(new_time_index)
    interp_data = np.zeros((T_new, TOTAL_STATIONS, INDICATORS), dtype=np.float64)

    nan_count_before = np.isnan(data_array).sum()
    filled_count = 0

    for s in range(TOTAL_STATIONS):
        for c in range(INDICATORS):
            series = data_array[:, s, c].copy()

            # 记录有效值位置
            valid = ~np.isnan(series)

            if valid.sum() < 4:
                # 有效数据太少 → 线性插值
                print(f"警告: Station{s+1}_{INDICATOR_NAMES[c]} 有效点仅 {valid.sum()} 个，"
                      f"使用线性插值")
                interp_data[:, s, c] = np.interp(
                    new_time_numeric, time_numeric[valid], series[valid]
                )
            else:
                # PCHIP 插值
                pchip = PchipInterpolator(time_numeric[valid], series[valid])
                interp_data[:, s, c] = pchip(new_time_numeric)

            filled_count += int(np.isnan(series).sum())

    print(f"\nPCHIP 插值完成:")
    print(f"  原始点数: {len(time_index)} (缺失 {nan_count_before} 个)")
    print(f"  插值后点: {T_new}")
    print(f"  频率: {freq}")
    print(f"  填充缺失值: {filled_count}")

    return new_time_index, interp_data


def save_interpolated(
    time_index: pd.DatetimeIndex,
    data_array: np.ndarray,
    output_path: str,
):
    """
    保存插值后的数据为 CSV 文件。

    Args:
        time_index: 插值后的时间索引
        data_array: (T, N, C) 插值后数据
        output_path: 输出路径
    """
    T, N, C = data_array.shape
    data_dict = {INTERP_TIME_COLUMN: time_index.strftime("%Y-%m-%d %H:%M:%S")}

    for s in range(N):
        for c, ind in enumerate(INDICATOR_NAMES):
            data_dict[f"Station{s+1}_{ind}"] = data_array[:, s, c]

    df = pd.DataFrame(data_dict)
    df.to_csv(output_path, index=False)
    print(f"插值数据已保存: {output_path}")
    print(f"  Shape: {T} × {N*C+1} ({N*C} 数据列 + 1 时间列)")


def main():
    """主流程: 加载 → 插值 → 保存。"""
    print("=" * 60)
    print("  水质数据 PCHIP 插值")
    print("=" * 60)

    # ---- 检查原始数据是否存在 ----
    if not os.path.exists(RAW_DATA_PATH):
        print(f"\n原始数据不存在: {RAW_DATA_PATH}")
        print("生成模拟数据用于测试...")
        generate_sample_data(RAW_DATA_PATH, n_days=180, missing_ratio=0.05)

    # ---- 加载 ----
    df, time_index, data_array = load_data(RAW_DATA_PATH)
    print(f"\n原始数据: {data_array.shape}")
    print(f"  时间范围: {time_index.min()} ~ {time_index.max()}")
    print(f"  缺失率: {np.isnan(data_array).mean():.2%}")

    # ---- PCHIP 插值 ----
    new_time, interp_data = pchip_interpolate(
        data_array, time_index, freq=INTERP_FREQUENCY
    )

    # ---- 保存 ----
    save_interpolated(new_time, interp_data, INTERPOLATED_DATA_PATH)

    print("\n[OK] PCHIP interpolation completed")


if __name__ == "__main__":
    main()
