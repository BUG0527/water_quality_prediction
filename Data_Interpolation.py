"""
PCHIP 分段三次埃尔米特插值 — 水质数据预处理

功能:
1. 读取长格式原始 CSV（中文列名，每个站点一行）
2. 清洗非数值（如 "<0.02" 低于检出限）
3. 转换为下游模型所需的宽格式（每行=时间戳，每列=站点×指标）
4. 使用 PCHIP 插值填充缺失和加密时间分辨率
5. 输出插值后的 CSV 文件

PCHIP (Piecewise Cubic Hermite Interpolating Polynomial):
- 保单调性的三次插值
- 不会像样条插值那样在数据稀疏区域产生振荡
- 适合水质数据这类物理量
"""

import os
import sys
import re
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
    RAW_COL_TIME,
    RAW_COL_STATION,
    RAW_INDICATOR_MAP,
    DETECTION_LIMIT_REPLACE,
    get_csv_columns,
)


def _clean_value(val):
    """
    清洗单个数值：将非数值字符串（如 "<0.02"）转换为浮点数。

    Args:
        val: 原始单元格值

    Returns:
        float 或 np.nan
    """
    if isinstance(val, (int, float, np.floating, np.integer)):
        if np.isnan(float(val)):
            return np.nan
        return float(val)

    val_str = str(val).strip()
    if not val_str:
        return np.nan

    # 先检查检出限替换表
    if val_str in DETECTION_LIMIT_REPLACE:
        return DETECTION_LIMIT_REPLACE[val_str]

    # 尝试直接转换
    try:
        return float(val_str)
    except ValueError:
        # 尝试提取数字（处理 "<0.02" 这类不在表中的变体）
        match = re.search(r'[\d.]+', val_str)
        if match:
            return float(match.group(0))
        return np.nan


def _extract_station_index(name: str) -> int:
    """
    从站点名称提取编号（1-based）。

    示例:
        "白洋湾金墅水源地1" → 1
        "白洋湾金墅水源地5" → 5

    Args:
        name: 站点名称字符串

    Returns:
        站点编号 (1 ~ TOTAL_STATIONS)
    """
    match = re.search(r'(\d+)$', name.strip())
    if match:
        return int(match.group(1))
    raise ValueError(f"无法从站点名称中提取编号: {name}")


def load_data(path: str) -> Tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    """
    加载长格式水质 CSV 文件，清洗并转换为宽格式。

    输入格式（长格式）:
        采样日期, 点位名称, 采样经度, 采样纬度, pH值, 溶解氧, 电导率, 浑浊度, 氨氮, 耗氧量
        2021/6/21, 白洋湾金墅水源地1, 120.3813, 31.3749, 8.1, 6.85, 452, 12, 0.06, 2.87

    输出格式（宽格式）:
        time, Station1_PH, Station1_DO, ..., Station5_COD
        2021-06-21 00:00:00, 8.1, 6.85, ..., 2.79

    Returns:
        df_wide:     宽格式 DataFrame（含 time 列和所有 Station{N}_{Indicator} 列）
        time_index:   时间索引
        data_array:  (T, N_stations, C) numpy 数组
    """
    df = pd.read_csv(path, encoding="utf-8")
    print(f"原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"  列名: {list(df.columns)}")

    # --- 解析日期 ---
    df[RAW_COL_TIME] = pd.to_datetime(df[RAW_COL_TIME])
    print(f"  日期范围: {df[RAW_COL_TIME].min().date()} ~ {df[RAW_COL_TIME].max().date()}")
    print(f"  唯一日期: {df[RAW_COL_TIME].nunique()}")
    print(f"  唯一站点: {df[RAW_COL_STATION].nunique()}")

    # --- 清洗每个指标列的数值 ---
    raw_ind_cols = list(RAW_INDICATOR_MAP.keys())
    nan_before = 0
    for col in raw_ind_cols:
        if col not in df.columns:
            print(f"  警告: 列 '{col}' 不存在于 CSV 中，跳过")
            continue
        # 统计清洗前非数值
        nan_before += df[col].apply(
            lambda x: not isinstance(x, (int, float, np.floating, np.integer))
        ).sum()
        df[col] = df[col].apply(_clean_value).astype(np.float64)

    nan_after = df[raw_ind_cols].isna().sum().sum()
    if nan_before > 0:
        print(f"  清洗非数值: {nan_before} 个 → 替换为数值或 NaN ({int(nan_after)} 个 NaN)")

    # --- 提取站点编号 ---
    df["_station_idx"] = df[RAW_COL_STATION].apply(_extract_station_index)

    # --- 长格式 → 宽格式 ---
    # 每条记录: (日期, 站点编号) → {PH: v, DO: v, ...}
    # 先构建 pivot-friendly 的 DataFrame
    rows = []
    for _, row in df.iterrows():
        date = row[RAW_COL_TIME]
        s_idx = row["_station_idx"]
        for cn_name, en_name in RAW_INDICATOR_MAP.items():
            col_name = f"Station{s_idx}_{en_name}"
            rows.append({
                RAW_COL_TIME: date,
                "column": col_name,
                "value": row[cn_name],
            })

    df_pivot = pd.DataFrame(rows)
    df_wide = df_pivot.pivot_table(
        index=RAW_COL_TIME,
        columns="column",
        values="value",
        aggfunc="first",  # 每个 (date, column) 组合应唯一
    ).reset_index()

    # 重命名时间列为输出格式
    df_wide.rename(columns={RAW_COL_TIME: INTERP_TIME_COLUMN}, inplace=True)

    # 确保所有预期列都存在
    expected_cols = get_csv_columns()
    missing_cols = [c for c in expected_cols if c not in df_wide.columns]
    if missing_cols:
        print(f"  警告: 缺少 {len(missing_cols)} 列，填充 NaN")
        for c in missing_cols:
            df_wide[c] = np.nan

    # 按日期排序
    df_wide.sort_values(INTERP_TIME_COLUMN, inplace=True)
    df_wide.reset_index(drop=True, inplace=True)

    time_index = pd.to_datetime(df_wide[INTERP_TIME_COLUMN])
    T = len(df_wide)

    # --- 构建 (T, N, C) 数组 ---
    data_array = np.full((T, TOTAL_STATIONS, INDICATORS), np.nan, dtype=np.float64)
    for s in range(TOTAL_STATIONS):
        for c, ind in enumerate(INDICATOR_NAMES):
            col = f"Station{s+1}_{ind}"
            if col in df_wide.columns:
                data_array[:, s, c] = df_wide[col].values.astype(np.float64)

    print(f"\n宽格式: {df_wide.shape[0]} 时间点 × {df_wide.shape[1]-1} 指标列")
    print(f"  缺失率: {np.isnan(data_array).mean():.2%}")

    return df_wide, time_index, data_array


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
    # 将时间转为数值（相对于起点的秒数，使用 int64 纳秒表示）
    time_index = pd.DatetimeIndex(time_index)
    t0 = time_index.asi8[0]
    time_numeric = (time_index.asi8 - t0).astype(np.float64) / 1e9

    # 生成新的均匀时间网格
    new_time_index = pd.date_range(
        start=time_index.min(),
        end=time_index.max(),
        freq=freq,
    )
    new_time_numeric = (new_time_index.asi8 - t0).astype(np.float64) / 1e9

    T_new = len(new_time_index)
    interp_data = np.zeros((T_new, TOTAL_STATIONS, INDICATORS), dtype=np.float64)

    nan_count_before = int(np.isnan(data_array).sum())
    filled_count = 0

    for s in range(TOTAL_STATIONS):
        for c in range(INDICATORS):
            series = data_array[:, s, c].copy()

            # 记录有效值位置
            valid = ~np.isnan(series)

            if valid.sum() < 4:
                # 有效数据太少 → 降级为线性插值
                print(f"  警告: Station{s+1}_{INDICATOR_NAMES[c]} 有效点仅 {valid.sum()} 个，"
                      f"使用线性插值")
                if valid.sum() >= 2:
                    interp_data[:, s, c] = np.interp(
                        new_time_numeric, time_numeric[valid], series[valid]
                    )
                else:
                    interp_data[:, s, c] = np.nan
            else:
                # PCHIP 插值
                pchip = PchipInterpolator(time_numeric[valid], series[valid])
                interp_data[:, s, c] = pchip(new_time_numeric)

            filled_count += int((~valid).sum())

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
        print(f"\n[错误] 原始数据文件不存在: {RAW_DATA_PATH}")
        print("请确认数据文件路径是否正确。")
        sys.exit(1)

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
