"""
全局配置文件 —— 水质预测系统 (SimVPv2 + TGGC)
所有超参数、路径、模型配置均在此文件中集中管理。
"""

import os

def _get_device():
    """延迟导入 torch，避免非 ML 模块导入失败。"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

# ============================================================================
# 路径配置
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_water_quality.csv")       # 原始 CSV
INTERPOLATED_DATA_PATH = os.path.join(DATA_DIR, "interpolated.csv")   # PCHIP 插值后
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")              # GAF 图像 + 样本存放目录

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================================
# 数据配置
# ============================================================================
TOTAL_STATIONS = 5               # 站点数量
INDICATORS = 6                   # 每个站点的水质指标数

INDICATOR_NAMES = [
    "PH",                        # pH 值
    "DO",                        # 溶解氧 (Dissolved Oxygen)
    "COND",                      # 电导率 (Conductivity)
    "TURB",                      # 浑浊度 (Turbidity)
    "NH3N",                      # 氨氮 (Ammonia Nitrogen)
    "COD",                       # 耗氧量 (Chemical Oxygen Demand)
]

STATION_NAMES = [f"Station_{i+1}" for i in range(TOTAL_STATIONS)]

# CSV 列名生成（假设格式: Station1_PH, Station1_DO, ..., Station5_COD）
def get_csv_columns():
    """返回 CSV 中所有数据列的列表（不含时间列）"""
    cols = []
    for s in range(1, TOTAL_STATIONS + 1):
        for ind in INDICATOR_NAMES:
            cols.append(f"Station{s}_{ind}")
    return cols

# ============================================================================
# PCHIP 插值配置
# ============================================================================
INTERP_TIME_COLUMN = "time"          # CSV 中的时间列名
INTERP_FREQUENCY = "1H"              # 插值目标频率: "1H"(每小时), "30T"(每30分钟), "D"(每天)
INTERP_METHOD = "PCHIP"              # 插值方法: PCHIP / linear / cubic

# ============================================================================
# GAF 变换配置
# ============================================================================
GAF_WINDOW_SIZE = 24                 # 用于构建 GAF 图像的时间窗口 L
GAF_METHOD = "gasf"                  # 默认方法: "gasf" (GASF) / "gadf" (GADF)
GAF_IMAGE_SIZE = GAF_WINDOW_SIZE     # GAF 图像尺寸 H = W = L

# ============================================================================
# 时序预测配置
# ============================================================================
T_IN = 12                            # 输入时间步（历史帧数）
T_OUT = 6                            # 输出时间步（预测帧数）
TOTAL_FRAMES = T_IN + T_OUT          # 每个样本总帧数: 18

# ============================================================================
# 数据集划分
# ============================================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# ============================================================================
# 模型超参数 (与原始 SimVPv2 + TGGC 论文一致)
# ============================================================================
# --- SimVPv2 (基于原作者 configs/SimVP+IncepU.py) ---
SIMVP_IN_CHANNELS = INDICATORS       # 6 — GAF 图像通道数
SIMVP_HID_S = 64                     # Encoder/Decoder 隐藏通道 (原文: hid_S=64)
SIMVP_HID_T = 256                    # Translator 隐藏通道 (缩减版, 原文: hid_T=512)
SIMVP_N_S = 4                        # Encoder/Decoder 层数 (原文: N_S=4)
SIMVP_N_T = 8                        # Translator Inception 层数 (原文: N_T=8)
SIMVP_MODEL_TYPE = "IncepU"          # Translator 类型: "IncepU" / "gSTA"

# --- TGGC (基于原作者 main.py 参数) ---
TGGC_ORDER = 4                       # Gegenbauer 多项式阶数 (原文: order=4)
TGGC_LAYERS = 2                      # StockBlock 堆叠层数 (原文: layers=2)
TGGC_GCONV = "gegen"                 # 图卷积类型: "gegen" / "cheby" / "jacobi"
TGGC_MULTI_LAYER = 5                 # 时序多尺度因子 (原文: multi_layer=5)
TGGC_MODES = 5                       # FFT 频率分量数 (原文: modes=5)
TGGC_COE_A = 1.2                     # Gegenbauer 参数 α
TGGC_COE_B = 1.0                     # Jacobi 参数 β

# --- Fusion ---
FUSION_TYPE = "residual"             # 融合方式: "residual" / "gate"

# ============================================================================
# 训练配置
# ============================================================================
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_T_0 = 20                          # CosineAnnealingWarmRestarts 重启周期
LR_T_MULT = 2                        # 每次重启周期倍增因子
LR_ETA_MIN = 1e-6                    # 最小学习率
GRAD_CLIP = 1.0                      # 梯度裁剪阈值

# Early Stopping
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-4

# ============================================================================
# 设备和精度
# ============================================================================
DEVICE = _get_device()
USE_AMP = False                      # 混合精度训练 (仅 CUDA 有效)
NUM_WORKERS = 4                      # DataLoader 进程数

# ============================================================================
# 日志和保存
# ============================================================================
LOG_INTERVAL = 10                    # 每 N 个 batch 记录一次日志
VAL_INTERVAL = 1                     # 每 N 个 epoch 验证一次
SAVE_INTERVAL = 10                   # 每 N 个 epoch 保存一次 checkpoint
CHECKPOINT_FILENAME = "best_model.pth"

# ============================================================================
# 打印配置摘要
# ============================================================================
def print_config():
    """打印当前关键配置"""
    print("=" * 60)
    print("  水质预测系统 (SimVPv2 + TGGC) 配置摘要")
    print("=" * 60)
    print(f"  设备: {DEVICE}")
    print(f"  站点数: {TOTAL_STATIONS} | 指标数: {INDICATORS}")
    print(f"  GAF 窗口: {GAF_WINDOW_SIZE} ({GAF_WINDOW_SIZE}×{GAF_WINDOW_SIZE})")
    print(f"  时序: T_in={T_IN}, T_out={T_OUT}")
    print(f"  SimVPv2: hid_S={SIMVP_HID_S}, hid_T={SIMVP_HID_T}, "
          f"N_S={SIMVP_N_S}, N_T={SIMVP_N_T}, type={SIMVP_MODEL_TYPE}")
    print(f"  TGGC: order={TGGC_ORDER}, layers={TGGC_LAYERS}, "
          f"gconv={TGGC_GCONV}")
    print(f"  训练: lr={LEARNING_RATE}, batch={BATCH_SIZE}, "
          f"epochs={NUM_EPOCHS}")
    print(f"  数据: {RAW_DATA_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
