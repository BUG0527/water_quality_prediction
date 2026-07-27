"""
一键运行入口 —— 水质预测系统

用法:
  python main.py --mode all          # 完整流程: 插值→预处理→训练→测试
  python main.py --mode interpolate  # 仅数据插值
  python main.py --mode preprocess   # 仅数据预处理
  python main.py --mode train        # 仅训练
  python main.py --mode test         # 仅测试
  python main.py --mode sample       # 生成模拟数据
  python main.py --mode info         # 打印配置信息

可选参数:
  --epochs N          覆盖训练 epoch 数
  --batch_size N      覆盖 batch size
  --lr LR             覆盖学习率
  --device DEVICE     指定设备 (cuda/cpu)
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_interpolate():
    """Step 1: PCHIP 数据插值。"""
    print("\n" + "=" * 60)
    print("  Step 1: PCHIP 数据插值")
    print("=" * 60)
    from Data_Interpolation import main as interp_main
    interp_main()
    return True


def run_preprocess():
    """Step 2: 数据预处理 (GAF 变换 + 样本构建)。"""
    print("\n" + "=" * 60)
    print("  Step 2: 数据预处理")
    print("=" * 60)
    from data_preprocessing import main as preproc_main
    preproc_main()
    return True


def run_train(args):
    """Step 3: 模型训练。"""
    print("\n" + "=" * 60)
    print("  Step 3: 模型训练")
    print("=" * 60)

    # 应用命令行覆盖
    overrides = {}
    if args.epochs:
        import config
        config.NUM_EPOCHS = args.epochs
        overrides["epochs"] = args.epochs
    if args.batch_size:
        import config
        config.BATCH_SIZE = args.batch_size
        overrides["batch_size"] = args.batch_size
    if args.lr:
        import config
        config.LEARNING_RATE = args.lr
        overrides["lr"] = args.lr
    if args.device:
        import config
        config.DEVICE = args.device
        overrides["device"] = args.device

    if overrides:
        print(f"配置覆盖: {overrides}")

    from train import train
    train()
    return True


def run_test(args):
    """Step 4: 模型测试。"""
    print("\n" + "=" * 60)
    print("  Step 4: 模型测试")
    print("=" * 60)

    if args.device:
        import config
        config.DEVICE = args.device

    from test import main as test_main
    test_main()
    return True


def run_sample():
    """生成模拟水质数据用于测试。"""
    print("\n" + "=" * 60)
    print("  生成模拟数据")
    print("=" * 60)
    from Data_Interpolation import generate_sample_data
    from config import RAW_DATA_PATH
    generate_sample_data(RAW_DATA_PATH, n_days=365, missing_ratio=0.05)
    return True


def run_all(args):
    """完整流程: 插值 → 预处理 → 训练 → 测试。"""
    t_start = time.time()

    steps = [
        ("PCHIP 插值", run_interpolate),
        ("数据预处理", run_preprocess),
        ("模型训练", lambda: run_train(args)),
        ("模型测试", lambda: run_test(args)),
    ]

    for step_name, step_fn in steps:
        print(f"\n{'#'*60}")
        print(f"# {step_name}")
        print(f"{'#'*60}")
        try:
            step_fn()
        except Exception as e:
            print(f"\n错误: {step_name} 失败!")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("\n流程中断。请解决上述错误后重新运行。")
            return False

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  全部流程完成! 总耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")
    return True


def run_info():
    """打印配置信息。"""
    from config import print_config
    print_config()

    # 检查各阶段文件状态
    from config import RAW_DATA_PATH, INTERPOLATED_DATA_PATH
    from config import PROCESSED_DATA_DIR, CHECKPOINT_DIR, CHECKPOINT_FILENAME

    print("\n--- 文件状态 ---")
    checks = [
        ("原始数据", RAW_DATA_PATH),
        ("插值数据", INTERPOLATED_DATA_PATH),
        ("预处理数据", os.path.join(PROCESSED_DATA_DIR, "dataset.pt")),
        ("训练模型", os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)),
    ]
    for name, path in checks:
        status = "[OK] exists" if os.path.exists(path) else "[--] missing"
        print(f"  {name:10s}: {status}  ({path})")


def main():
    parser = argparse.ArgumentParser(
        description="水质预测系统 — SimVPv2 + TGGC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode info        # 查看配置和文件状态
  python main.py --mode sample      # 生成模拟数据
  python main.py --mode all         # 一键运行完整流程
  python main.py --mode train --epochs 300 --lr 5e-4  # 自定义训练
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="info",
        choices=["all", "interpolate", "preprocess", "train", "test", "sample", "info"],
        help="运行模式 (默认: info)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")

    args = parser.parse_args()

    print("=" * 60)
    print("  水质预测系统")
    print("  SimVPv2 + TGGC 时空融合模型")
    print("=" * 60)

    mode_handlers = {
        "info": run_info,
        "sample": run_sample,
        "interpolate": run_interpolate,
        "preprocess": run_preprocess,
        "train": lambda: run_train(args),
        "test": lambda: run_test(args),
        "all": lambda: run_all(args),
    }

    handler = mode_handlers[args.mode]
    handler()


if __name__ == "__main__":
    main()
