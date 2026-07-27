"""
训练脚本

功能:
- 加载预处理后的数据集
- 初始化 WaterQualityPredictor 模型
- 训练循环（含验证、checkpoint、TensorBoard 日志）
- 支持 Early Stopping 和混合精度训练
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    RESULT_DIR,
    TOTAL_STATIONS,
    INDICATORS,
    INDICATOR_NAMES,
    GAF_WINDOW_SIZE,
    GAF_IMAGE_SIZE,
    T_IN,
    T_OUT,
    SIMVP_IN_CHANNELS,
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
    ENCODER_NUM_LAYERS,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LR_T_0,
    LR_T_MULT,
    LR_ETA_MIN,
    GRAD_CLIP,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    DEVICE,
    USE_AMP,
    NUM_WORKERS,
    LOG_INTERVAL,
    VAL_INTERVAL,
    CHECKPOINT_FILENAME,
    print_config,
)
from model import WaterQualityPredictor
from data_preprocessing import WaterQualityDataset, create_dataloaders
from utils.early_stopping import EarlyStopping
from utils.metrics import compute_all_metrics


def load_datasets() -> Dict[str, WaterQualityDataset]:
    """加载保存的数据集。"""
    dataset_path = os.path.join(PROCESSED_DATA_DIR, "dataset.pt")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"数据集不存在: {dataset_path}\n请先运行 data_preprocessing.py"
        )

    data = torch.load(dataset_path, map_location="cpu")
    train_ds = WaterQualityDataset(data["train_X"], data["train_Y"])
    val_ds = WaterQualityDataset(data["val_X"], data["val_Y"])
    test_ds = WaterQualityDataset(data["test_X"], data["test_Y"])

    print(f"数据集已加载:")
    print(f"  Train: {len(train_ds)} 样本")
    print(f"  Val:   {len(val_ds)} 样本")
    print(f"  Test:  {len(test_ds)} 样本")

    return {"train": train_ds, "val": val_ds, "test": test_ds}


def build_model() -> WaterQualityPredictor:
    """构建融合模型 (基于原始 SimVPv2 + TGGC 架构)。"""
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
    return model


def train_epoch(
    model: WaterQualityPredictor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Optional[GradScaler],
    epoch: int,
    writer: SummaryWriter,
) -> float:
    """
    一个训练 epoch。

    Returns:
        平均训练 loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    batch_count_global = epoch * len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d} [Train]", leave=False)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(DEVICE, non_blocking=True)
        batch_y = batch_y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP and scaler is not None:
            with autocast():
                output = model(batch_x, return_alpha=False)
                loss = criterion(output["pred"], batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(batch_x, return_alpha=False)
            loss = criterion(output["pred"], batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        # TensorBoard 日志
        if num_batches % LOG_INTERVAL == 0:
            writer.add_scalar("train/batch_loss", loss.item(), batch_count_global)
        batch_count_global += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: WaterQualityPredictor,
    dataloader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    writer: SummaryWriter,
) -> Dict[str, float]:
    """
    验证阶段。

    Returns:
        包含 val_loss 和各项指标的字典
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d} [Val]  ", leave=False)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(DEVICE, non_blocking=True)
        batch_y = batch_y.to(DEVICE, non_blocking=True)

        output = model(batch_x, return_alpha=False)
        loss = criterion(output["pred"], batch_y)

        total_loss += loss.item()
        num_batches += 1

        all_preds.append(output["pred"].cpu())
        all_targets.append(batch_y.cpu())

        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / max(num_batches, 1)

    # 计算综合指标
    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_all_metrics(preds_cat, targets_cat, prefix="val/")

    # TensorBoard
    writer.add_scalar("val/loss", avg_loss, epoch)
    for k, v in metrics.items():
        writer.add_scalar(k, v, epoch)

    return {"loss": avg_loss, **metrics}


def train(config_override: Optional[Dict] = None):
    """完整训练流程。"""
    print_config()
    print(f"\n设备: {DEVICE}")
    print(f"混合精度: {USE_AMP}")

    # ---- 加载数据 ----
    datasets = load_datasets()
    loaders = create_dataloaders(
        datasets["train"], datasets["val"], datasets["test"],
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    )

    # ---- 构建模型 ----
    model = build_model().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    # ---- 优化器和调度器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=LR_T_0, T_mult=LR_T_MULT, eta_min=LR_ETA_MIN,
    )

    # ---- 损失函数 ----
    criterion = nn.MSELoss()

    # ---- 混合精度 ----
    scaler = GradScaler(enabled=USE_AMP)

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir=LOG_DIR)

    # ---- Early Stopping ----
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
    early_stopper = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        mode="min",
    )

    # ---- 训练循环 ----
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print(f"\n{'='*60}")
    print(f"  开始训练 — {NUM_EPOCHS} epochs")
    print(f"{'='*60}\n")

    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # 训练
        train_loss = train_epoch(
            model, loaders["train"], optimizer, criterion,
            scaler, epoch, writer,
        )
        train_losses.append(train_loss)

        # 学习率
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/lr", current_lr, epoch)

        # 验证
        if epoch % VAL_INTERVAL == 0:
            val_metrics = validate(
                model, loaders["val"], criterion, epoch, writer,
            )
            val_loss = val_metrics["loss"]
            val_losses.append(val_loss)

            # 打印进度
            elapsed = time.time() - t_start
            eta = elapsed / epoch * (NUM_EPOCHS - epoch)
            print(
                f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MAE: {val_metrics.get('val/mae', 0):.6f} | "
                f"Val SSIM: {val_metrics.get('val/ssim', 0):.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
            )

            # Early Stopping
            is_best = early_stopper(val_loss, model, optimizer, epoch, checkpoint_path)
            if is_best:
                best_val_loss = val_loss
                print(f"  >>> 最佳模型已保存 (val_loss={val_loss:.6f})")

            scheduler.step()

            if early_stopper.early_stop:
                print(f"\nEarly Stopping 触发! (patience={EARLY_STOPPING_PATIENCE})")
                break
        else:
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
            scheduler.step()

    t_total = time.time() - t_start
    print(f"\n训练完成! 总耗时: {t_total:.0f}s ({t_total/60:.1f}min)")
    print(f"最佳 val_loss: {best_val_loss:.6f} (epoch {early_stopper.best_epoch})")

    # ---- 加载最佳模型 ----
    best_epoch = early_stopper.load_best(model, checkpoint_path)
    print(f"已加载最佳模型 (epoch {best_epoch})")

    # ---- 保存训练历史 ----
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": early_stopper.best_epoch,
        "best_val_loss": best_val_loss,
    }
    import json
    with open(os.path.join(RESULT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ---- 测试集评估 ----
    print(f"\n{'='*60}")
    print(f"  测试集评估")
    print(f"{'='*60}")
    model.eval()
    test_metrics = validate(model, loaders["test"], criterion, 0, writer)
    for k, v in test_metrics.items():
        if k != "loss":
            print(f"  {k}: {v:.6f}")

    writer.close()
    return model, history


if __name__ == "__main__":
    train()
