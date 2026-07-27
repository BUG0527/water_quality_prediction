"""
Early Stopping 工具

监控验证指标，在指标不再提升时提前终止训练。
"""

import numpy as np
import torch


class EarlyStopping:
    """
    Early Stopping 实现。

    用法:
        early_stopper = EarlyStopping(patience=15, min_delta=1e-4)
        for epoch in range(epochs):
            val_loss = validate()
            early_stopper(val_loss, model, optimizer, epoch)
            if early_stopper.early_stop:
                break
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0

    def __call__(
        self,
        score: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        path: str,
    ) -> bool:
        """
        Args:
            score:  当前验证分数（通常是 val_loss）
            model:  模型
            optimizer: 优化器
            epoch:  当前 epoch
            path:   checkpoint 保存路径

        Returns:
            True 如果 score 是最佳的（刚保存了 checkpoint）
        """
        if self.mode == "min":
            is_better = self.best_score is None or score < self.best_score - self.min_delta
        else:
            is_better = self.best_score is None or score > self.best_score + self.min_delta

        if is_better:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model, optimizer, epoch, score, path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def _save_checkpoint(self, model, optimizer, epoch, score, path):
        """保存最佳 checkpoint。"""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_score": score,
            },
            path,
        )

    def load_best(self, model: torch.nn.Module, path: str) -> int:
        """
        从 checkpoint 加载最佳模型。

        Returns:
            最佳 epoch 数
        """
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint["epoch"]
