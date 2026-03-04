# -*- coding: utf-8 -*-
"""LSTM 训练函数：在给定 DataLoader 上训练模型并返回损失曲线。"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple

from neutral_network.net import ClosePredictorLSTM  # 兼容单路/多路 LSTM


def train_model(
    model: ClosePredictorLSTM,  # 或 MultiCategoryLSTM，接口一致
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    val_loader: Optional[DataLoader] = None,
) -> Tuple[ClosePredictorLSTM, List[float], List[float]]:
    """
    训练 LSTM 预测下一日 close。

    Args:
        model: ClosePredictorLSTM 实例
        train_loader: 训练 DataLoader
        device: 设备 (cuda/cpu)
        epochs: 训练轮数
        lr: 学习率
        val_loader: 可选验证 DataLoader

    Returns:
        (训练好的 model, train_losses, val_losses)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    val_loss += criterion(pred, y).item()
                    n_val += 1
            avg_val = val_loss / max(n_val, 1)
            val_losses.append(avg_val)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}  train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")
        else:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}  train_loss={avg_train:.6f}")

    return model, train_losses, val_losses
