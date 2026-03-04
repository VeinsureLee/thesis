# -*- coding: utf-8 -*-
"""LSTM 测试函数：在测试集上评估并返回 MAE、RMSE 等指标。"""

import torch
from torch.utils.data import DataLoader
from typing import Dict

from neutral_network.net import ClosePredictorLSTM  # 兼容单路/多路 LSTM


def evaluate_model(
    model: ClosePredictorLSTM,  # 或 MultiCategoryLSTM，接口一致
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    在测试集上评估模型，计算 MAE、RMSE、MAPE（百分比）。

    Returns:
        含 'mae', 'rmse', 'mape' 的字典
    """
    model = model.to(device)
    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred = model(X)
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())

    import numpy as np
    preds = np.concatenate(preds, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()

    mae = np.abs(preds - targets).mean()
    rmse = np.sqrt(((preds - targets) ** 2).mean())
    # MAPE: 避免除零
    mask = np.abs(targets) > 1e-8
    mape = (np.abs((preds[mask] - targets[mask]) / targets[mask]).mean() * 100.0) if mask.any() else float("nan")

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}
