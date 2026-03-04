# -*- coding: utf-8 -*-
"""
调试入口：用 data/stock_000001.csv 训练 LSTM（60 天预测 1 天 close），并跑测试。
运行: python -m neutral_network
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from neutral_network.dataset import build_datasets_by_category
from neutral_network.net import MultiCategoryLSTM
from neutral_network.train import train_model
from neutral_network.test import evaluate_model


def _project_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root


def main():
    root = _project_root()
    csv_path = root / "data" / "stock_000001.csv"
    yml_path = root / "config" / "stock_columns.yml"

    if not csv_path.exists():
        print(f"未找到数据文件: {csv_path}")
        return

    seq_len = 60
    batch_size = 32
    epochs = 50
    lr = 1e-3
    hidden_size = 64
    num_layers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("构建数据集（按类型分 LSTM：60 天预测 1 天 close）...")
    train_ds, val_ds, test_ds, value_cols, category_dims, _ = build_datasets_by_category(
        str(csv_path),
        seq_len=seq_len,
        train_ratio=0.7,
        val_ratio=0.15,
        yml_path=str(yml_path),
        target_col="close",
    )
    n_features = len(value_cols)
    print(f"总特征数: {n_features}, 各类别维度: {category_dims}")
    print(f"训练样本: {len(train_ds)}, 验证: {len(val_ds)}, 测试: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MultiCategoryLSTM(
        category_dims=category_dims,
        hidden_per_category=hidden_size,
        num_layers=num_layers,
        dropout=0.1,
    )

    print("开始训练...")
    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        device,
        epochs=epochs,
        lr=lr,
        val_loader=val_loader,
    )

    print("\n测试集评估:")
    metrics = evaluate_model(model, test_loader, device)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if k != "mape" or v == v else f"  {k}: {v}")


if __name__ == "__main__":
    main()
