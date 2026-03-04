# -*- coding: utf-8 -*-
"""
构建 60 天预测 1 天的序列数据集。
根据 config/stock_columns.yml 按数据类型分类：每类单独 LSTM 后拼接再全连接预测 close。
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

from loader import load_stock_columns, get_columns_by_category


# 不参与建模的列（标识或日期）
SKIP_COLS = {"ts_code", "trade_date"}

# 与 config/stock_columns.yml 一致：按类型分别送入不同 LSTM 的顺序
CATEGORY_KEYS = [
    "raw_market_data",
    "adjusted_price_qfq",
    "adjusted_price_hfq",
    "derived",
    "technical_indicators",
]


def get_feature_columns(yml_path=None):
    """
    从 config/stock_columns.yml 获取用于 LSTM 的特征列（all_columns 中的数值列，排除 ts_code/trade_date）。
    """
    config = load_stock_columns(yml_path)
    all_cols = config.get("all_columns") or []
    return [c for c in all_cols if c not in SKIP_COLS and c in all_cols]


def get_feature_columns_by_category(yml_path=None, df_columns=None):
    """
    按 config/stock_columns.yml 中的类型获取特征列。
    返回 (value_cols, category_dims):
      - value_cols: 按类别顺序拼接后的列名（仅保留在 df_columns 中存在的）
      - category_dims: 每个类别对应的特征数 [d1, d2, ...]，用于模型按段切片
    """
    config = load_stock_columns(yml_path)
    available = set(df_columns) if df_columns is not None else None
    value_cols = []
    category_dims = []
    for key in CATEGORY_KEYS:
        cols = get_columns_by_category(config, key)
        cols = [c for c in cols if c not in SKIP_COLS]
        if available is not None:
            cols = [c for c in cols if c in available]
        category_dims.append(len(cols))
        value_cols.extend(cols)
    return value_cols, category_dims


def load_stock_df(csv_path: str) -> pd.DataFrame:
    """读取股票 CSV，按 trade_date 排序。"""
    df = pd.read_csv(csv_path, parse_dates=["trade_date"])
    return df.sort_values("trade_date").reset_index(drop=True)


def prepare_xy(
    df: pd.DataFrame,
    feature_columns: list,
    target_col: str = "close",
    seq_len: int = 60,
):
    """
    从整表构建 (X, y)。
    X: (n_samples, seq_len, n_features)，y: (n_samples,) 下一日 close。
    缺失值用前向/后向填充后填 0。
    返回 (X, y, value_cols)，value_cols 为实际参与的特征列顺序，用于确定 input_size。
    """
    value_cols = [c for c in feature_columns if c in df.columns]
    if target_col not in df.columns:
        raise ValueError(f"目标列 {target_col} 不在数据中")
    if target_col not in value_cols:
        value_cols = value_cols + [target_col]

    data = df[value_cols].replace([np.inf, -np.inf], np.nan)
    data = data.ffill().bfill().fillna(0).values.astype(np.float32)

    n = len(data)
    if n <= seq_len:
        return None, None, value_cols

    X_list = []
    y_list = []
    target_idx = value_cols.index(target_col)
    for i in range(n - seq_len):
        X_list.append(data[i : i + seq_len])
        y_list.append(data[i + seq_len, target_idx])
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    return X, y, value_cols


class StockCloseDataset(Dataset):
    """PyTorch Dataset: 60 天特征 -> 下一日 close。"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_datasets(
    csv_path: str,
    seq_len: int = 60,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    yml_path=None,
    target_col: str = "close",
):
    """
    从 CSV 构建训练/验证/测试 Dataset，并按时间划分（前 train_ratio 训练，中间 val_ratio 验证，其余测试）。
    返回 (train_ds, val_ds, test_ds, feature_columns, scaler_dict)。
    scaler_dict 含 'mean' 与 'std'，用于对 X 做标准化（按训练集统计）；y 不标准化，直接预测 close。
    """
    feature_columns = get_feature_columns(yml_path)
    df = load_stock_df(csv_path)
    X, y, value_cols = prepare_xy(df, feature_columns, target_col=target_col, seq_len=seq_len)
    if X is None:
        raise ValueError("样本数不足，无法构建 60 天序列")

    n = len(X)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    X_train, X_val, X_test = X[:t1], X[t1:t2], X[t2:]
    y_train, y_val, y_test = y[:t1], y[t1:t2], y[t2:]

    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0).astype(np.float32)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    scaler_dict = {"mean": mean, "std": std}
    train_ds = StockCloseDataset(X_train, y_train)
    val_ds = StockCloseDataset(X_val, y_val)
    test_ds = StockCloseDataset(X_test, y_test)
    return train_ds, val_ds, test_ds, value_cols, scaler_dict


def build_datasets_by_category(
    csv_path: str,
    seq_len: int = 60,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    yml_path=None,
    target_col: str = "close",
):
    """
    按 config/stock_columns.yml 中的数据类型分别构建特征，用于「每类一个 LSTM → 拼接 → 全连接」模型。
    返回 (train_ds, val_ds, test_ds, value_cols, category_dims, scaler_dict)。
    category_dims[i] 为第 i 类特征数，模型据此对输入按段切片。
    """
    df = load_stock_df(csv_path)
    value_cols, category_dims = get_feature_columns_by_category(yml_path, df.columns.tolist())
    if not value_cols:
        raise ValueError("未找到任何类别下的有效特征列，请检查 config/stock_columns.yml 与 CSV 列名")

    X, y, value_cols_used = prepare_xy(df, value_cols, target_col=target_col, seq_len=seq_len)
    if X is None:
        raise ValueError("样本数不足，无法构建 60 天序列")

    # 重新计算 category_dims：prepare_xy 可能去掉了不存在的列，value_cols_used 与 value_cols 一致（因已用 df 过滤过）
    # 若 prepare_xy 内部会再过滤，这里 value_cols 已是 df 中存在的，故 value_cols_used 与 value_cols 相同
    n = len(X)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    X_train, X_val, X_test = X[:t1], X[t1:t2], X[t2:]
    y_train, y_val, y_test = y[:t1], y[t1:t2], y[t2:]

    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0).astype(np.float32)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    scaler_dict = {"mean": mean, "std": std}
    train_ds = StockCloseDataset(X_train, y_train)
    val_ds = StockCloseDataset(X_val, y_val)
    test_ds = StockCloseDataset(X_test, y_test)
    return train_ds, val_ds, test_ds, value_cols_used, category_dims, scaler_dict
