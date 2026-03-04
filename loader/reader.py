# -*- coding: utf-8 -*-
"""从 CSV 读取原始数据并按股票分组。"""

import pandas as pd

from .constants import GROUP_COL, TIME_COL


def read_stock_csv(file_path: str) -> pd.DataFrame:
    """
    加载股票 CSV，解析日期。
    假设包含列: trade_date, ts_code, open, high, low, close, ...
    """
    return pd.read_csv(file_path, parse_dates=[TIME_COL])


def group_by_stock(df: pd.DataFrame):
    """按 ts_code 分组，迭代 (stock_code, group_df)。"""
    return df.groupby(GROUP_COL)
