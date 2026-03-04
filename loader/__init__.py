# -*- coding: utf-8 -*-
"""
数据加载包：从 CSV 加载股票数据并转为 Darts TimeSeries，支持按时间划分。

用法示例:
    from loader import load_and_prepare_data, split_by_time

    series_list = load_and_prepare_data("data/stock_000001.csv")
    train_series, val_series, test_series = split_by_time(
        series_list, train_end="2019-12-31", val_end="2023-12-31"
    )
"""

from .reader import read_stock_csv, group_by_stock
from .prepare import (
    select_value_columns,
    clean_nan_inf,
    group_to_timeseries,
    dataframes_to_series_list,
)
from .split import split_by_time
from .yml_loader import load_yml, load_stock_columns, get_columns_by_category
from . import constants


def load_and_prepare_data(
    file_path: str,
    min_trading_days: int = None,
    verbose: bool = True,
):
    """
    加载 CSV 文件并转换为 Darts TimeSeries 列表。
    假设 CSV 包含列: trade_date, ts_code, open, high, low, close, ...
    """
    from .constants import MIN_TRADING_DAYS
    min_len = min_trading_days if min_trading_days is not None else MIN_TRADING_DAYS
    if verbose:
        print("正在加载数据...")
    df = read_stock_csv(file_path)
    grouped = group_by_stock(df)
    return dataframes_to_series_list(grouped, min_len=min_len, verbose=verbose)


__all__ = [
    "load_and_prepare_data",
    "split_by_time",
    "read_stock_csv",
    "group_by_stock",
    "select_value_columns",
    "clean_nan_inf",
    "group_to_timeseries",
    "dataframes_to_series_list",
    "load_yml",
    "load_stock_columns",
    "get_columns_by_category",
    "constants",
]
