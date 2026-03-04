# -*- coding: utf-8 -*-
"""将分组后的 DataFrame 清洗并转换为 Darts TimeSeries。"""

import numpy as np
import pandas as pd
from darts import TimeSeries

from .constants import (
    FEATURE_CANDIDATES,
    MIN_TRADING_DAYS,
    TIME_COL,
    FREQ,
)


def select_value_columns(group: pd.DataFrame) -> list:
    """只保留在 group 中存在的候选特征列。"""
    return [c for c in FEATURE_CANDIDATES if c in group.columns]


def clean_nan_inf(df: pd.DataFrame, value_cols: list) -> pd.DataFrame:
    """将 Inf 替换为 NaN，再前向/后向填充，剩余填 0。"""
    sub = df[value_cols].replace([np.inf, -np.inf], np.nan)
    sub = sub.ffill().bfill().fillna(0)
    return sub


def group_to_timeseries(
    group: pd.DataFrame,
    value_cols: list,
    min_len: int = MIN_TRADING_DAYS,
) -> TimeSeries | None:
    """
    将单只股票的一组数据转为 Darts TimeSeries。
    若长度不足 min_len 则返回 None。
    """
    group = group.sort_values(TIME_COL).copy()
    group[value_cols] = clean_nan_inf(group, value_cols)

    series = TimeSeries.from_dataframe(
        group,
        time_col=TIME_COL,
        value_cols=value_cols,
        fill_missing_dates=True,
        freq=FREQ,
    )
    if len(series) < min_len:
        return None
    return series


def dataframes_to_series_list(
    grouped,
    min_len: int = MIN_TRADING_DAYS,
    verbose: bool = True,
):
    """
    将 groupby 后的 (stock_code, group) 转为 TimeSeries 列表。
    verbose 为 True 时打印前 5 只股票信息及总数。
    """
    series_list = []
    for stock_code, group in grouped:
        value_cols = select_value_columns(group)
        if not value_cols:
            continue
        series = group_to_timeseries(group, value_cols, min_len=min_len)
        if series is None:
            continue
        series_list.append(series)
        if verbose and len(series_list) <= 5:
            print(f"股票 {stock_code}: {len(series)} 个交易日")
    if verbose:
        print(f"成功加载 {len(series_list)} 只股票的数据")
    return series_list
