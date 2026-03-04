# -*- coding: utf-8 -*-
"""按时间划分训练/验证/测试集。"""

import pandas as pd


def split_by_time(
    series_list,
    train_end: str = "2019-12-31",
    val_end: str = "2023-12-31",
    verbose: bool = True,
):
    """
    严格按时间划分训练/验证/测试集，使用 Darts 的 split_before。
    返回 (train_series, val_series, test_series)。
    """
    train_series = []
    val_series = []
    test_series = []

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    stock_stats = []

    for i, series in enumerate(series_list):
        try:
            train_part, rest = series.split_before(train_end_ts)
            val_part, test_part = rest.split_before(val_end_ts)

            if len(train_part) > 0:
                train_series.append(train_part)
            if len(val_part) > 0:
                val_series.append(val_part)
            if len(test_part) > 0:
                test_series.append(test_part)

            if i < 3:
                stock_stats.append({
                    "index": i,
                    "总天数": len(series),
                    "训练集天数": len(train_part) if len(train_part) > 0 else 0,
                    "验证集天数": len(val_part) if len(val_part) > 0 else 0,
                    "测试集天数": len(test_part) if len(test_part) > 0 else 0,
                    "训练集范围": f"{train_part.start_time()} 到 {train_part.end_time()}" if len(train_part) > 0 else "无",
                    "验证集范围": f"{val_part.start_time()} 到 {val_part.end_time()}" if len(val_part) > 0 else "无",
                    "测试集范围": f"{test_part.start_time()} 到 {test_part.end_time()}" if len(test_part) > 0 else "无",
                })
        except Exception as e:
            if verbose:
                print(f"处理股票 {i} 时出错: {e}")
            continue

    if verbose:
        print(f"\n训练集: {len(train_series)} 只股票")
        print(f"验证集: {len(val_series)} 只股票")
        print(f"测试集: {len(test_series)} 只股票")
        if stock_stats:
            print("\n前3只股票划分详情:")
            for stat in stock_stats:
                print(f"\n股票 {stat['index']+1}:")
                print(f"  总天数: {stat['总天数']}")
                print(f"  训练集: {stat['训练集天数']}天 ({stat['训练集范围']})")
                print(f"  验证集: {stat['验证集天数']}天 ({stat['验证集范围']})")
                print(f"  测试集: {stat['测试集天数']}天 ({stat['测试集范围']})")

    return train_series, val_series, test_series
