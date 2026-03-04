# -*- coding: utf-8 -*-
"""数据加载用到的常量：特征列候选等。"""

# 候选特征列（只使用 CSV 中实际存在的列）
FEATURE_CANDIDATES = [
    "close", "open", "high", "low", "pre_close", "change", "pct_chg",
    "vol", "amount",
    "open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq",
    "open_hfq", "high_hfq", "low_hfq", "close_hfq", "pre_close_hfq",
    "turnover_rate", "turnover_rate_f", "volume_ratio",
    "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm",
    "total_share", "float_share", "free_share", "total_mv", "circ_mv",
    "adj_factor",
    "ema_bfq_250", "ema_qfq_250", "ema_hfq_250",
]

# 单只股票最少交易日数量，少于此数的股票会被过滤
MIN_TRADING_DAYS = 500

# 时间列与频率
TIME_COL = "trade_date"
GROUP_COL = "ts_code"
FREQ = "D"
