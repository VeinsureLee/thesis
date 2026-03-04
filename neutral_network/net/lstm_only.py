# -*- coding: utf-8 -*-
"""
LSTM 网络：用过去 60 天特征预测下一日收盘价（close）。
- ClosePredictorLSTM: 单一大 LSTM。
- MultiCategoryLSTM: 按 config/stock_columns.yml 类型分多路 LSTM，拼接后全连接预测。
"""

import torch
import torch.nn as nn
from typing import List


class ClosePredictorLSTM(nn.Module):
    """
    使用 LSTM 根据过去 seq_len 天的多特征序列预测下一日 close。
    参考 config/stock_columns.yml 中的数据类型，input_size 为特征数。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(hidden_size * num_directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, 1) 下一日 close 预测值
        """
        out, _ = self.lstm(x)
        last_h = out[:, -1, :]
        return self.linear(last_h)


class MultiCategoryLSTM(nn.Module):
    """
    按 config/stock_columns.yml 中不同类型数据分别经过独立 LSTM，拼接隐状态后再通过全连接层预测下一日 close。
    输入 x: (batch, seq_len, total_features)，按 category_dims 切分为多段，每段送入对应 LSTM。
    """

    def __init__(
        self,
        category_dims: List[int],
        hidden_per_category: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.category_dims = category_dims
        self.hidden_per_category = hidden_per_category
        # 只对特征数 > 0 的类别建 LSTM
        self.lstms = nn.ModuleList()
        self.valid_indices = []  # 每个 (start, end) 对应一个 LSTM
        start = 0
        for d in category_dims:
            if d > 0:
                self.lstms.append(
                    nn.LSTM(
                        input_size=d,
                        hidden_size=hidden_per_category,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0.0,
                    )
                )
                self.valid_indices.append((start, start + d))
            start += d
        total_hidden = len(self.lstms) * hidden_per_category
        self.fc = nn.Linear(total_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, total_features)，特征按类别顺序拼接
        Returns:
            (batch, 1) 下一日 close 预测值
        """
        outs = []
        for (s, e), lstm in zip(self.valid_indices, self.lstms):
            seg = x[:, :, s:e]
            out, _ = lstm(seg)
            outs.append(out[:, -1, :])
        h = torch.cat(outs, dim=1)
        return self.fc(h)
