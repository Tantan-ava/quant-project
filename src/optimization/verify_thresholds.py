# -*- coding: utf-8 -*-
"""
验证优化后的阈值是否正确应用
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.strategy.daily_tactician import DailyTactician

# 加载情绪数据
sentiment = pd.read_csv('data/processed/daily_sentiment_index.csv', index_col=0, parse_dates=True)

print("="*60)
print("验证优化后的情绪阈值")
print("="*60)

# 创建DailyTactician实例
dt = DailyTactician(sentiment, use_shock_signal=True)

print("\n【优化后的阈值配置】")
for k, v in dt.thresholds.items():
    print(f"  {k:20s}: {v:8.4f}")

# 测试几个日期的信号
print("\n【样本日期信号测试】")
test_dates = ['2022-02-07', '2022-03-04', '2022-06-01', '2023-01-03', '2024-01-02']

for date_str in test_dates:
    try:
        date = pd.Timestamp(date_str)
        scalar, signal_type, score = dt.get_position_scalar(date)
        zscore = dt.sentiment.loc[date, 'sentiment_zscore']
        print(f"  {date_str}: Z={zscore:7.4f} -> {signal_type:25s} (系数={scalar})")
    except Exception as e:
        print(f"  {date_str}: 错误 - {e}")

# 统计所有交易日的信号分布
print("\n【全样本信号分布】")
signal_counts = {}
for date in dt.sentiment.index:
    try:
        scalar, signal_type, _ = dt.get_position_scalar(date)
        signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
    except:
        pass

total = sum(signal_counts.values())
for signal_type, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
    pct = count / total * 100
    print(f"  {signal_type:25s}: {count:4d} 天 ({pct:5.2f}%)")

print("\n" + "="*60)
