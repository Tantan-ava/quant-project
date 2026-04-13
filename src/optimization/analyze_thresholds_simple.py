# -*- coding: utf-8 -*-
"""
简化版阈值分析 - 直接查看情绪分布和信号触发情况
"""

import pandas as pd
import numpy as np

# 加载情绪数据
sentiment = pd.read_csv('data/processed/daily_sentiment_index.csv', index_col=0, parse_dates=True)

print("="*60)
print("情绪数据分布分析")
print("="*60)

# 基础统计
print(f"\n数据期间: {sentiment.index.min().strftime('%Y-%m-%d')} 至 {sentiment.index.max().strftime('%Y-%m-%d')}")
print(f"总天数: {len(sentiment)}")

# 计算Z-score
sentiment['zscore'] = (sentiment['sentiment_score'] - sentiment['sentiment_score'].mean()) / sentiment['sentiment_score'].std()

print(f"\n情绪分数统计:")
print(f"  均值: {sentiment['sentiment_score'].mean():.4f}")
print(f"  标准差: {sentiment['sentiment_score'].std():.4f}")
print(f"  最小值: {sentiment['sentiment_score'].min():.4f}")
print(f"  最大值: {sentiment['sentiment_score'].max():.4f}")

print(f"\nZ-score统计:")
print(f"  最小值: {sentiment['zscore'].min():.4f}")
print(f"  最大值: {sentiment['zscore'].max():.4f}")

# 百分位数
print("\n【Z-score百分位】")
for p in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
    val = np.percentile(sentiment['zscore'], p)
    print(f"  P{p:2d}: {val:8.4f}")

# 原阈值 vs 优化阈值
print("\n【阈值对比】")
print("\n原阈值（基于正态分布理论）:")
old_thresholds = {
    'extreme_panic': -2.0,
    'panic': -1.0,
    'neutral_low': -0.5,
    'neutral_high': 0.5,
    'optimistic': 1.0,
    'extreme_greed': 2.0,
}

z = sentiment['zscore']
total = len(z)

for name, threshold in old_thresholds.items():
    if 'panic' in name:
        count = (z < threshold).sum() if 'extreme' in name else ((z >= -2.0) & (z < threshold)).sum()
    elif 'greed' in name:
        count = (z > threshold).sum() if 'extreme' in name else ((z > 1.0) & (z <= threshold)).sum()
    elif 'neutral' in name:
        continue
    else:
        count = 0
    print(f"  {name:20s}: {threshold:8.4f} -> 触发 {count:4d} 天 ({count/total*100:5.2f}%)")

print("\n优化阈值（基于百分位数）:")
new_thresholds = {
    'extreme_panic': np.percentile(z, 5),
    'panic': np.percentile(z, 20),
    'neutral_low': np.percentile(z, 40),
    'neutral_high': np.percentile(z, 60),
    'optimistic': np.percentile(z, 80),
    'extreme_greed': np.percentile(z, 95),
}

for name, threshold in new_thresholds.items():
    print(f"  {name:20s}: {threshold:8.4f}")

# 各区间触发频率（优化后）
print("\n【优化后区间分布】")
extreme_panic = (z < new_thresholds['extreme_panic']).sum()
panic = ((z >= new_thresholds['extreme_panic']) & (z < new_thresholds['panic'])).sum()
concern = ((z >= new_thresholds['panic']) & (z < new_thresholds['neutral_low'])).sum()
neutral = ((z >= new_thresholds['neutral_low']) & (z <= new_thresholds['neutral_high'])).sum()
optimistic_hold = ((z > new_thresholds['neutral_high']) & (z <= new_thresholds['optimistic'])).sum()
optimistic_reduce = ((z > new_thresholds['optimistic']) & (z <= new_thresholds['extreme_greed'])).sum()
extreme_greed = (z > new_thresholds['extreme_greed']).sum()

print(f"极端恐慌 (超配150%): {extreme_panic:4d} 天 ({extreme_panic/total*100:5.2f}%)")
print(f"恐慌 (超配120%): {panic:4d} 天 ({panic/total*100:5.2f}%)")
print(f"轻度担忧 (低配80%): {concern:4d} 天 ({concern/total*100:5.2f}%)")
print(f"中性 (标配100%): {neutral:4d} 天 ({neutral/total*100:5.2f}%)")
print(f"轻度乐观 (标配100%): {optimistic_hold:4d} 天 ({optimistic_hold/total*100:5.2f}%)")
print(f"乐观 (减仓80%): {optimistic_reduce:4d} 天 ({optimistic_reduce/total*100:5.2f}%)")
print(f"极度乐观 (减仓50%): {extreme_greed:4d} 天 ({extreme_greed/total*100:5.2f}%)")

# 加权平均仓位系数
weighted_scalar = (
    extreme_panic * 1.5 + 
    panic * 1.2 + 
    concern * 0.8 + 
    neutral * 1.0 + 
    optimistic_hold * 1.0 + 
    optimistic_reduce * 0.8 + 
    extreme_greed * 0.5
) / total

print(f"\n加权平均仓位系数: {weighted_scalar:.4f}")
print("="*60)
