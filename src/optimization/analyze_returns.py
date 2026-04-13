# -*- coding: utf-8 -*-
"""
分析优化阈值后的收益率序列
"""

import pandas as pd
import numpy as np

# 读取收益率序列
returns_df = pd.read_csv('results/tables/hybrid_strategy_returns_optimized.csv')
returns_df['date'] = pd.to_datetime(returns_df['date'])

print("="*60)
print("优化阈值后的策略收益率分析")
print("="*60)

print(f"\n数据期间: {returns_df['date'].min().strftime('%Y-%m-%d')} 至 {returns_df['date'].max().strftime('%Y-%m-%d')}")
print(f"总月数: {len(returns_df)}")

# 月度收益率统计
monthly_returns = returns_df['return_0cost']

print(f"\n【月度收益率统计】")
print(f"  均值: {monthly_returns.mean()*100:.4f}%")
print(f"  标准差: {monthly_returns.std()*100:.4f}%")
print(f"  最小值: {monthly_returns.min()*100:.4f}%")
print(f"  最大值: {monthly_returns.max()*100:.4f}%")

# 年化收益率计算
# 方法1: 基于月度均值年化
annualized_return_mean = (1 + monthly_returns.mean())**12 - 1

# 方法2: 基于总收益计算
# 计算累计收益
cumulative_return = (1 + monthly_returns).prod() - 1
# 年化 = (1 + 总收益)^(12/月数) - 1
n_months = len(monthly_returns)
annualized_return_cumulative = (1 + cumulative_return)**(12/n_months) - 1

# 年化波动率
annualized_vol = monthly_returns.std() * np.sqrt(12)

# 夏普比率（假设无风险利率0）
sharpe = annualized_return_mean / annualized_vol if annualized_vol > 0 else 0

# 最大回撤
portfolio_value = (1 + monthly_returns).cumprod()
running_max = portfolio_value.cummax()
drawdown = (portfolio_value - running_max) / running_max
max_drawdown = drawdown.min()

print(f"\n【年化绩效指标】")
print(f"  累计收益率: {cumulative_return*100:.2f}%")
print(f"  年化收益率(均值法): {annualized_return_mean*100:.2f}%")
print(f"  年化收益率(复利法): {annualized_return_cumulative*100:.2f}%")
print(f"  年化波动率: {annualized_vol*100:.2f}%")
print(f"  夏普比率: {sharpe:.3f}")
print(f"  最大回撤: {max_drawdown*100:.2f}%")

# 年度收益分解
print(f"\n【年度收益分解】")
returns_df['year'] = returns_df['date'].dt.year
annual_returns = returns_df.groupby('year')['return_0cost'].apply(lambda x: (1 + x).prod() - 1)
for year, ret in annual_returns.items():
    print(f"  {year}年: {ret*100:+.2f}%")

print("="*60)
