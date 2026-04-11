# -*- coding: utf-8 -*-
"""
生成策略收益序列
根据 Strategy_Card.md 中的优化配置：
- 形成期 K=6
- TopK=100
- 标准化=Winsorization
- 再平衡=月度
"""

import pandas as pd
import numpy as np
import os

# 设置数据路径
DATA_PATH = '/Users/xinyutan/Documents/量化投资/quant-project/data/processed/TRD_Mnth.xlsx'
OUTPUT_PATH = '/Users/xinyutan/Documents/量化投资/quant-project/data/processed/Strategy_Returns_K6_Top100.xlsx'

print("="*80)
print("生成策略收益序列")
print("="*80)
print(f"\n策略配置:")
print(f"  - 形成期 K=6")
print(f"  - TopK=100")
print(f"  - 标准化=Winsorization")
print(f"  - 再平衡=月度")
print(f"\n数据来源: {DATA_PATH}")

# ============================================================================
# 1. 读取数据
# ============================================================================
print("\n" + "="*80)
print("步骤1: 读取收益率数据")
print("="*80)

df_returns = pd.read_excel(DATA_PATH, index_col=0, engine='openpyxl')
df_returns.index = pd.to_datetime(df_returns.index)

print(f"数据形状: {df_returns.shape}")
print(f"数据范围: {df_returns.index[0].strftime('%Y-%m')} 至 {df_returns.index[-1].strftime('%Y-%m')}")

# ============================================================================
# 2. 构造反转信号
# ============================================================================
print("\n" + "="*80)
print("步骤2: 构造6个月反转信号")
print("="*80)

# 信号定义: s_{i,t} = -sum_{j=1}^{6} r_{i,t-j}
K = 6
past_cumulative_returns = df_returns.rolling(window=K).sum()
raw_signal = -1 * past_cumulative_returns.shift(1)  # shift(1) 防止未来数据泄露

print(f"信号矩阵形状: {raw_signal.shape}")
print(f"信号定义: Signal_t = -Sum(Return_{{t-1}} to Return_{{t-{K}}})")

# ============================================================================
# 3. 信号标准化 (Winsorization)
# ============================================================================
print("\n" + "="*80)
print("步骤3: Winsorization标准化")
print("="*80)

def winsorize_row(row):
    """5%分位数缩尾处理"""
    lower = row.quantile(0.05)
    upper = row.quantile(0.95)
    return row.clip(lower, upper)

signal = raw_signal.apply(winsorize_row, axis=1)
print("标准化方法: Winsorization (5%分位数上下限)")

# ============================================================================
# 4. 组合构建与回测
# ============================================================================
print("\n" + "="*80)
print("步骤4: 组合构建与回测")
print("="*80)

TopK = 100
rebalance_freq = 'M'

# 初始化权重矩阵
weights = pd.DataFrame(0.0, index=df_returns.index, columns=df_returns.columns, dtype=float)

# 确定再平衡月份
if rebalance_freq == 'M':
    rebalance_months = df_returns.index[K:]
elif rebalance_freq == 'Q':
    rebalance_months = df_returns.index[K:][::3]
else:
    raise ValueError(f"Unknown rebalance frequency: {rebalance_freq}")

current_weights = None

for date in df_returns.index:
    if date in rebalance_months:
        current_signal = signal.loc[date]
        ranks = current_signal.rank(ascending=False, method='first')
        selected = (ranks <= TopK).astype(float)
        selected_count = selected.sum()
        if selected_count > 0:
            current_weights = selected / selected_count
        else:
            current_weights = selected
    
    if current_weights is not None:
        weights.loc[date] = current_weights

# 计算组合收益
port_ret = (weights * df_returns).sum(axis=1)
port_ret = port_ret.iloc[K:]  # 去除前K期（无有效信号）
weights = weights.iloc[K:]

# 计算换手率
monthly_turnover = weights.diff().abs().sum(axis=1) * 0.5
monthly_turnover.iloc[0] = weights.iloc[0].sum() * 0.5

print(f"选股数量: Top {TopK}")
print(f"再平衡频率: {rebalance_freq} (月度)")
print(f"收益序列长度: {len(port_ret)} 个月")

# ============================================================================
# 5. 计算策略指标
# ============================================================================
print("\n" + "="*80)
print("步骤5: 计算策略指标")
print("="*80)

# 年化收益
cum_ret = (1 + port_ret).prod() - 1
ann_ret = (1 + cum_ret) ** (12 / len(port_ret)) - 1

# 夏普比率
sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(12) if port_ret.std() != 0 else 0

# 最大回撤
cum_wealth = (1 + port_ret).cumprod()
drawdown = (cum_wealth - cum_wealth.cummax()) / cum_wealth.cummax()
max_dd = drawdown.min()

# 年化换手率
annual_turnover = monthly_turnover.mean() * 12 * 100

print(f"\n策略绩效指标:")
print(f"  年化收益:     {ann_ret*100:.2f}%")
print(f"  夏普比率:     {sharpe:.2f}")
print(f"  最大回撤:     {max_dd*100:.2f}%")
print(f"  年化换手率:   {annual_turnover:.2f}%")

# ============================================================================
# 6. 成本敏感性分析
# ============================================================================
print("\n" + "="*80)
print("步骤6: 成本敏感性分析")
print("="*80)

cost_levels = [0, 10, 20, 50]
print(f"\n{'成本(bps)':<12} {'年化收益':<12} {'夏普比率':<12} {'成本拖累':<12}")
print("-" * 50)

results = []
for cost_bps in cost_levels:
    cost_rate = cost_bps / 10000
    cost = monthly_turnover * cost_rate
    net_returns = port_ret - cost
    
    net_cum_ret = (1 + net_returns).prod() - 1
    net_ann_ret = (1 + net_cum_ret) ** (12 / len(net_returns)) - 1
    net_sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(12) if net_returns.std() != 0 else 0
    
    drag = (ann_ret - net_ann_ret) * 100
    
    results.append({
        'Cost_bps': cost_bps,
        'Annual_Return': net_ann_ret,
        'Sharpe': net_sharpe,
        'Cost_Drag': drag
    })
    
    print(f"{cost_bps:<12} {net_ann_ret*100:>10.2f}% {net_sharpe:>10.2f}   {drag:>10.2f}%")

# 计算盈亏平衡点
base_return = results[0]['Annual_Return']
cost_50_return = results[3]['Annual_Return']
slope = (base_return - cost_50_return) / 50
break_even = base_return / slope if slope > 0 else 0
print(f"\n盈亏平衡点: 约 {break_even:.0f} bps")

# ============================================================================
# 7. 保存收益序列
# ============================================================================
print("\n" + "="*80)
print("步骤7: 保存策略收益序列")
print("="*80)

# 创建输出DataFrame
output_df = pd.DataFrame({
    'Date': port_ret.index,
    'Portfolio_Return': port_ret.values,
    'Cumulative_Wealth': cum_wealth.values,
    'Drawdown': drawdown.values,
    'Monthly_Turnover': monthly_turnover.values
})

# 添加成本调整后的收益
for cost_bps in cost_levels:
    cost_rate = cost_bps / 10000
    cost = monthly_turnover * cost_rate
    net_returns = port_ret - cost
    output_df[f'Net_Return_{cost_bps}bps'] = net_returns.values

output_df.to_excel(OUTPUT_PATH, index=False)
print(f"\n策略收益序列已保存至:")
print(f"  {OUTPUT_PATH}")

# 输出文件信息
print(f"\n文件包含以下列:")
print(f"  - Date: 日期")
print(f"  - Portfolio_Return: 组合月度收益（无成本）")
print(f"  - Cumulative_Wealth: 累计净值")
print(f"  - Drawdown: 回撤")
print(f"  - Monthly_Turnover: 月度换手率")
print(f"  - Net_Return_Xbps: X bps成本下的净收益")

print("\n" + "="*80)
print("策略收益序列生成完成!")
print("="*80)
