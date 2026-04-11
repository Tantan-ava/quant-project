# -*- coding: utf-8 -*-
"""
合并策略收益序列与CH3因子数据，生成可用于OLS回归的数据
"""

import pandas as pd
import numpy as np

print("="*80)
print("合并策略收益与CH3因子数据")
print("="*80)

# ============================================================================
# 1. 读取数据
# ============================================================================
print("\n步骤1: 读取数据")

# 读取策略收益数据
df_returns = pd.read_csv('/Users/xinyutan/Documents/量化投资/quant-project/results/tables/strategy_returns.csv')
df_returns['Date'] = pd.to_datetime(df_returns['Date'])
# 将日期转换为月末格式以便对齐
df_returns['Date_MonthEnd'] = df_returns['Date'] + pd.offsets.MonthEnd(0)
print(f"策略收益数据: {df_returns.shape}")
print(f"  日期范围: {df_returns['Date'].min().strftime('%Y-%m')} 至 {df_returns['Date'].max().strftime('%Y-%m')}")

# 读取CH3因子数据
df_ch3 = pd.read_excel('/Users/xinyutan/Documents/量化投资/quant-project/data/processed/CH3_factors_monthly_202602.xlsx')
# 将mnthdt转换为日期格式
df_ch3['Date_MonthEnd'] = pd.to_datetime(df_ch3['mnthdt'].astype(str), format='%Y%m%d')
print(f"CH3因子数据: {df_ch3.shape}")
print(f"  日期范围: {df_ch3['Date_MonthEnd'].min().strftime('%Y-%m')} 至 {df_ch3['Date_MonthEnd'].max().strftime('%Y-%m')}")

# ============================================================================
# 2. 数据对齐与合并
# ============================================================================
print("\n步骤2: 数据对齐与合并")

# 选择策略收益中的0成本收益序列
df_returns_selected = df_returns[['Date', 'Date_MonthEnd', 'Portfolio_Return']].copy()
df_returns_selected.columns = ['Date', 'Date_MonthEnd', 'Strategy_Return']

# 选择CH3因子列
df_ch3_selected = df_ch3[['Date_MonthEnd', 'rf_mon', 'mktrf', 'SMB', 'VMG']].copy()

# 合并数据（内连接，取交集）
df_merged = pd.merge(df_returns_selected, df_ch3_selected, on='Date_MonthEnd', how='inner')
df_merged = df_merged.sort_values('Date').reset_index(drop=True)

print(f"合并后数据: {df_merged.shape}")
print(f"  日期范围: {df_merged['Date'].min().strftime('%Y-%m')} 至 {df_merged['Date'].max().strftime('%Y-%m')}")

# ============================================================================
# 3. 计算超额收益
# ============================================================================
print("\n步骤3: 计算超额收益")

# 计算策略超额收益
df_merged['Strategy_Excess_Return'] = df_merged['Strategy_Return'] - df_merged['rf_mon']

print(f"策略平均月收益: {df_merged['Strategy_Return'].mean()*100:.4f}%")
print(f"无风险利率均值: {df_merged['rf_mon'].mean()*100:.4f}%")
print(f"策略超额收益均值: {df_merged['Strategy_Excess_Return'].mean()*100:.4f}%")

# ============================================================================
# 4. 准备回归数据
# ============================================================================
print("\n步骤4: 准备OLS回归数据")

# 创建标准的回归数据集
# 因变量: Strategy_Excess_Return (策略超额收益)
# 自变量: mktrf (市场因子), SMB (规模因子), VMG (价值因子)

df_regression = df_merged[['Date', 'Strategy_Return', 'rf_mon', 'Strategy_Excess_Return', 
                            'mktrf', 'SMB', 'VMG']].copy()

# 添加常数项（用于OLS回归）
df_regression['const'] = 1.0

print("\n回归数据列说明:")
print("  - Date: 日期（月初）")
print("  - Strategy_Return: 策略原始收益")
print("  - rf_mon: 无风险利率")
print("  - Strategy_Excess_Return: 策略超额收益（因变量）")
print("  - mktrf: 市场超额收益因子")
print("  - SMB: 规模因子")
print("  - VMG: 价值因子")
print("  - const: 常数项")

# ============================================================================
# 5. 数据统计摘要
# ============================================================================
print("\n步骤5: 数据统计摘要")

print("\n各变量描述统计:")
print(df_regression[['Strategy_Excess_Return', 'mktrf', 'SMB', 'VMG']].describe().round(4))

# 计算相关系数
print("\n相关系数矩阵:")
corr_cols = ['Strategy_Excess_Return', 'mktrf', 'SMB', 'VMG']
print(df_regression[corr_cols].corr().round(4))

# ============================================================================
# 6. 保存数据
# ============================================================================
print("\n步骤6: 保存回归数据")

# 保存为CSV
output_path = '/Users/xinyutan/Documents/量化投资/quant-project/results/tables/regression_data.csv'
df_regression.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n回归数据已保存至: {output_path}")

# 同时保存为Excel
output_path_excel = '/Users/xinyutan/Documents/量化投资/quant-project/results/tables/regression_data.xlsx'
df_regression.to_excel(output_path_excel, index=False)
print(f"回归数据已保存至: {output_path_excel}")

print("\n" + "="*80)
print("数据合并完成！")
print("="*80)
print(f"\n可用于OLS回归的数据已准备就绪:")
print(f"  - 样本量: {len(df_regression)} 个月")
print(f"  - 因变量: Strategy_Excess_Return")
print(f"  - 自变量: const, mktrf, SMB, VMG")
