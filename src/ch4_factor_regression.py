# -*- coding: utf-8 -*-
"""
CH-4因子回归分析 + 5年滚动窗口归因
检验CH-4能否完全解释反转异象
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CH-4因子回归分析")
print("="*80)

# ============================================================================
# 1. 读取数据
# ============================================================================
print("\n步骤1: 读取数据")

# 读取策略收益数据
df_returns = pd.read_csv('/Users/xinyutan/Documents/量化投资/quant-project/results/tables/strategy_returns.csv')
df_returns['Date'] = pd.to_datetime(df_returns['Date'])
df_returns['Date_MonthEnd'] = df_returns['Date'] + pd.offsets.MonthEnd(0)
print(f"策略收益数据: {df_returns.shape}")

# 读取CH4因子数据
df_ch4 = pd.read_excel('/Users/xinyutan/Documents/量化投资/quant-project/data/processed/CH4_factors_monthly_202602.xlsx')
df_ch4['mnthdt'] = pd.to_datetime(df_ch4['mnthdt'], format='%Y%m%d')
df_ch4['Date_MonthEnd'] = df_ch4['mnthdt'] + pd.offsets.MonthEnd(0)
print(f"CH4因子数据: {df_ch4.shape}")

# ============================================================================
# 2. 数据对齐与合并
# ============================================================================
print("\n步骤2: 数据对齐与合并")

df_returns_selected = df_returns[['Date', 'Date_MonthEnd', 'Portfolio_Return']].copy()
df_returns_selected.columns = ['Date', 'Date_MonthEnd', 'Strategy_Return']

df_merged = pd.merge(df_returns_selected, 
                     df_ch4[['Date_MonthEnd', 'mktrf', 'SMB', 'VMG', 'PMO']], 
                     on='Date_MonthEnd', how='inner')
df_merged = df_merged.sort_values('Date').reset_index(drop=True)

print(f"合并后数据: {df_merged.shape}")
print(f"  日期范围: {df_merged['Date'].min().strftime('%Y-%m')} 至 {df_merged['Date'].max().strftime('%Y-%m')}")

# ============================================================================
# Step 4: CH-4全样本回归
# ============================================================================
print("\n" + "="*80)
print("Step 4: CH-4全样本回归")
print("="*80)

y = df_merged['Strategy_Return']
X_ch4 = df_merged[['mktrf', 'SMB', 'VMG', 'PMO']]
X_ch4 = sm.add_constant(X_ch4)

model_ch4 = sm.OLS(y, X_ch4).fit(cov_type='HC0')

print("\nCH-4回归结果:")
print(model_ch4.summary())

# 提取关键结果
print("\n" + "="*80)
print("CH-4关键指标")
print("="*80)

alpha_monthly = model_ch4.params['const']
alpha_annual = alpha_monthly * 12
alpha_tstat = model_ch4.tvalues['const']
alpha_pvalue = model_ch4.pvalues['const']

print(f"\n【Alpha (策略超额收益)】")
print(f"  月度 Alpha: {alpha_monthly*100:.4f}%")
print(f"  年化 Alpha: {alpha_annual*100:.2f}%")
print(f"  t 统计量:   {alpha_tstat:.2f}")
print(f"  p 值:       {alpha_pvalue:.4f}")
if alpha_pvalue < 0.01:
    print(f"  显著性:     *** 在1%水平显著")
elif alpha_pvalue < 0.05:
    print(f"  显著性:     ** 在5%水平显著")
elif alpha_pvalue < 0.1:
    print(f"  显著性:     * 在10%水平显著")
else:
    print(f"  显著性:     不显著")

print(f"\n【因子暴露 (Beta)】")
print(f"  {'因子':<12} {'Beta':>10} {'t统计量':>10} {'p值':>10} {'显著性':>8}")
print("-" * 60)

factors = ['mktrf', 'SMB', 'VMG', 'PMO']
factor_names = {'mktrf': '市场因子', 'SMB': '规模因子', 'VMG': '价值因子', 'PMO': '换手率因子'}

for factor in factors:
    beta = model_ch4.params[factor]
    tstat = model_ch4.tvalues[factor]
    pval = model_ch4.pvalues[factor]
    
    if pval < 0.01:
        sig = '***'
    elif pval < 0.05:
        sig = '**'
    elif pval < 0.1:
        sig = '*'
    else:
        sig = ''
    
    print(f"  {factor_names[factor]:<12} {beta:>10.4f} {tstat:>10.2f} {pval:>10.4f} {sig:>8}")

print(f"\n【模型拟合度】")
print(f"  R-squared:        {model_ch4.rsquared:.4f} ({model_ch4.rsquared*100:.2f}%)")
print(f"  Adjusted R-squared: {model_ch4.rsquared_adj:.4f}")

residual_std = np.std(model_ch4.resid) * np.sqrt(12)
print(f"  残差年化波动率:   {residual_std*100:.2f}%")
if residual_std > 0:
    print(f"  信息比率: {alpha_annual/residual_std:.2f}")

# ============================================================================
# Step 5: 5年滚动窗口归因
# ============================================================================
print("\n" + "="*80)
print("Step 5: 5年滚动窗口归因 (60个月)")
print("="*80)

window = 60  # 5年 = 60个月
rolling_results = []

for i in range(window, len(df_merged) + 1):
    window_data = df_merged.iloc[i-window:i]
    
    y_roll = window_data['Strategy_Return']
    X_roll = window_data[['mktrf', 'SMB', 'VMG', 'PMO']]
    X_roll = sm.add_constant(X_roll)
    
    try:
        model_roll = sm.OLS(y_roll, X_roll).fit(cov_type='HC0')
        
        rolling_results.append({
            'Date': window_data['Date'].iloc[-1],
            'Alpha': model_roll.params['const'],
            'Alpha_tstat': model_roll.tvalues['const'],
            'MKT_Beta': model_roll.params['mktrf'],
            'SMB_Beta': model_roll.params['SMB'],
            'VMG_Beta': model_roll.params['VMG'],
            'PMO_Beta': model_roll.params['PMO'],
            'R2': model_roll.rsquared,
            'Residual_Std': np.std(model_roll.resid)
        })
    except:
        pass

df_rolling = pd.DataFrame(rolling_results)
print(f"滚动窗口数量: {len(df_rolling)}")
print(f"  期间: {df_rolling['Date'].min().strftime('%Y-%m')} 至 {df_rolling['Date'].max().strftime('%Y-%m')}")

# 滚动统计
print("\n【滚动Alpha统计】")
print(f"  均值: {df_rolling['Alpha'].mean()*100:.4f}%")
print(f"  标准差: {df_rolling['Alpha'].std()*100:.4f}%")
print(f"  最小值: {df_rolling['Alpha'].min()*100:.4f}%")
print(f"  最大值: {df_rolling['Alpha'].max()*100:.4f}%")
print(f"  t>2的比例: {(df_rolling['Alpha_tstat'] > 2).mean()*100:.2f}%")

print("\n【滚动Beta统计】")
for factor, name in [('MKT_Beta', '市场'), ('SMB_Beta', '规模'), ('VMG_Beta', '价值'), ('PMO_Beta', '换手率')]:
    print(f"  {name} Beta: 均值={df_rolling[factor].mean():.4f}, 标准差={df_rolling[factor].std():.4f}")

print("\n【滚动R²统计】")
print(f"  均值: {df_rolling['R2'].mean():.4f}")
print(f"  最小值: {df_rolling['R2'].min():.4f}")
print(f"  最大值: {df_rolling['R2'].max():.4f}")

# ============================================================================
# 保存结果
# ============================================================================
print("\n" + "="*80)
print("保存结果")
print("="*80)

# 保存全样本回归数据
df_regression = df_merged[['Date', 'Strategy_Return', 'mktrf', 'SMB', 'VMG', 'PMO']].copy()
df_regression['const'] = 1.0
df_regression['CH4_Residuals'] = model_ch4.resid
df_regression.to_csv('/Users/xinyutan/Documents/量化投资/quant-project/results/tables/ch4_regression_data.csv', index=False, encoding='utf-8-sig')
print("CH-4回归数据已保存")

# 保存滚动窗口结果
df_rolling.to_csv('/Users/xinyutan/Documents/量化投资/quant-project/results/tables/ch4_rolling_attribution.csv', index=False, encoding='utf-8-sig')
print("滚动归因结果已保存")

# ============================================================================
# 绘制滚动归因图
# ============================================================================
print("\n绘制滚动归因图...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 图1: 滚动Alpha
ax1 = axes[0]
ax1.plot(df_rolling['Date'], df_rolling['Alpha'] * 100, 'b-', linewidth=1.5, label='Alpha')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax1.axhline(y=df_rolling['Alpha'].mean() * 100, color='r', linestyle='--', alpha=0.5, label=f'均值: {df_rolling["Alpha"].mean()*100:.2f}%')
ax1.fill_between(df_rolling['Date'], 0, df_rolling['Alpha'] * 100, 
                  where=(df_rolling['Alpha_tstat'] > 2), alpha=0.3, color='green', label='t>2 (显著)')
ax1.set_ylabel('月度 Alpha (%)')
ax1.set_title('CH-4 滚动Alpha (5年窗口)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: 滚动Beta
ax2 = axes[1]
ax2.plot(df_rolling['Date'], df_rolling['MKT_Beta'], label='MKT', linewidth=1.5)
ax2.plot(df_rolling['Date'], df_rolling['SMB_Beta'], label='SMB', linewidth=1.5)
ax2.plot(df_rolling['Date'], df_rolling['VMG_Beta'], label='VMG', linewidth=1.5)
ax2.plot(df_rolling['Date'], df_rolling['PMO_Beta'], label='PMO', linewidth=1.5)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
ax2.set_ylabel('Beta')
ax2.set_title('CH-4 滚动Beta (5年窗口)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3: 滚动R²
ax3 = axes[2]
ax3.plot(df_rolling['Date'], df_rolling['R2'], 'purple', linewidth=1.5)
ax3.axhline(y=df_rolling['R2'].mean(), color='r', linestyle='--', alpha=0.5, label=f'均值: {df_rolling["R2"].mean():.2f}')
ax3.set_ylabel('R²')
ax3.set_xlabel('日期')
ax3.set_title('CH-4 滚动R² (5年窗口)', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/xinyutan/Documents/量化投资/quant-project/results/figures/ch4_rolling_attribution.png', dpi=300, bbox_inches='tight')
print("滚动归因图已保存至: results/figures/ch4_rolling_attribution.png")

print("\n" + "="*80)
print("CH-4因子回归分析完成")
print("="*80)
