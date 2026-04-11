# -*- coding: utf-8 -*-
"""
FF-3因子回归分析
策略超额收益 = Alpha + Beta1*MKT + Beta2*SMB + Beta3*HML + epsilon
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FF-3因子回归分析")
print("="*80)

# ============================================================================
# 1. 读取数据
# ============================================================================
print("\n步骤1: 读取数据")

# 读取策略收益数据
df_returns = pd.read_csv('/Users/xinyutan/Documents/量化投资/quant-project/results/tables/strategy_returns.csv')
df_returns['Date'] = pd.to_datetime(df_returns['Date'])
# 策略收益日期是月初，转换为月末格式用于对齐
df_returns['Date_MonthEnd'] = df_returns['Date'] + pd.offsets.MonthEnd(0)
print(f"策略收益数据: {df_returns.shape}")
print(f"  日期范围: {df_returns['Date'].min().strftime('%Y-%m-%d')} 至 {df_returns['Date'].max().strftime('%Y-%m-%d')}")

# 读取FF-3因子数据 (跳过前3行标题)
df_ff3 = pd.read_excel('/Users/xinyutan/Documents/量化投资/quant-project/data/processed/STK_MKT_THRFACMONTH.xlsx', skiprows=3)
df_ff3.columns = ['MarkettypeID', 'TradingMonth', 'RiskPremium1', 'RiskPremium2', 'SMB1', 'SMB2', 'HML1', 'HML2']

# 转换日期格式
df_ff3['TradingMonth'] = pd.to_datetime(df_ff3['TradingMonth'])
# FF-3因子日期是月初格式，转换为月末
df_ff3['Date_MonthEnd'] = df_ff3['TradingMonth'] + pd.offsets.MonthEnd(0)

# 选择沪深A股市场 (P9709: 沪深A股和创业板)
df_ff3_selected = df_ff3[df_ff3['MarkettypeID'] == 'P9709'].copy()

# 重命名列便于分析
df_ff3_selected = df_ff3_selected.rename(columns={
    'RiskPremium1': 'MKT',  # 市场风险溢价 (流通市值加权)
    'SMB1': 'SMB',          # 市值因子 (流通市值加权)
    'HML1': 'HML'           # 账面市值比因子 (流通市值加权)
})

# 转换数值类型
df_ff3_selected['MKT'] = pd.to_numeric(df_ff3_selected['MKT'], errors='coerce')
df_ff3_selected['SMB'] = pd.to_numeric(df_ff3_selected['SMB'], errors='coerce')
df_ff3_selected['HML'] = pd.to_numeric(df_ff3_selected['HML'], errors='coerce')

# 删除缺失值
df_ff3_selected = df_ff3_selected.dropna(subset=['MKT', 'SMB', 'HML'])

print(f"FF-3因子数据 (P9709): {df_ff3_selected.shape}")
print(f"  日期范围: {df_ff3_selected['TradingMonth'].min().strftime('%Y-%m-%d')} 至 {df_ff3_selected['TradingMonth'].max().strftime('%Y-%m-%d')}")

# ============================================================================
# 2. 数据对齐与合并
# ============================================================================
print("\n步骤2: 数据对齐与合并")

# 选择策略收益列
df_returns_selected = df_returns[['Date', 'Date_MonthEnd', 'Portfolio_Return']].copy()
df_returns_selected.columns = ['Date', 'Date_MonthEnd', 'Strategy_Return']

# 合并数据 - 使用月末日期对齐
df_merged = pd.merge(df_returns_selected, df_ff3_selected[['Date_MonthEnd', 'MKT', 'SMB', 'HML']], 
                     on='Date_MonthEnd', how='inner')
df_merged = df_merged.sort_values('Date').reset_index(drop=True)

print(f"合并后数据: {df_merged.shape}")
print(f"  日期范围: {df_merged['Date'].min().strftime('%Y-%m')} 至 {df_merged['Date'].max().strftime('%Y-%m')}")

# ============================================================================
# 3. 计算超额收益
# ============================================================================
print("\n步骤3: 计算超额收益")

# 策略超额收益 (因变量)
y = df_merged['Strategy_Return']  # 策略收益

# FF-3 因子 (自变量)
X = df_merged[['MKT', 'SMB', 'HML']]
X = sm.add_constant(X)  # 添加常数项（alpha）

print(f"样本量: {len(y)} 个月")
print(f"策略平均月收益: {y.mean()*100:.4f}%")
print(f"MKT平均: {df_merged['MKT'].mean()*100:.4f}%")
print(f"SMB平均: {df_merged['SMB'].mean()*100:.4f}%")
print(f"HML平均: {df_merged['HML'].mean()*100:.4f}%")

# ============================================================================
# 4. OLS 回归
# ============================================================================
print("\n步骤4: OLS回归")

model = sm.OLS(y, X).fit(cov_type='HC0')  # 异方差稳健标准误

print("\n" + "="*80)
print("回归结果")
print("="*80)
print(model.summary())

# ============================================================================
# 5. 提取关键结果
# ============================================================================
print("\n" + "="*80)
print("关键指标摘要")
print("="*80)

# Alpha (年化)
alpha_monthly = model.params['const']
alpha_annual = alpha_monthly * 12
alpha_tstat = model.tvalues['const']
alpha_pvalue = model.pvalues['const']

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

# 因子暴露
print(f"\n【因子暴露 (Beta)】")
print(f"  {'因子':<10} {'Beta':>10} {'t统计量':>10} {'p值':>10} {'显著性':>8}")
print("-" * 60)

factors = ['MKT', 'SMB', 'HML']
factor_names = {'MKT': '市场因子', 'SMB': '市值因子', 'HML': '价值因子'}

for factor in factors:
    beta = model.params[factor]
    tstat = model.tvalues[factor]
    pval = model.pvalues[factor]
    
    if pval < 0.01:
        sig = '***'
    elif pval < 0.05:
        sig = '**'
    elif pval < 0.1:
        sig = '*'
    else:
        sig = ''
    
    print(f"  {factor:<10} {beta:>10.4f} {tstat:>10.2f} {pval:>10.4f} {sig:>8}")

# 模型拟合度
print(f"\n【模型拟合度】")
print(f"  R-squared:        {model.rsquared:.4f}")
print(f"  Adjusted R-squared: {model.rsquared_adj:.4f}")
print(f"  说明: 因子解释了策略收益变动的 {model.rsquared*100:.2f}%")

# 风险调整后的Alpha
print(f"\n【风险分析】")
residual_std = np.std(model.resid) * np.sqrt(12)  # 年化残差波动率
print(f"  残差年化波动率:   {residual_std*100:.2f}%")
if residual_std > 0:
    print(f"  信息比率 (Alpha/残差波动): {alpha_annual/residual_std:.2f}")

# 因子贡献分解
print(f"\n【因子收益贡献 (月度平均)】")
avg_factors = df_merged[['MKT', 'SMB', 'HML']].mean()
for factor in factors:
    contribution = model.params[factor] * avg_factors[factor]
    print(f"  {factor_names[factor]}: {contribution*100:.4f}%")

# ============================================================================
# 6. 保存回归数据
# ============================================================================
print("\n步骤6: 保存回归数据")

df_regression = df_merged[['Date', 'Strategy_Return', 'MKT', 'SMB', 'HML']].copy()
df_regression['const'] = 1.0
df_regression['Residuals'] = model.resid

# 保存为CSV
output_path = '/Users/xinyutan/Documents/量化投资/quant-project/results/tables/ff3_regression_data.csv'
df_regression.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"FF-3回归数据已保存至: {output_path}")

print("\n" + "="*80)
print("FF-3因子回归分析完成")
print("="*80)
