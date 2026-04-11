# -*- coding: utf-8 -*-
"""
CH3因子回归分析
策略超额收益 = Alpha + Beta1*MKT + Beta2*SMB + Beta3*VMG + epsilon
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

print("="*80)
print("CH3因子回归分析")
print("="*80)

# 读取回归数据
data = pd.read_csv('/Users/xinyutan/Documents/量化投资/quant-project/results/tables/regression_data.csv')
data['Date'] = pd.to_datetime(data['Date'])

print(f"\n数据样本: {len(data)} 个月")
print(f"数据范围: {data['Date'].min().strftime('%Y-%m')} 至 {data['Date'].max().strftime('%Y-%m')}")

# 策略超额收益 (因变量)
y = data['Strategy_Excess_Return']

# CH-3 因子 (自变量)
X = data[['mktrf', 'SMB', 'VMG']]
X = sm.add_constant(X)  # 添加常数项（alpha）

# OLS 回归
model = sm.OLS(y, X).fit(cov_type='HC0')  # 异方差稳健标准误

print("\n" + "="*80)
print("回归结果")
print("="*80)
print(model.summary())

# 提取关键结果
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

factors = ['mktrf', 'SMB', 'VMG']
factor_names = {'mktrf': 'MKT', 'SMB': 'SMB', 'VMG': 'VMG'}

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
    
    print(f"  {factor_names[factor]:<10} {beta:>10.4f} {tstat:>10.2f} {pval:>10.4f} {sig:>8}")

# 模型拟合度
print(f"\n【模型拟合度】")
print(f"  R-squared:        {model.rsquared:.4f}")
print(f"  Adjusted R-squared: {model.rsquared_adj:.4f}")
print(f"  说明: 因子解释了策略收益变动的 {model.rsquared*100:.2f}%")

# 风险调整后的Alpha
print(f"\n【风险分析】")
residual_std = np.std(model.resid) * np.sqrt(12)  # 年化残差波动率
print(f"  残差年化波动率:   {residual_std*100:.2f}%")
print(f"  信息比率 (Alpha/残差波动): {alpha_annual/residual_std:.2f}")

# 因子贡献分解
print(f"\n【因子收益贡献 (月度平均)】")
avg_factors = data[['mktrf', 'SMB', 'VMG']].mean()
for factor in factors:
    contribution = model.params[factor] * avg_factors[factor]
    print(f"  {factor_names[factor]}: {contribution*100:.4f}%")

print("\n" + "="*80)
print("回归分析完成")
print("="*80)
