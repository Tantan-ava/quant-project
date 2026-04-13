# -*- coding: utf-8 -*-
"""
优化权重价值投资策略CH3因子归因分析

权重配置: EP=80%, DP=20%, BP=SP=0%
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def load_strategy_returns():
    """加载优化后的策略收益序列"""
    file_path = 'results/tables/value_strategy_returns_optimized.csv'
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    
    print(f"加载策略收益率: {len(df)} 条记录")
    print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
    
    return df


def load_ch3_factors():
    """加载CH3因子数据"""
    file_path = 'data/processed/CH3_factors_monthly_202602.xlsx'
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return None
    
    df = pd.read_excel(file_path)
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['mnthdt'], format='%Y%m%d')
    df['year_month'] = df['date'].dt.to_period('M')
    
    # 重命名列
    df['MKT'] = df['mktrf']
    df['rf'] = df['rf_mon']
    df['VMG'] = df['VMG']  # 价值因子
    
    print(f"加载CH3因子: {len(df)} 条记录")
    print(f"因子日期范围: {df['date'].min()} 至 {df['date'].max()}")
    
    return df


def ch3_regression(strategy_returns, ch3_factors):
    """
    CH3因子回归分析
    
    模型: R_p - R_f = α + β_mkt * MKT + β_smb * SMB + β_vmg * VMG + ε
    """
    # 合并数据
    merged = strategy_returns.merge(ch3_factors, on='year_month', how='inner')
    
    if len(merged) == 0:
        print("错误: 数据合并失败，无重叠日期")
        return None
    
    print(f"\n合并后数据: {len(merged)} 个月")
    
    # 策略超额收益
    y = merged['return_0cost'] - merged['rf']
    
    # CH3因子
    X = merged[['MKT', 'SMB', 'VMG']]
    X = sm.add_constant(X)  # 添加常数项（alpha）
    
    # OLS回归（异方差稳健标准误）
    model = sm.OLS(y, X).fit(cov_type='HC0')
    
    return model, merged


def calculate_factor_contribution(model, merged_df):
    """计算各因子贡献"""
    # 因子均值
    mkt_mean = merged_df['MKT'].mean()
    smb_mean = merged_df['SMB'].mean()
    vmg_mean = merged_df['VMG'].mean()
    
    # 因子贡献
    contributions = {
        'MKT': model.params['MKT'] * mkt_mean,
        'SMB': model.params['SMB'] * smb_mean,
        'VMG': model.params['VMG'] * vmg_mean,
        'Alpha': model.params['const']
    }
    
    return contributions


def main():
    """主函数"""
    print("="*70)
    print("优化权重价值投资策略CH3因子归因分析")
    print("="*70)
    print("\n权重配置: EP=80%, DP=20%, BP=SP=0%")
    
    # 加载数据
    strategy_returns = load_strategy_returns()
    ch3_factors = load_ch3_factors()
    
    if strategy_returns is None or ch3_factors is None:
        print("数据加载失败")
        return
    
    # CH3回归
    print("\n" + "="*70)
    print("CH3因子回归结果")
    print("="*70)
    
    model, merged = ch3_regression(strategy_returns, ch3_factors)
    
    if model is None:
        return
    
    # 输出回归结果
    print(f"\n{'指标':<15} {'系数':>10} {'标准误':>10} {'t统计量':>10} {'p值':>10} {'显著性':>8}")
    print("-"*70)
    
    params = ['const', 'MKT', 'SMB', 'VMG']
    names = ['Alpha', 'MKT Beta', 'SMB Beta', 'VMG Beta']
    
    for param, name in zip(params, names):
        coef = model.params[param]
        std_err = model.bse[param]
        t_stat = model.tvalues[param]
        p_val = model.pvalues[param]
        
        if p_val < 0.01:
            sig = "***"
        elif p_val < 0.05:
            sig = "**"
        elif p_val < 0.10:
            sig = "*"
        else:
            sig = ""
        
        print(f"{name:<15} {coef:>10.4f} {std_err:>10.4f} {t_stat:>10.3f} {p_val:>10.4f} {sig:>8}")
    
    # 模型拟合
    print("\n" + "="*70)
    print("模型拟合")
    print("="*70)
    print(f"R²: {model.rsquared:.4f} ({model.rsquared*100:.2f}%)")
    print(f"调整后R²: {model.rsquared_adj:.4f}")
    print(f"F统计量: {model.fvalue:.2f} (p={model.f_pvalue:.4f})")
    
    # 因子贡献分解
    print("\n" + "="*70)
    print("因子贡献分解")
    print("="*70)
    
    contributions = calculate_factor_contribution(model, merged)
    
    print(f"\n{'因子':<15} {'月度贡献':>12} {'年化贡献':>12}")
    print("-"*45)
    for factor, contrib in contributions.items():
        annual_contrib = contrib * 12
        print(f"{factor:<15} {contrib:>11.4f}% {annual_contrib:>11.2f}%")
    
    total_monthly = sum(contributions.values())
    total_annual = total_monthly * 12
    print("-"*45)
    print(f"{'合计':<15} {total_monthly:>11.4f}% {total_annual:>11.2f}%")
    
    # 可复述结论
    print("\n" + "="*70)
    print("可复述结论")
    print("="*70)
    
    alpha = model.params['const']
    alpha_t = model.tvalues['const']
    alpha_p = model.pvalues['const']
    
    significance = "高度显著***" if alpha_p < 0.01 else "显著**" if alpha_p < 0.05 else "边际显著*" if alpha_p < 0.10 else "不显著"
    
    print(f"""
我们的优化权重价值投资策略(EP=80%, DP=20%)在2000-2025年期间：
- 年化Alpha约{alpha*12:.2f}%（t={alpha_t:.2f}，{significance}）
- 市场Beta={model.params['MKT']:.3f}，SMB Beta={model.params['SMB']:.3f}，VMG Beta={model.params['VMG']:.3f}
- R²={model.rsquared:.2%}，表明{model.rsquared*100:.1f}%的收益可被CH3因子解释
- 策略{'具有' if alpha_p < 0.10 else '不具有'}显著的增量alpha能力
""")
    
    # 保存结果
    output_dir = 'results/tables'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'alpha_monthly': alpha,
        'alpha_annual': alpha * 12,
        'alpha_t': alpha_t,
        'alpha_p': alpha_p,
        'mkt_beta': model.params['MKT'],
        'smb_beta': model.params['SMB'],
        'vmg_beta': model.params['VMG'],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'{output_dir}/ch3_attribution_value_optimized.csv', index=False)
    print(f"\n结果已保存: {output_dir}/ch3_attribution_value_optimized.csv")


if __name__ == '__main__':
    main()