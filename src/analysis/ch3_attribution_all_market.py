# -*- coding: utf-8 -*-
"""
全市场策略CH3因子归因分析

使用 CH3_factors_monthly_202602.xlsx 中的中国三因子（VMG基于EP构建）
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def load_strategy_returns():
    """加载全市场策略收益序列"""
    file_path = 'results/tables/monthly_combo_all_market_returns.csv'
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
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
    
    # 处理日期列 mnthdt 格式为 YYYYMMDD
    if 'mnthdt' in df.columns:
        df['date'] = pd.to_datetime(df['mnthdt'].astype(str), format='%Y%m%d')
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        # 尝试第一列
        df['date'] = pd.to_datetime(df.iloc[:, 0])
    
    print(f"加载CH3因子: {len(df)} 条记录")
    print(f"因子日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"因子列: mktrf, SMB, VMG（中国三因子，VMG基于EP构建）")
    
    return df


def ch3_regression(strategy_returns, ch3_factors):
    """CH3因子回归分析"""
    # 统一日期格式为年月
    strategy_returns['year_month'] = strategy_returns['date'].dt.to_period('M')
    ch3_factors['year_month'] = ch3_factors['date'].dt.to_period('M')
    
    merged = strategy_returns.merge(ch3_factors, on='year_month', how='inner')
    
    if len(merged) == 0:
        print("错误: 数据合并失败，无重叠日期")
        return None
    
    print(f"\n合并后数据: {len(merged)} 个月")
    
    # 使用策略收益率
    y = merged['monthly_return']
    
    # CH3因子列名
    factor_cols = ['mktrf', 'SMB', 'VMG']
    
    # 检查列是否存在
    available_cols = [c for c in factor_cols if c in merged.columns]
    if len(available_cols) < 3:
        print(f"警告: 只找到 {available_cols} 因子列")
        print(f"可用列: {merged.columns.tolist()}")
    
    X = merged[available_cols]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit(cov_type='HC0')
    
    return model, merged


def main():
    """主函数"""
    print("="*70)
    print("全市场策略CH3因子归因分析")
    print("="*70)
    print("\n策略配置: 价值(40%) + 反转(60%) - 全市场股票池")
    print("因子来源: CH3_factors_monthly_202602.xlsx（中国三因子，VMG基于EP构建）")
    
    strategy_returns = load_strategy_returns()
    ch3_factors = load_ch3_factors()
    
    if strategy_returns is None or ch3_factors is None:
        print("数据加载失败")
        return
    
    print("\n" + "="*70)
    print("CH3因子回归结果")
    print("="*70)
    
    model, merged = ch3_regression(strategy_returns, ch3_factors)
    
    if model is None:
        return
    
    # 输出回归结果
    print(f"\n{'指标':<15} {'系数':>10} {'标准误':>10} {'t统计量':>10} {'p值':>10} {'显著性':>8}")
    print("-"*70)
    
    # 获取参数名
    params = model.params.index.tolist()
    
    for param in params:
        coef = model.params[param]
        std_err = model.bse[param]
        t_stat = model.tvalues[param]
        p_val = model.pvalues[param]
        
        # 映射显示名称
        name_map = {
            'const': 'Alpha',
            'mktrf': 'MKT Beta',
            'MKT': 'MKT Beta',
            'SMB': 'SMB Beta',
            'VMG': 'VMG Beta',
            'HML': 'HML Beta'
        }
        name = name_map.get(param, param)
        
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
    
    # 可复述结论
    print("\n" + "="*70)
    print("可复述结论（CH3归因 - 全市场）")
    print("="*70)
    
    alpha = model.params['const']
    alpha_t = model.tvalues['const']
    alpha_p = model.pvalues['const']
    
    significance = "高度显著***" if alpha_p < 0.01 else "显著**" if alpha_p < 0.05 else "边际显著*" if alpha_p < 0.10 else "不显著"
    
    # 获取各因子beta
    mkt_beta = model.params.get('mktrf', model.params.get('MKT', 0))
    smb_beta = model.params.get('SMB', 0)
    vmg_beta = model.params.get('VMG', model.params.get('HML', 0))
    
    print(f"""
我们的月度组合策略(价值40%+反转60%，全市场股票池)在2006-2025年期间（CH3归因）：
- 年化Alpha约{alpha*12:.2f}%（t={alpha_t:.2f}，{significance}）
- 市场Beta={mkt_beta:.3f}，SMB Beta={smb_beta:.3f}，VMG Beta={vmg_beta:.3f}
- R²={model.rsquared:.2%}，表明{model.rsquared*100:.1f}%的收益可被CH3因子解释
- VMG因子基于盈利收益率(EP)构建（Liu et al. 2019）
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
        'mkt_beta': mkt_beta,
        'smb_beta': smb_beta,
        'vmg_beta': vmg_beta,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'factor_source': 'CH3_factors_monthly_202602.xlsx (中国三因子, VMG基于EP)',
        'universe': '全市场股票池'
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'{output_dir}/ch3_attribution_all_market.csv', index=False)
    print(f"\n结果已保存: {output_dir}/ch3_attribution_all_market.csv")


if __name__ == '__main__':
    main()