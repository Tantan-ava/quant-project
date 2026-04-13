# -*- coding: utf-8 -*-
"""
月度组合策略标准FF3因子归因分析

使用 ff3_regression_data.csv 中的标准FF3因子（HML基于BP构建）
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def load_strategy_returns():
    """加载组合策略收益序列"""
    file_path = 'results/tables/monthly_combo_returns.csv'
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"加载策略收益率: {len(df)} 条记录")
    print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
    
    return df


def load_ff3_factors():
    """加载标准FF3因子数据（来自ff3_regression_data.csv）"""
    file_path = 'results/tables/ff3_regression_data.csv'
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['Date'])
    
    # 重命名列以匹配
    df['strategy_return'] = df['Strategy_Return']
    
    print(f"加载标准FF3因子: {len(df)} 条记录")
    print(f"因子日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"因子列: MKT, SMB, HML（标准FF3，HML基于BP构建）")
    
    return df


def ff3_regression(strategy_returns, ff3_factors):
    """FF3因子回归分析"""
    # 统一日期格式为年月
    strategy_returns['year_month'] = strategy_returns['date'].dt.to_period('M')
    ff3_factors['year_month'] = ff3_factors['date'].dt.to_period('M')
    
    merged = strategy_returns.merge(ff3_factors, on='year_month', how='inner')
    
    if len(merged) == 0:
        print("错误: 数据合并失败，无重叠日期")
        return None
    
    print(f"\n合并后数据: {len(merged)} 个月")
    
    # 使用策略收益率减去无风险利率（这里简化为直接使用超额收益）
    # 注意：ff3_regression_data.csv中的Strategy_Return已经是超额收益
    y = merged['return_0cost'] - merged.get('rf', 0)  # 如果没有rf列，假设为0
    
    # 如果y的均值明显大于merged['strategy_return']，可能rf已经包含
    if 'rf' not in merged.columns:
        print("注意: 无风险利率数据缺失，使用原始收益率")
        y = merged['return_0cost']
    
    X = merged[['MKT', 'SMB', 'HML']]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit(cov_type='HC0')
    
    return model, merged


def main():
    """主函数"""
    print("="*70)
    print("月度组合策略标准FF3因子归因分析")
    print("="*70)
    print("\n策略配置: 价值(40%) + 反转(60%)")
    print("因子来源: ff3_regression_data.csv（标准FF3，HML基于BP构建）")
    
    strategy_returns = load_strategy_returns()
    ff3_factors = load_ff3_factors()
    
    if strategy_returns is None or ff3_factors is None:
        print("数据加载失败")
        return
    
    print("\n" + "="*70)
    print("标准FF3因子回归结果")
    print("="*70)
    
    model, merged = ff3_regression(strategy_returns, ff3_factors)
    
    if model is None:
        return
    
    # 输出回归结果
    print(f"\n{'指标':<15} {'系数':>10} {'标准误':>10} {'t统计量':>10} {'p值':>10} {'显著性':>8}")
    print("-"*70)
    
    params = ['const', 'MKT', 'SMB', 'HML']
    names = ['Alpha', 'MKT Beta', 'SMB Beta', 'HML Beta']
    
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
    
    # 可复述结论
    print("\n" + "="*70)
    print("可复述结论（标准FF3归因）")
    print("="*70)
    
    alpha = model.params['const']
    alpha_t = model.tvalues['const']
    alpha_p = model.pvalues['const']
    
    significance = "高度显著***" if alpha_p < 0.01 else "显著**" if alpha_p < 0.05 else "边际显著*" if alpha_p < 0.10 else "不显著"
    
    print(f"""
我们的月度组合策略(价值40%+反转60%)在2005-2025年期间（标准FF3归因）：
- 年化Alpha约{alpha*12:.2f}%（t={alpha_t:.2f}，{significance}）
- 市场Beta={model.params['MKT']:.3f}，SMB Beta={model.params['SMB']:.3f}，HML Beta={model.params['HML']:.3f}
- R²={model.rsquared:.2%}，表明{model.rsquared*100:.1f}%的收益可被标准FF3因子解释
- HML因子基于账面市值比(BP)构建，符合经典Fama-French定义
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
        'hml_beta': model.params['HML'],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'factor_source': 'ff3_regression_data.csv (标准FF3, HML基于BP)'
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'{output_dir}/ff3_attribution_monthly_combo_correct.csv', index=False)
    print(f"\n结果已保存: {output_dir}/ff3_attribution_monthly_combo_correct.csv")


if __name__ == '__main__':
    main()
