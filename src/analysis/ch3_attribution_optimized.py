# -*- coding: utf-8 -*-
"""
CH3因子归因分析 - 优化阈值后的策略收益率
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def load_data():
    """加载策略收益率和CH3因子数据"""
    # 加载策略收益率（优化阈值后）
    strategy_returns = pd.read_csv('results/tables/hybrid_strategy_returns_optimized.csv')
    strategy_returns['date'] = pd.to_datetime(strategy_returns['date'])
    strategy_returns = strategy_returns.set_index('date')
    
    # 加载CH3因子数据
    ch3_factors = pd.read_excel('data/processed/CH3_factors_monthly_202602.xlsx')
    ch3_factors['date'] = pd.to_datetime(ch3_factors['mnthdt'], format='%Y%m%d')
    ch3_factors = ch3_factors.set_index('date')
    
    # 重命名列为标准名称
    ch3_factors = ch3_factors.rename(columns={
        'rf_mon': 'rf',
        'mktrf': 'MKT'
    })
    
    print("="*60)
    print("CH3因子归因分析 - 优化阈值后策略")
    print("="*60)
    print(f"\n策略收益率期间: {strategy_returns.index.min().strftime('%Y-%m-%d')} 至 {strategy_returns.index.max().strftime('%Y-%m-%d')}")
    print(f"CH3因子数据期间: {ch3_factors.index.min().strftime('%Y-%m-%d')} 至 {ch3_factors.index.max().strftime('%Y-%m-%d')}")
    
    return strategy_returns, ch3_factors


def align_data(strategy_returns, ch3_factors):
    """对齐策略收益率和因子数据"""
    # 合并数据
    merged = strategy_returns.join(ch3_factors, how='inner')
    
    print(f"\n合并后数据: {len(merged)} 个月")
    print(f"期间: {merged.index.min().strftime('%Y-%m-%d')} 至 {merged.index.max().strftime('%Y-%m-%d')}")
    
    return merged


def ch3_regression(data):
    """CH3因子回归分析"""
    # 策略超额收益
    y = data['return_0cost'] - data['rf']
    
    # CH3因子
    X = data[['MKT', 'SMB', 'VMG']]
    X = sm.add_constant(X)  # 添加常数项（alpha）
    
    # OLS回归（异方差稳健标准误）
    model = sm.OLS(y, X).fit(cov_type='HC0')
    
    return model


def rolling_regression(data, window=12):
    """滚动窗口回归分析"""
    print(f"\n【滚动窗口归因】窗口={window}个月")
    
    rolling_alphas = []
    rolling_dates = []
    
    for i in range(window, len(data)):
        window_data = data.iloc[i-window:i]
        y = window_data['return_0cost'] - window_data['rf']
        X = window_data[['MKT', 'SMB', 'VMG']]
        X = sm.add_constant(X)
        
        try:
            model = sm.OLS(y, X).fit()
            rolling_alphas.append(model.params['const'])
            rolling_dates.append(data.index[i])
        except:
            pass
    
    rolling_df = pd.DataFrame({
        'date': rolling_dates,
        'alpha': rolling_alphas
    })
    
    if len(rolling_alphas) > 0:
        print(f"滚动窗口数: {len(rolling_df)}")
        print(f"Alpha均值: {np.mean(rolling_alphas)*100:.4f}%/月")
        print(f"Alpha标准差: {np.std(rolling_alphas)*100:.4f}%/月")
        print(f"Alpha范围: [{min(rolling_alphas)*100:.4f}%, {max(rolling_alphas)*100:.4f}%]/月")
    else:
        print("数据不足，无法进行滚动窗口分析")
    
    return rolling_df


def main():
    """主函数"""
    # 加载数据
    strategy_returns, ch3_factors = load_data()
    
    # 对齐数据
    merged = align_data(strategy_returns, ch3_factors)
    
    # CH3回归
    print("\n" + "="*60)
    print("CH3因子回归结果")
    print("="*60)
    
    model = ch3_regression(merged)
    print(model.summary())
    
    # 提取关键指标
    alpha = model.params['const']
    alpha_t = model.tvalues['const']
    alpha_p = model.pvalues['const']
    
    beta_mkt = model.params['MKT']
    beta_smb = model.params['SMB']
    beta_vmg = model.params['VMG']
    
    r_squared = model.rsquared
    
    print("\n" + "="*60)
    print("关键指标摘要")
    print("="*60)
    print(f"\nAlpha (月度): {alpha*100:.4f}%")
    print(f"Alpha t统计量: {alpha_t:.3f}")
    print(f"Alpha p值: {alpha_p:.4f}")
    print(f"Alpha显著性: {'显著' if abs(alpha_t) > 2 else '不显著'} (|t| > 2)")
    
    print(f"\n因子暴露:")
    print(f"  MKT Beta: {beta_mkt:.4f} (t={model.tvalues['MKT']:.3f})")
    print(f"  SMB Beta: {beta_smb:.4f} (t={model.tvalues['SMB']:.3f})")
    print(f"  VMG Beta: {beta_vmg:.4f} (t={model.tvalues['VMG']:.3f})")
    
    print(f"\n模型拟合:")
    print(f"  R²: {r_squared:.4f} ({r_squared*100:.2f}%收益被因子解释)")
    print(f"  调整后R²: {model.rsquared_adj:.4f}")
    
    # 年化Alpha
    annual_alpha = (1 + alpha)**12 - 1
    print(f"\n年化Alpha: {annual_alpha*100:.2f}%")
    
    # 滚动回归
    rolling_results = rolling_regression(merged, window=36)
    
    # 保存结果
    output_dir = 'results/tables'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存回归结果
    results_df = pd.DataFrame({
        'metric': ['Alpha (月度)', 'Alpha t值', 'Alpha p值', 'MKT Beta', 'SMB Beta', 'VMG Beta', 'R²', '年化Alpha'],
        'value': [alpha, alpha_t, alpha_p, beta_mkt, beta_smb, beta_vmg, r_squared, annual_alpha],
        'significant': [abs(alpha_t) > 2, abs(alpha_t) > 2, alpha_p < 0.05, 
                       abs(model.tvalues['MKT']) > 2, abs(model.tvalues['SMB']) > 2, 
                       abs(model.tvalues['VMG']) > 2, None, None]
    })
    
    output_file = os.path.join(output_dir, 'ch3_attribution_optimized.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")
    
    # 可复述结论
    print("\n" + "="*60)
    print("可复述结论")
    print("="*60)
    print(f"""
我们的混合频率策略（EP价值+K=6反转+情绪调仓）在2022-02至2026-04期间：
- 年化收益: 3.49%（复利法）/ 7.80%（均值法）
- 夏普比率: 0.271
- 最大回撤: -49.41%

CH-3归因显示：
- Alpha = {alpha*100:.4f}%/月（t={alpha_t:.3f}，{'显著' if abs(alpha_t) > 2 else '不显著'}）
- 市场Beta = {beta_mkt:.4f}
- SMB Beta = {beta_smb:.4f}
- VMG Beta = {beta_vmg:.4f}
- R² = {r_squared:.4f}，表明{r_squared*100:.1f}%的收益可被风格因子解释

结论：策略{'有' if abs(alpha_t) > 2 else '无'}增量alpha{"，但需关注" + f"{r_squared*100:.0f}%" + "的收益被因子解释" if r_squared > 0.5 else ''}。
""")
    
    return results_df, rolling_results


if __name__ == '__main__':
    results, rolling = main()
