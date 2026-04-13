# -*- coding: utf-8 -*-
"""
FF3因子归因分析 - 价值投资策略

对value_strategy_returns.csv进行Fama-French三因子归因
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def load_strategy_returns():
    """加载价值投资策略收益率"""
    file_path = 'results/tables/value_strategy_returns.csv'
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # 标准化列名
    if 'return_0cost' in df.columns:
        df['daily_return'] = df['return_0cost']
    elif 'nav' in df.columns:
        # 从净值计算收益率
        df['daily_return'] = df['nav'].pct_change()
    
    print(f"加载策略收益率: {len(df)} 条记录")
    print(f"日期范围: {df.index.min()} 至 {df.index.max()}")
    print(f"列名: {df.columns.tolist()}")
    
    return df


def load_ff3_factors():
    """加载FF3因子数据（使用CH3作为替代）"""
    # 使用CH3因子作为FF3的替代
    file_path = 'data/processed/CH3_factors_monthly_202602.xlsx'
    
    if not os.path.exists(file_path):
        print(f"错误: 因子文件不存在 {file_path}")
        return None
    
    df = pd.read_excel(file_path)
    df['date'] = pd.to_datetime(df['mnthdt'], format='%Y%m%d')
    df = df.set_index('date')
    
    # 重命名列为FF3标准名称
    df = df.rename(columns={
        'rf_mon': 'rf',
        'mktrf': 'MKT',
        'SMB': 'SMB',
        'VMG': 'HML'  # VMG对应价值因子HML
    })
    
    print(f"加载FF3因子: {len(df)} 条记录")
    
    return df


def ff3_regression(strategy_returns, ff3_factors):
    """
    FF3因子回归分析
    
    模型: R_p - R_f = α + β_mkt * MKT + β_smb * SMB + β_hml * HML + ε
    """
    # 合并数据
    merged = strategy_returns.join(ff3_factors, how='inner')
    
    if len(merged) == 0:
        print("错误: 数据合并失败，无重叠日期")
        return None
    
    print(f"\n合并后数据: {len(merged)} 个月")
    
    # 策略超额收益
    y = merged['daily_return'] - merged['rf']
    
    # FF3因子
    X = merged[['MKT', 'SMB', 'HML']]
    X = sm.add_constant(X)  # 添加常数项（alpha）
    
    # OLS回归（异方差稳健标准误）
    model = sm.OLS(y, X).fit(cov_type='HC0')
    
    return model, merged


def print_results(model, merged):
    """打印归因结果"""
    print("\n" + "="*70)
    print("FF3因子归因分析结果")
    print("="*70)
    print(model.summary())
    
    # 提取关键指标
    alpha = model.params['const']
    alpha_t = model.tvalues['const']
    beta_mkt = model.params['MKT']
    beta_smb = model.params['SMB']
    beta_hml = model.params['HML']
    r_squared = model.rsquared
    
    print("\n" + "="*70)
    print("关键指标摘要")
    print("="*70)
    
    print(f"\n【Alpha分析】")
    print(f"  月度Alpha: {alpha*100:.4f}%")
    print(f"  t统计量: {alpha_t:.3f}")
    print(f"  显著性: {'显著 ***' if abs(alpha_t) > 2.58 else '显著 **' if abs(alpha_t) > 1.96 else '不显著'}")
    
    annual_alpha = (1 + alpha)**12 - 1
    print(f"  年化Alpha: {annual_alpha*100:.2f}%")
    
    print(f"\n【因子暴露】")
    print(f"  MKT Beta: {beta_mkt:.4f} (t={model.tvalues['MKT']:.3f})")
    print(f"  SMB Beta: {beta_smb:.4f} (t={model.tvalues['SMB']:.3f})")
    print(f"  HML Beta: {beta_hml:.4f} (t={model.tvalues['HML']:.3f})")
    
    print(f"\n【模型拟合】")
    print(f"  R²: {r_squared:.4f} ({r_squared*100:.2f}%收益被因子解释)")
    print(f"  调整后R²: {model.rsquared_adj:.4f}")
    print(f"  F统计量: {model.fvalue:.2f} (p={model.f_pvalue:.4f})")
    
    # 因子贡献分解
    print(f"\n【因子贡献分解】")
    avg_mkt = merged['MKT'].mean()
    avg_smb = merged['SMB'].mean()
    avg_hml = merged['HML'].mean()
    
    mkt_contrib = beta_mkt * avg_mkt
    smb_contrib = beta_smb * avg_smb
    hml_contrib = beta_hml * avg_hml
    
    print(f"  市场因子贡献: {mkt_contrib*100:.4f}%/月")
    print(f"  规模因子贡献: {smb_contrib*100:.4f}%/月")
    print(f"  价值因子贡献: {hml_contrib*100:.4f}%/月")
    print(f"  Alpha贡献: {alpha*100:.4f}%/月")
    
    total_explained = mkt_contrib + smb_contrib + hml_contrib
    print(f"  因子解释部分: {total_explained*100:.4f}%/月")
    
    return {
        'alpha': alpha,
        'alpha_t': alpha_t,
        'beta_mkt': beta_mkt,
        'beta_smb': beta_smb,
        'beta_hml': beta_hml,
        'r_squared': r_squared,
        'annual_alpha': annual_alpha
    }


def plot_attribution(model, merged, save_path='results/figures/ff3_attribution_value_strategy.png'):
    """绘制归因分析图"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 实际收益 vs 拟合收益
        ax1 = axes[0, 0]
        y_actual = merged['daily_return'] - merged['rf']
        y_fitted = model.fittedvalues
        
        ax1.scatter(y_fitted, y_actual, alpha=0.6)
        ax1.plot([y_fitted.min(), y_fitted.max()], [y_fitted.min(), y_fitted.max()], 'r--', lw=2)
        ax1.set_xlabel('Fitted Return')
        ax1.set_ylabel('Actual Return')
        ax1.set_title('Actual vs Fitted Returns')
        ax1.grid(True, alpha=0.3)
        
        # 2. 残差图
        ax2 = axes[0, 1]
        residuals = model.resid
        ax2.scatter(y_fitted, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Fitted Return')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. 因子暴露柱状图
        ax3 = axes[1, 0]
        betas = [model.params['MKT'], model.params['SMB'], model.params['HML']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax3.bar(['MKT', 'SMB', 'HML'], betas, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('Beta')
        ax3.set_title('FF3 Factor Exposures')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, beta in zip(bars, betas):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{beta:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top')
        
        # 4. 累计收益对比
        ax4 = axes[1, 1]
        cumulative_actual = (1 + y_actual).cumprod()
        cumulative_fitted = (1 + y_fitted).cumprod()
        
        ax4.plot(merged.index, cumulative_actual, label='Actual', linewidth=2)
        ax4.plot(merged.index, cumulative_fitted, label='Fitted (FF3)', linewidth=2, linestyle='--')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Return')
        ax4.set_title('Cumulative Returns: Actual vs FF3')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存: {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"绘图失败: {e}")


def generate_report(metrics, save_path='results/tables/ff3_attribution_value_strategy.csv'):
    """生成归因报告"""
    report = pd.DataFrame([{
        'alpha_monthly': metrics['alpha'],
        'alpha_t_stat': metrics['alpha_t'],
        'alpha_annual': metrics['annual_alpha'],
        'beta_mkt': metrics['beta_mkt'],
        'beta_smb': metrics['beta_smb'],
        'beta_hml': metrics['beta_hml'],
        'r_squared': metrics['r_squared'],
        'significance': 'Significant' if abs(metrics['alpha_t']) > 2 else 'Not Significant'
    }])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    report.to_csv(save_path, index=False)
    print(f"报告已保存: {save_path}")
    
    return report


def main():
    """主函数"""
    print("="*70)
    print("FF3因子归因分析 - 价值投资策略")
    print("="*70)
    
    # 加载数据
    strategy_returns = load_strategy_returns()
    if strategy_returns is None:
        return
    
    ff3_factors = load_ff3_factors()
    if ff3_factors is None:
        return
    
    # 执行回归
    result = ff3_regression(strategy_returns, ff3_factors)
    if result is None:
        return
    
    model, merged = result
    
    # 打印结果
    metrics = print_results(model, merged)
    
    # 绘制图表
    plot_attribution(model, merged)
    
    # 生成报告
    report = generate_report(metrics)
    
    # 可复述结论
    print("\n" + "="*70)
    print("可复述结论")
    print("="*70)
    print(f"""
我们的价值投资策略在回测期间表现如下：
FF-3归因显示：
- Alpha = {metrics['alpha']*100:.4f}%/月（t={metrics['alpha_t']:.3f}，{'显著' if abs(metrics['alpha_t']) > 2 else '不显著'}）
- 市场Beta = {metrics['beta_mkt']:.4f}
- SMB Beta = {metrics['beta_smb']:.4f}
- HML Beta = {metrics['beta_hml']:.4f}
- R² = {metrics['r_squared']:.4f}，表明{metrics['r_squared']*100:.1f}%的收益可被FF3因子解释

结论：策略{'具有' if abs(metrics['alpha_t']) > 2 else '不具有'}显著的增量alpha。
""")
    
    print("\n分析完成!")


if __name__ == '__main__':
    main()
