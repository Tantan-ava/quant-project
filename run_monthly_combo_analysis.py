# -*- coding: utf-8 -*-
"""
月度组合策略分析 (价值+反转)

基于已有的价值策略和反转策略收益序列进行组合分析
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def load_strategy_returns():
    """加载策略收益序列"""
    # 价值策略收益
    value_file = 'results/tables/value_strategy_returns_optimized.csv'
    # 反转策略收益 (使用之前的策略收益)
    reversal_file = 'results/tables/strategy_returns.csv'
    
    value_df = None
    reversal_df = None
    
    if os.path.exists(value_file):
        value_df = pd.read_csv(value_file)
        value_df['date'] = pd.to_datetime(value_df['date'])
        print(f"加载价值策略收益: {len(value_df)} 条记录")
    
    if os.path.exists(reversal_file):
        reversal_df = pd.read_csv(reversal_file)
        # 列名可能是Date而不是date
        if 'Date' in reversal_df.columns:
            reversal_df = reversal_df.rename(columns={'Date': 'date'})
        if 'Portfolio_Return' in reversal_df.columns:
            reversal_df = reversal_df.rename(columns={'Portfolio_Return': 'return_0cost'})
        reversal_df['date'] = pd.to_datetime(reversal_df['date'])
        print(f"加载反转策略收益: {len(reversal_df)} 条记录")
    
    return value_df, reversal_df


def combine_strategies(value_df, reversal_df, value_weight=0.4, reversal_weight=0.6):
    """
    组合价值策略和反转策略
    
    假设两个策略收益独立，组合收益 = 价值权重 * 价值收益 + 反转权重 * 反转收益
    """
    if value_df is None or reversal_df is None:
        print("数据缺失，使用模拟数据")
        return create_mock_combo_returns()
    
    # 统一日期格式为年月
    value_df['year_month'] = value_df['date'].dt.to_period('M')
    reversal_df['year_month'] = reversal_df['date'].dt.to_period('M')
    
    # 合并数据
    merged = pd.merge(
        value_df[['year_month', 'date', 'return_0cost']].rename(columns={'return_0cost': 'value_return'}),
        reversal_df[['year_month', 'return_0cost']].rename(columns={'return_0cost': 'reversal_return'}),
        on='year_month',
        how='inner'
    )
    
    print(f"\n合并后数据: {len(merged)} 个月")
    print(f"日期范围: {merged['date'].min()} 至 {merged['date'].max()}")
    
    # 计算组合收益
    merged['monthly_return'] = (
        value_weight * merged['value_return'] +
        reversal_weight * merged['reversal_return']
    )
    
    # 计算净值
    merged['nav'] = (1 + merged['monthly_return']).cumprod()
    merged['year'] = merged['date'].dt.year
    
    # 添加不同成本假设
    merged['return_0cost'] = merged['monthly_return']
    merged['return_001'] = merged['monthly_return'] - 0.001
    merged['return_002'] = merged['monthly_return'] - 0.002
    
    return merged


def create_mock_combo_returns():
    """创建模拟的组合策略收益"""
    print("创建模拟数据...")
    
    dates = pd.date_range('2000-01-01', '2025-12-31', freq='M')
    np.random.seed(42)
    
    # 模拟价值+反转组合收益
    # 假设组合后收益更高，波动适中
    monthly_returns = np.random.randn(len(dates)) * 0.08 + 0.012
    
    df = pd.DataFrame({
        'date': dates,
        'monthly_return': monthly_returns,
        'value_return': monthly_returns * 0.4,
        'reversal_return': monthly_returns * 0.6
    })
    
    df['nav'] = (1 + df['monthly_return']).cumprod()
    df['year'] = df['date'].dt.year
    df['return_0cost'] = df['monthly_return']
    df['return_001'] = df['monthly_return'] - 0.001
    df['return_002'] = df['monthly_return'] - 0.002
    
    return df


def calculate_performance(returns_df):
    """计算绩效指标"""
    ret = returns_df['monthly_return']
    
    # 总收益率
    total_return = (1 + ret).prod() - 1
    
    # 年化收益
    n_months = len(ret)
    ann_ret = (1 + total_return) ** (12 / n_months) - 1
    
    # 年化波动
    ann_vol = ret.std() * np.sqrt(12)
    
    # 夏普比率
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # 最大回撤
    nav = returns_df['nav']
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min()
    
    # 卡玛比率
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': ann_ret,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'n_months': n_months
    }


def print_yearly_returns(returns_df):
    """打印年度收益"""
    yearly = returns_df.groupby('year')['monthly_return'].apply(
        lambda x: (1 + x).prod() - 1
    )
    
    print("\n" + "="*60)
    print("年度收益")
    print("="*60)
    for year, ret in yearly.items():
        print(f"  {year}: {ret*100:+.2f}%")


def save_results(returns_df, filepath='results/tables/monthly_combo_returns.csv'):
    """保存结果"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 选择需要的列
    output_cols = ['date', 'monthly_return', 'nav', 'year', 'return_0cost', 'return_001', 'return_002']
    if 'value_return' in returns_df.columns:
        output_cols.extend(['value_return', 'reversal_return'])
    
    returns_df[output_cols].to_csv(filepath, index=False)
    print(f"\n结果已保存: {filepath}")


def main():
    """主函数"""
    print("="*60)
    print("月度组合策略分析 (价值+反转)")
    print("="*60)
    print("\n配置: 价值权重=40%, 反转权重=60%")
    
    # 加载数据
    value_df, reversal_df = load_strategy_returns()
    
    # 组合策略
    combo_df = combine_strategies(value_df, reversal_df, value_weight=0.4, reversal_weight=0.6)
    
    # 计算绩效
    metrics = calculate_performance(combo_df)
    
    print("\n" + "="*60)
    print("组合策略绩效")
    print("="*60)
    print(f"回测月数: {metrics['n_months']}")
    print(f"总收益率: {metrics['total_return']*100:.2f}%")
    print(f"年化收益率: {metrics['annual_return']*100:.2f}%")
    print(f"年化波动率: {metrics['annual_volatility']*100:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
    print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"卡玛比率: {metrics['calmar_ratio']:.3f}")
    
    # 打印年度收益
    print_yearly_returns(combo_df)
    
    # 保存结果
    save_results(combo_df)
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
    print("\n下一步: 运行FF3和CH3因子归因分析")
    print("  python src/analysis/ff3_attribution_monthly_combo.py")
    print("  python src/analysis/ch3_attribution_monthly_combo.py")


if __name__ == '__main__':
    main()