# -*- coding: utf-8 -*-
"""
使用最优参数进行混合频率策略回测
生成收益率序列用于因子模型检验
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest.hybrid_backtester import HybridBacktester


def run_backtest_with_optimal_params():
    """
    使用最优参数执行回测
    """
    # 加载最优参数
    config_path = 'config/hybrid_strategy_optimized_full.json'
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        params = config['optimized_params']
        print("已加载最优参数配置")
    else:
        print("未找到配置文件，使用默认参数")
        params = {
            'ep_weight': 0.4,
            'reversal_weight': 0.6,
            'top_k': 50,
            'osa_threshold_extreme': -2.5,
            'osa_threshold_panic': -1.5,
            'osa_threshold_greed': 1.5,
            'scalar_extreme': 1.5,
            'scalar_greed': 0.5
        }
    
    print("\n最优参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 创建回测引擎
    print("\n初始化回测引擎...")
    backtester = HybridBacktester(
        daily_returns_path='data/raw/TRD-daily.csv',
        sentiment_path='data/processed/daily_sentiment_index.csv',
        start_date='2022-02-01',
        end_date='2026-04-10',
        initial_capital=1e8,
        **params
    )
    
    # 执行回测
    print("\n执行回测...")
    results = backtester.run()
    
    # 获取绩效指标
    metrics = backtester.get_performance_metrics()
    
    print("\n" + "="*60)
    print("回测绩效指标")
    print("="*60)
    print(f"总收益率: {metrics['total_return']*100:.2f}%")
    print(f"年化收益: {metrics['annual_return']*100:.2f}%")
    print(f"年化波动: {metrics['annual_volatility']*100:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
    print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"平均仓位系数: {metrics['avg_position_scalar']:.2f}")
    print("="*60)
    
    # 生成收益率序列
    returns_df = pd.DataFrame({
        'date': results.index,
        'daily_return': results['daily_return'].values,
        'portfolio_value': results['portfolio_value'].values,
        'position_scalar': results['position_scalar'].values,
        'sentiment_score': results['sentiment_score'].values
    })
    
    # 保存日度收益率
    output_dir = 'results/tables'
    os.makedirs(output_dir, exist_ok=True)
    
    daily_returns_file = os.path.join(output_dir, 'hybrid_strategy_daily_returns.csv')
    returns_df.to_csv(daily_returns_file, index=False)
    print(f"\n日度收益率序列已保存至: {daily_returns_file}")
    
    # 生成月度收益率（用于因子回归）
    results_copy = results.copy()
    results_copy['month'] = results_copy.index.to_period('M')
    
    # 计算月度收益率
    monthly_returns = results_copy.groupby('month').apply(
        lambda x: (1 + x['daily_return']).prod() - 1
    )
    
    # 获取月末组合价值
    monthly_end_value = results_copy.groupby('month')['portfolio_value'].last()
    monthly_start_value = results_copy.groupby('month')['portfolio_value'].first()
    
    monthly_df = pd.DataFrame({
        'month': monthly_returns.index.astype(str),
        'monthly_return': monthly_returns.values,
        'start_value': monthly_start_value.values,
        'end_value': monthly_end_value.values
    })
    
    monthly_returns_file = os.path.join(output_dir, 'hybrid_strategy_monthly_returns.csv')
    monthly_df.to_csv(monthly_returns_file, index=False)
    print(f"月度收益率序列已保存至: {monthly_returns_file}")
    
    # 输出收益率统计
    print("\n" + "="*60)
    print("收益率统计")
    print("="*60)
    print(f"日度收益率:")
    print(f"  均值: {returns_df['daily_return'].mean()*100:.4f}%")
    print(f"  标准差: {returns_df['daily_return'].std()*100:.4f}%")
    print(f"  最小值: {returns_df['daily_return'].min()*100:.4f}%")
    print(f"  最大值: {returns_df['daily_return'].max()*100:.4f}%")
    print(f"\n月度收益率:")
    print(f"  均值: {monthly_df['monthly_return'].mean()*100:.4f}%")
    print(f"  标准差: {monthly_df['monthly_return'].std()*100:.4f}%")
    print(f"  最小值: {monthly_df['monthly_return'].min()*100:.4f}%")
    print(f"  最大值: {monthly_df['monthly_return'].max()*100:.4f}%")
    print(f"\n总交易月数: {len(monthly_df)}")
    print("="*60)
    
    return results, returns_df, monthly_df


if __name__ == '__main__':
    results, daily_returns, monthly_returns = run_backtest_with_optimal_params()
