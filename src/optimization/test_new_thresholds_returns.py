# -*- coding: utf-8 -*-
"""
测试优化情绪阈值后的策略收益率
生成收益率序列用于因子分析
"""

import sys
import os
import pandas as pd
import numpy as np
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.backtest.hybrid_backtester import HybridBacktester


def generate_optimized_returns():
    """生成优化阈值后的策略收益率序列"""
    
    print("="*60)
    print("优化情绪阈值后的策略收益率生成")
    print("="*60)
    
    # 加载最优参数配置
    config_path = 'config/hybrid_strategy_optimized_full.json'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        params = config['optimized_params']
        print("\n已加载最优参数配置")
    else:
        print("\n未找到配置文件，使用默认参数")
        params = {
            'ep_weight': 0.4,
            'reversal_weight': 0.6,
            'top_k': 50,
        }
    
    print("\n策略参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 创建回测引擎
    print("\n初始化回测引擎（使用优化后的情绪阈值）...")
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
    
    # 获取日度收益率
    daily_returns = results['daily_return'].copy()
    
    # 转换为月末日期格式（与因子数据对齐）
    daily_df = pd.DataFrame({
        'date': results.index,
        'daily_return': daily_returns.values
    })
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df['month'] = daily_df['date'].dt.to_period('M')
    
    # 计算月度收益率（复利）
    monthly_returns = daily_df.groupby('month').apply(
        lambda x: (1 + x['daily_return']).prod() - 1
    )
    
    # 获取月末日期
    month_end_dates = daily_df.groupby('month')['date'].last()
    
    # 构建输出DataFrame
    output_df = pd.DataFrame({
        'date': month_end_dates.values,
        'return_0cost': monthly_returns.values,
        'return_001': monthly_returns.values,
        'return_002': monthly_returns.values
    })
    
    # 确保日期格式正确（月末）
    output_df['date'] = pd.to_datetime(output_df['date'])
    
    # 保存结果
    output_dir = 'results/tables'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'hybrid_strategy_returns_optimized.csv')
    output_df.to_csv(output_file, index=False)
    
    # 计算绩效指标
    returns = results['daily_return'].dropna()
    total_return = (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_vol = returns.std() * (252 ** 0.5)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    cummax = results['portfolio_value'].cummax()
    drawdown = (results['portfolio_value'] - cummax) / cummax
    max_dd = drawdown.min()
    
    print("\n" + "="*60)
    print("策略收益率序列生成完成（优化阈值后）")
    print("="*60)
    print(f"输出文件: {output_file}")
    print(f"数据期间: {output_df['date'].min().strftime('%Y-%m-%d')} 至 {output_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"总月数: {len(output_df)}")
    print(f"\n收益率统计:")
    print(f"  均值: {output_df['return_0cost'].mean()*100:.4f}%")
    print(f"  标准差: {output_df['return_0cost'].std()*100:.4f}%")
    print(f"  年化收益: {((1 + output_df['return_0cost'].mean())**12 - 1)*100:.2f}%")
    print(f"  年化波动: {output_df['return_0cost'].std() * (12**0.5)*100:.2f}%")
    print(f"  夏普比率: {(output_df['return_0cost'].mean() / output_df['return_0cost'].std()) * (12**0.5):.3f}")
    print(f"\n回测绩效:")
    print(f"  总收益: {total_return*100:.2f}%")
    print(f"  年化收益: {annual_return*100:.2f}%")
    print(f"  最大回撤: {max_dd*100:.2f}%")
    print("="*60)
    
    # 分析情绪信号触发情况
    print("\n【情绪信号触发统计】")
    if hasattr(backtester, 'daily_tactician') and backtester.daily_tactician:
        dt = backtester.daily_tactician
        
        # 统计各信号类型
        signal_counts = {}
        for date in results.index:
            _, signal_type, _ = dt.get_position_scalar(date)
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
        
        print(f"\n信号类型分布:")
        for signal_type, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
            pct = count / len(results) * 100
            print(f"  {signal_type:25s}: {count:4d} 天 ({pct:5.2f}%)")
        
        # 显示阈值配置
        print(f"\n实际使用的阈值配置:")
        for k, v in dt.thresholds.items():
            print(f"  {k:20s}: {v:8.4f}")
    
    return output_df


if __name__ == '__main__':
    returns_df = generate_optimized_returns()
