# -*- coding: utf-8 -*-
"""
根据最优参数回测生成策略收益率序列
用于CH-3/FF-3/CH-4因子回归分析

输出格式与 strategy_returns.csv 保持一致：
- date: 月末日期
- return_0cost: 0成本收益率（无交易成本）
- return_001: 0.1%单边成本
- return_002: 0.2%单边成本
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest.hybrid_backtester import HybridBacktester


def generate_strategy_returns():
    """
    生成策略收益率序列
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
        'return_001': monthly_returns.values,  # 无成本模型，所有成本列相同
        'return_002': monthly_returns.values
    })
    
    # 确保日期格式正确（月末）
    output_df['date'] = pd.to_datetime(output_df['date'])
    
    # 保存结果
    output_dir = 'results/tables'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'hybrid_strategy_returns.csv')
    output_df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("策略收益率序列生成完成")
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
    print("="*60)
    
    return output_df


if __name__ == '__main__':
    strategy_returns = generate_strategy_returns()
