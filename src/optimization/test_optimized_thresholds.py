# -*- coding: utf-8 -*-
"""
测试优化后的情绪阈值参数效果
"""

import sys
import os
import pandas as pd
import numpy as np
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.backtest.hybrid_backtester import HybridBacktester


def test_optimized_strategy():
    """测试优化阈值后的策略效果"""
    
    print("="*60)
    print("测试优化后情绪阈值的策略效果")
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
            'osa_threshold_extreme': -2.5,
            'osa_threshold_panic': -1.5,
            'osa_threshold_greed': 1.5,
            'scalar_extreme': 1.5,
            'scalar_greed': 0.5
        }
    
    print("\n策略参数:")
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
    
    # 计算绩效指标
    returns = results['daily_return'].dropna()
    total_return = (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_vol = returns.std() * (252 ** 0.5)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # 计算最大回撤
    cummax = results['portfolio_value'].cummax()
    drawdown = (results['portfolio_value'] - cummax) / cummax
    max_dd = drawdown.min()
    
    # 计算Calmar比率
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    print("\n" + "="*60)
    print("回测结果（优化后阈值）")
    print("="*60)
    print(f"数据期间: {results.index[0].strftime('%Y-%m-%d')} 至 {results.index[-1].strftime('%Y-%m-%d')}")
    print(f"总交易日: {len(returns)}")
    print(f"\n收益指标:")
    print(f"  总收益: {total_return*100:.2f}%")
    print(f"  年化收益: {annual_return*100:.2f}%")
    print(f"  年化波动: {annual_vol*100:.2f}%")
    print(f"\n风险调整指标:")
    print(f"  夏普比率: {sharpe:.3f}")
    print(f"  最大回撤: {max_dd*100:.2f}%")
    print(f"  Calmar比率: {calmar:.3f}")
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
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'results': results
    }


if __name__ == '__main__':
    results = test_optimized_strategy()
