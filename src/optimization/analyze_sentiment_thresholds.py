# -*- coding: utf-8 -*-
"""
分析情绪数据分布并优化阈值参数
"""

import pandas as pd
import numpy as np
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def analyze_sentiment_distribution():
    """分析情绪数据分布"""
    # 加载情绪数据
    sentiment_path = 'data/processed/daily_sentiment_index.csv'
    sentiment = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
    
    print("="*60)
    print("情绪数据分布分析")
    print("="*60)
    
    # 基础统计
    print("\n【基础统计】")
    print(f"数据期间: {sentiment.index.min().strftime('%Y-%m-%d')} 至 {sentiment.index.max().strftime('%Y-%m-%d')}")
    print(f"总天数: {len(sentiment)}")
    print(f"\n情绪分数统计:")
    print(f"  均值: {sentiment['sentiment_score'].mean():.4f}")
    print(f"  标准差: {sentiment['sentiment_score'].std():.4f}")
    print(f"  最小值: {sentiment['sentiment_score'].min():.4f}")
    print(f"  最大值: {sentiment['sentiment_score'].max():.4f}")
    print(f"  中位数: {sentiment['sentiment_score'].median():.4f}")
    
    # 计算Z-score
    sentiment['sentiment_zscore'] = (
        (sentiment['sentiment_score'] - sentiment['sentiment_score'].mean()) /
        sentiment['sentiment_score'].std()
    )
    
    print(f"\nZ-score统计:")
    print(f"  均值: {sentiment['sentiment_zscore'].mean():.4f}")
    print(f"  标准差: {sentiment['sentiment_zscore'].std():.4f}")
    print(f"  最小值: {sentiment['sentiment_zscore'].min():.4f}")
    print(f"  最大值: {sentiment['sentiment_zscore'].max():.4f}")
    
    # 百分位数分析
    print("\n【百分位数分析】")
    percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    print("原始分数百分位:")
    for p in percentiles:
        val = np.percentile(sentiment['sentiment_score'], p)
        print(f"  P{p:2d}: {val:8.4f}")
    
    print("\nZ-score百分位:")
    for p in percentiles:
        val = np.percentile(sentiment['sentiment_zscore'], p)
        print(f"  P{p:2d}: {val:8.4f}")
    
    # 当前阈值触发频率分析
    print("\n【当前阈值触发频率】")
    current_thresholds = {
        'extreme_panic': -2.0,
        'panic': -1.0,
        'neutral_low': -0.5,
        'neutral_high': 0.5,
        'optimistic': 1.0,
        'extreme_greed': 2.0,
    }
    
    z = sentiment['sentiment_zscore']
    
    extreme_panic = (z < current_thresholds['extreme_panic']).sum()
    panic = ((z >= current_thresholds['extreme_panic']) & (z < current_thresholds['panic'])).sum()
    concern = ((z >= current_thresholds['panic']) & (z < current_thresholds['neutral_low'])).sum()
    neutral = ((z >= current_thresholds['neutral_low']) & (z <= current_thresholds['neutral_high'])).sum()
    optimistic_hold = ((z > current_thresholds['neutral_high']) & (z <= current_thresholds['optimistic'])).sum()
    optimistic_reduce = ((z > current_thresholds['optimistic']) & (z <= current_thresholds['extreme_greed'])).sum()
    extreme_greed = (z > current_thresholds['extreme_greed']).sum()
    
    total = len(z)
    
    print(f"极端恐慌 (< -2.0): {extreme_panic:4d} 天 ({extreme_panic/total*100:5.2f}%) -> 超配150%")
    print(f"恐慌 (-2.0 ~ -1.0): {panic:4d} 天 ({panic/total*100:5.2f}%) -> 超配120%")
    print(f"轻度担忧 (-1.0 ~ -0.5): {concern:4d} 天 ({concern/total*100:5.2f}%) -> 低配80%")
    print(f"中性 (-0.5 ~ 0.5): {neutral:4d} 天 ({neutral/total*100:5.2f}%) -> 标配100%")
    print(f"轻度乐观 (0.5 ~ 1.0): {optimistic_hold:4d} 天 ({optimistic_hold/total*100:5.2f}%) -> 标配100%")
    print(f"乐观 (1.0 ~ 2.0): {optimistic_reduce:4d} 天 ({optimistic_reduce/total*100:5.2f}%) -> 减仓80%")
    print(f"极度乐观 (> 2.0): {extreme_greed:4d} 天 ({extreme_greed/total*100:5.2f}%) -> 大幅减仓50%")
    
    # 优化建议
    print("\n【优化建议】")
    print("基于数据分布，建议调整阈值如下:")
    
    # 使用更合理的百分位作为阈值
    optimized_thresholds = {
        'extreme_panic': np.percentile(sentiment['sentiment_zscore'], 5),      # P5
        'panic': np.percentile(sentiment['sentiment_zscore'], 20),             # P20
        'neutral_low': np.percentile(sentiment['sentiment_zscore'], 40),       # P40
        'neutral_high': np.percentile(sentiment['sentiment_zscore'], 60),      # P60
        'optimistic': np.percentile(sentiment['sentiment_zscore'], 80),        # P80
        'extreme_greed': np.percentile(sentiment['sentiment_zscore'], 95),     # P95
    }
    
    print(f"\n优化后阈值（基于百分位）:")
    for k, v in optimized_thresholds.items():
        print(f"  {k}: {v:.4f}")
    
    # 优化后触发频率
    print("\n【优化后触发频率】")
    extreme_panic_o = (z < optimized_thresholds['extreme_panic']).sum()
    panic_o = ((z >= optimized_thresholds['extreme_panic']) & (z < optimized_thresholds['panic'])).sum()
    concern_o = ((z >= optimized_thresholds['panic']) & (z < optimized_thresholds['neutral_low'])).sum()
    neutral_o = ((z >= optimized_thresholds['neutral_low']) & (z <= optimized_thresholds['neutral_high'])).sum()
    optimistic_hold_o = ((z > optimized_thresholds['neutral_high']) & (z <= optimized_thresholds['optimistic'])).sum()
    optimistic_reduce_o = ((z > optimized_thresholds['optimistic']) & (z <= optimized_thresholds['extreme_greed'])).sum()
    extreme_greed_o = (z > optimized_thresholds['extreme_greed']).sum()
    
    print(f"极端恐慌: {extreme_panic_o:4d} 天 ({extreme_panic_o/total*100:5.2f}%) -> 超配150%")
    print(f"恐慌: {panic_o:4d} 天 ({panic_o/total*100:5.2f}%) -> 超配120%")
    print(f"轻度担忧: {concern_o:4d} 天 ({concern_o/total*100:5.2f}%) -> 低配80%")
    print(f"中性: {neutral_o:4d} 天 ({neutral_o/total*100:5.2f}%) -> 标配100%")
    print(f"轻度乐观: {optimistic_hold_o:4d} 天 ({optimistic_hold_o/total*100:5.2f}%) -> 标配100%")
    print(f"乐观: {optimistic_reduce_o:4d} 天 ({optimistic_reduce_o/total*100:5.2f}%) -> 减仓80%")
    print(f"极度乐观: {extreme_greed_o:4d} 天 ({extreme_greed_o/total*100:5.2f}%) -> 大幅减仓50%")
    
    return optimized_thresholds


def test_optimized_thresholds(optimized_thresholds):
    """测试优化后的阈值效果"""
    print("\n" + "="*60)
    print("测试优化后阈值效果")
    print("="*60)
    
    from backtest.hybrid_backtester import HybridBacktester
    
    # 加载最优参数配置
    config_path = 'config/hybrid_strategy_optimized_full.json'
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        params = config['optimized_params']
    else:
        params = {
            'ep_weight': 0.4,
            'reversal_weight': 0.6,
            'top_k': 50,
        }
    
    # 使用优化后的阈值
    params['osa_threshold_extreme'] = optimized_thresholds['extreme_panic']
    params['osa_threshold_panic'] = optimized_thresholds['panic']
    params['osa_threshold_greed'] = optimized_thresholds['optimistic']
    
    print(f"\n测试参数:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # 执行回测
    backtester = HybridBacktester(
        daily_returns_path='data/raw/TRD-daily.csv',
        sentiment_path='data/processed/daily_sentiment_index.csv',
        start_date='2022-02-01',
        end_date='2026-04-10',
        initial_capital=1e8,
        **params
    )
    
    results = backtester.run()
    
    # 计算绩效
    returns = results['daily_return'].dropna()
    total_return = (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_vol = returns.std() * (252 ** 0.5)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    cummax = results['portfolio_value'].cummax()
    drawdown = (results['portfolio_value'] - cummax) / cummax
    max_dd = drawdown.min()
    
    print(f"\n回测结果:")
    print(f"  总收益: {total_return*100:.2f}%")
    print(f"  年化收益: {annual_return*100:.2f}%")
    print(f"  年化波动: {annual_vol*100:.2f}%")
    print(f"  夏普比率: {sharpe:.3f}")
    print(f"  最大回撤: {max_dd*100:.2f}%")
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'thresholds': optimized_thresholds
    }


if __name__ == '__main__':
    # 分析情绪分布
    optimized_thresholds = analyze_sentiment_distribution()
    
    # 测试优化后的阈值
    results = test_optimized_thresholds(optimized_thresholds)
