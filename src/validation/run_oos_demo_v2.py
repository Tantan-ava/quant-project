#!/usr/bin/env python3
"""
月度组合策略样本外验证 - 演示版本 V2

快速演示样本外验证流程
"""

import pandas as pd
import numpy as np
import sys
import json
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/xinyutan/Documents/量化投资/quant-project')

from src.backtest.monthly_combo_all_market import MonthlyComboAllMarketBacktester


def run_backtest(start_date, end_date, ep_weight, top_k):
    """运行回测"""
    try:
        backtester = MonthlyComboAllMarketBacktester(
            monthly_returns_path='data/raw/TRD_Mnth.xlsx',
            start_date=start_date,
            end_date=end_date,
            ep_weight=ep_weight,
            reversal_weight=1 - ep_weight,
            top_k=top_k,
            winsorize=True
        )
        
        # 运行回测，结果存储在backtester.metrics中
        backtester.run()
        
        return {
            'annual_return': backtester.metrics['annual_return'],
            'sharpe_ratio': backtester.metrics['sharpe_ratio'],
            'max_drawdown': backtester.metrics['max_drawdown'],
        }
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_oos_validation():
    """演示样本外验证"""
    print("="*60)
    print("月度组合策略样本外验证 - 演示")
    print("="*60)
    
    # 数据划分
    train_start, train_end = '2006-01-01', '2018-12-31'
    test_start, test_end = '2023-01-01', '2025-12-31'
    
    # 参数搜索空间
    param_grid = {
        'ep_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
        'top_k': [50, 100, 150],
    }
    
    print(f"\n数据划分:")
    print(f"  训练集: {train_start} ~ {train_end}")
    print(f"  测试集: {test_start} ~ {test_end}")
    print(f"\n参数搜索空间:")
    print(f"  EP权重: {param_grid['ep_weight']}")
    print(f"  TopK: {param_grid['top_k']}")
    print(f"  总组合数: {len(list(product(*param_grid.values())))}")
    
    # 1. 训练集参数搜索
    print("\n" + "-"*60)
    print("[步骤1] 训练集参数搜索...")
    print("-"*60)
    
    results = []
    for ep_weight in param_grid['ep_weight']:
        for top_k in param_grid['top_k']:
            result = run_backtest(train_start, train_end, ep_weight, top_k)
            if result:
                results.append({
                    'ep_weight': ep_weight,
                    'top_k': top_k,
                    **result
                })
                print(f"  EP={ep_weight:.1f}, TopK={top_k:3d} -> 夏普={result['sharpe_ratio']:.3f}, 收益={result['annual_return']:.2%}")
    
    if len(results) == 0:
        print("错误: 所有回测都失败了")
        return None
        
    results_df = pd.DataFrame(results)
    
    # 选择最优参数
    best_idx = results_df['sharpe_ratio'].idxmax()
    best_params = {
        'ep_weight': results_df.loc[best_idx, 'ep_weight'],
        'top_k': int(results_df.loc[best_idx, 'top_k']),
    }
    
    print(f"\n最优参数:")
    print(f"  EP权重: {best_params['ep_weight']}")
    print(f"  TopK: {best_params['top_k']}")
    print(f"  训练集夏普: {results_df.loc[best_idx, 'sharpe_ratio']:.3f}")
    
    # 2. 测试集样本外验证
    print("\n" + "-"*60)
    print("[步骤2] 测试集样本外验证...")
    print("-"*60)
    
    test_result = run_backtest(
        test_start, test_end,
        best_params['ep_weight'],
        best_params['top_k']
    )
    
    print(f"\n样本外结果:")
    print(f"  年化收益: {test_result['annual_return']:.2%}")
    print(f"  夏普比率: {test_result['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {test_result['max_drawdown']:.2%}")
    
    # 计算衰减
    train_sharpe = results_df.loc[best_idx, 'sharpe_ratio']
    train_return = results_df.loc[best_idx, 'annual_return']
    sharpe_decay = test_result['sharpe_ratio'] - train_sharpe
    return_decay = test_result['annual_return'] - train_return
    
    print(f"\n样本外衰减:")
    print(f"  夏普比率衰减: {sharpe_decay:.3f}")
    print(f"  年化收益衰减: {return_decay:.2%}")
    
    # 评估过拟合风险
    print(f"\n过拟合风险评估:")
    if sharpe_decay < -0.2:
        risk = "高"
    elif sharpe_decay < -0.1:
        risk = "中等"
    else:
        risk = "低"
    print(f"  风险等级: {risk}")
    
    # 保存结果
    result = {
        'best_params': best_params,
        'train_performance': {
            'sharpe_ratio': train_sharpe,
            'annual_return': train_return,
        },
        'test_performance': test_result,
        'sharpe_decay': sharpe_decay,
        'return_decay': return_decay,
        'overfitting_risk': risk,
    }
    
    output_file = '/Users/xinyutan/Documents/量化投资/quant-project/results/oos_demo_result.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n结果已保存: {output_file}")
    
    return result


if __name__ == '__main__':
    result = demo_oos_validation()
    
    print("\n" + "="*60)
    print("样本外验证演示完成")
    print("="*60)
