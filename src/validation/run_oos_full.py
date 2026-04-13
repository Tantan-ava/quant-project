#!/usr/bin/env python3
"""
月度组合策略完整样本外验证
实际运行，检测过拟合
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
        
        backtester.run()
        
        return {
            'annual_return': backtester.metrics['annual_return'],
            'annual_volatility': backtester.metrics['annual_volatility'],
            'sharpe_ratio': backtester.metrics['sharpe_ratio'],
            'max_drawdown': backtester.metrics['max_drawdown'],
            'calmar_ratio': backtester.metrics['calmar_ratio'],
            'total_return': backtester.metrics['total_return'],
        }
    except Exception as e:
        print(f"  回测失败: {e}")
        return None


def parameter_search(train_start, train_end, param_grid):
    """参数搜索"""
    print(f"\n{'='*60}")
    print(f"参数搜索: {train_start} ~ {train_end}")
    print(f"{'='*60}")
    
    results = []
    total = len(list(product(*param_grid.values())))
    print(f"总组合数: {total}")
    print("-"*60)
    
    count = 0
    for ep_weight in param_grid['ep_weight']:
        for top_k in param_grid['top_k']:
            count += 1
            print(f"\n[{count}/{total}] EP={ep_weight:.1f}, TopK={top_k}")
            
            result = run_backtest(train_start, train_end, ep_weight, top_k)
            
            if result:
                results.append({
                    'ep_weight': ep_weight,
                    'top_k': top_k,
                    **result
                })
                print(f"  -> 夏普={result['sharpe_ratio']:.3f}, 收益={result['annual_return']:.2%}")
    
    if len(results) == 0:
        return None, None
    
    results_df = pd.DataFrame(results)
    
    # 选择最优参数（按夏普比率）
    best_idx = results_df['sharpe_ratio'].idxmax()
    best_params = {
        'ep_weight': results_df.loc[best_idx, 'ep_weight'],
        'top_k': int(results_df.loc[best_idx, 'top_k']),
    }
    
    print(f"\n{'='*60}")
    print("最优参数:")
    print(f"  EP权重: {best_params['ep_weight']}")
    print(f"  TopK: {best_params['top_k']}")
    print(f"  训练集夏普: {results_df.loc[best_idx, 'sharpe_ratio']:.3f}")
    print(f"{'='*60}")
    
    return best_params, results_df


def fixed_split_validation():
    """固定划分验证"""
    print("\n" + "="*70)
    print("固定划分样本外验证")
    print("="*70)
    
    # 数据划分
    train_start, train_end = '2006-01-01', '2018-12-31'
    val_start, val_end = '2019-01-01', '2022-12-31'
    test_start, test_end = '2023-01-01', '2025-12-31'
    
    print(f"\n数据划分:")
    print(f"  训练集: {train_start} ~ {train_end} (13年)")
    print(f"  验证集: {val_start} ~ {val_end} (4年)")
    print(f"  测试集: {test_start} ~ {test_end} (3年)")
    
    # 参数搜索空间
    param_grid = {
        'ep_weight': [0.3, 0.4, 0.5, 0.6],
        'top_k': [50, 100, 150],
    }
    
    # 1. 训练集参数搜索
    best_params, train_results_df = parameter_search(train_start, train_end, param_grid)
    
    if best_params is None:
        print("错误: 参数搜索失败")
        return None
    
    # 2. 训练集回测（使用最优参数）
    print(f"\n{'='*60}")
    print(f"训练集最终回测: {train_start} ~ {train_end}")
    print(f"{'='*60}")
    train_perf = run_backtest(
        train_start, train_end,
        best_params['ep_weight'],
        best_params['top_k']
    )
    
    # 3. 验证集回测
    print(f"\n{'='*60}")
    print(f"验证集回测: {val_start} ~ {val_end}")
    print(f"{'='*60}")
    val_perf = run_backtest(
        val_start, val_end,
        best_params['ep_weight'],
        best_params['top_k']
    )
    
    # 4. 测试集回测（样本外）
    print(f"\n{'='*60}")
    print(f"测试集回测 (样本外): {test_start} ~ {test_end}")
    print(f"{'='*60}")
    test_perf = run_backtest(
        test_start, test_end,
        best_params['ep_weight'],
        best_params['top_k']
    )
    
    # 5. 计算衰减
    sharpe_decay_val = val_perf['sharpe_ratio'] - train_perf['sharpe_ratio']
    sharpe_decay_test = test_perf['sharpe_ratio'] - train_perf['sharpe_ratio']
    return_decay_val = val_perf['annual_return'] - train_perf['annual_return']
    return_decay_test = test_perf['annual_return'] - train_perf['annual_return']
    
    # 6. 评估过拟合风险
    def assess_risk(decay):
        if decay < -0.2:
            return "高"
        elif decay < -0.1:
            return "中等"
        else:
            return "低"
    
    # 7. 输出结果
    print("\n" + "="*70)
    print("固定划分验证结果汇总")
    print("="*70)
    
    print(f"\n最优参数:")
    print(f"  EP权重: {best_params['ep_weight']}")
    print(f"  TopK: {best_params['top_k']}")
    
    print(f"\n训练集表现 ({train_start} ~ {train_end}):")
    print(f"  年化收益: {train_perf['annual_return']:.2%}")
    print(f"  年化波动: {train_perf['annual_volatility']:.2%}")
    print(f"  夏普比率: {train_perf['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {train_perf['max_drawdown']:.2%}")
    
    print(f"\n验证集表现 ({val_start} ~ {val_end}):")
    print(f"  年化收益: {val_perf['annual_return']:.2%}")
    print(f"  年化波动: {val_perf['annual_volatility']:.2%}")
    print(f"  夏普比率: {val_perf['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {val_perf['max_drawdown']:.2%}")
    
    print(f"\n测试集表现 (样本外) ({test_start} ~ {test_end}):")
    print(f"  年化收益: {test_perf['annual_return']:.2%}")
    print(f"  年化波动: {test_perf['annual_volatility']:.2%}")
    print(f"  夏普比率: {test_perf['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {test_perf['max_drawdown']:.2%}")
    
    print(f"\n样本外衰减:")
    print(f"  验证集夏普衰减: {sharpe_decay_val:.3f} (风险: {assess_risk(sharpe_decay_val)})")
    print(f"  测试集夏普衰减: {sharpe_decay_test:.3f} (风险: {assess_risk(sharpe_decay_test)})")
    print(f"  验证集收益衰减: {return_decay_val:.2%}")
    print(f"  测试集收益衰减: {return_decay_test:.2%}")
    
    result = {
        'validation_type': 'fixed_split',
        'best_params': best_params,
        'train': {
            'period': f"{train_start} ~ {train_end}",
            **train_perf
        },
        'validation': {
            'period': f"{val_start} ~ {val_end}",
            **val_perf,
            'sharpe_decay': sharpe_decay_val,
            'return_decay': return_decay_val,
            'risk_level': assess_risk(sharpe_decay_val)
        },
        'test': {
            'period': f"{test_start} ~ {test_end}",
            **test_perf,
            'sharpe_decay': sharpe_decay_test,
            'return_decay': return_decay_test,
            'risk_level': assess_risk(sharpe_decay_test)
        }
    }
    
    return result


def rolling_window_validation():
    """滚动窗口验证"""
    print("\n" + "="*70)
    print("滚动窗口样本外验证")
    print("="*70)
    
    # 窗口设置
    windows = [
        ('2006-01-01', '2010-12-31', '2011-01-01', '2011-12-31'),
        ('2007-01-01', '2011-12-31', '2012-01-01', '2012-12-31'),
        ('2008-01-01', '2012-12-31', '2013-01-01', '2013-12-31'),
        ('2009-01-01', '2013-12-31', '2014-01-01', '2014-12-31'),
        ('2010-01-01', '2014-12-31', '2015-01-01', '2015-12-31'),
        ('2011-01-01', '2015-12-31', '2016-01-01', '2016-12-31'),
        ('2012-01-01', '2016-12-31', '2017-01-01', '2017-12-31'),
        ('2013-01-01', '2017-12-31', '2018-01-01', '2018-12-31'),
        ('2014-01-01', '2018-12-31', '2019-01-01', '2019-12-31'),
        ('2015-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
        ('2016-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
        ('2017-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
        ('2018-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
        ('2019-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ]
    
    # 参数搜索空间
    param_grid = {
        'ep_weight': [0.3, 0.4, 0.5, 0.6],
        'top_k': [50, 100, 150],
    }
    
    all_results = []
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"\n{'#'*70}")
        print(f"# 窗口 {i+1}/{len(windows)}: {train_start} ~ {test_end}")
        print(f"{'#'*70}")
        
        # 参数搜索
        best_params, _ = parameter_search(train_start, train_end, param_grid)
        
        if best_params is None:
            continue
        
        # 训练集回测
        train_perf = run_backtest(
            train_start, train_end,
            best_params['ep_weight'],
            best_params['top_k']
        )
        
        # 测试集回测
        test_perf = run_backtest(
            test_start, test_end,
            best_params['ep_weight'],
            best_params['top_k']
        )
        
        # 计算衰减
        sharpe_decay = test_perf['sharpe_ratio'] - train_perf['sharpe_ratio']
        return_decay = test_perf['annual_return'] - train_perf['annual_return']
        
        result = {
            'window': i+1,
            'train_period': f"{train_start} ~ {train_end}",
            'test_period': f"{test_start} ~ {test_end}",
            'best_params': best_params,
            'train_sharpe': train_perf['sharpe_ratio'],
            'train_return': train_perf['annual_return'],
            'test_sharpe': test_perf['sharpe_ratio'],
            'test_return': test_perf['annual_return'],
            'sharpe_decay': sharpe_decay,
            'return_decay': return_decay,
        }
        
        all_results.append(result)
        
        print(f"\n窗口 {i+1} 结果:")
        print(f"  最优参数: EP={best_params['ep_weight']}, TopK={best_params['top_k']}")
        print(f"  训练夏普: {train_perf['sharpe_ratio']:.3f} -> 测试夏普: {test_perf['sharpe_ratio']:.3f}")
        print(f"  夏普衰减: {sharpe_decay:.3f}")
    
    # 汇总统计
    if len(all_results) == 0:
        return None
    
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("滚动窗口验证汇总")
    print("="*70)
    
    print(f"\n窗口数量: {len(all_results)}")
    
    print(f"\n训练集表现:")
    print(f"  夏普比率 - 均值: {results_df['train_sharpe'].mean():.3f}, 标准差: {results_df['train_sharpe'].std():.3f}")
    print(f"  年化收益 - 均值: {results_df['train_return'].mean():.2%}, 标准差: {results_df['train_return'].std():.2%}")
    
    print(f"\n测试集表现 (样本外):")
    print(f"  夏普比率 - 均值: {results_df['test_sharpe'].mean():.3f}, 标准差: {results_df['test_sharpe'].std():.3f}")
    print(f"  年化收益 - 均值: {results_df['test_return'].mean():.2%}, 标准差: {results_df['test_return'].std():.2%}")
    
    print(f"\n样本外衰减:")
    print(f"  夏普比率衰减 - 均值: {results_df['sharpe_decay'].mean():.3f}, 标准差: {results_df['sharpe_decay'].std():.3f}")
    print(f"  年化收益衰减 - 均值: {results_df['return_decay'].mean():.2%}, 标准差: {results_df['return_decay'].std():.2%}")
    
    # 统计检验
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(results_df['test_sharpe'].dropna(), 0)
    
    print(f"\n样本外夏普比率统计检验 (H0: 夏普=0):")
    print(f"  t统计量: {t_stat:.3f}")
    print(f"  p值: {p_value:.4f}")
    print(f"  是否显著大于0 (α=0.05): {'是' if p_value < 0.05 and t_stat > 0 else '否'}")
    
    result = {
        'validation_type': 'rolling_window',
        'n_windows': len(all_results),
        'train': {
            'sharpe_mean': results_df['train_sharpe'].mean(),
            'sharpe_std': results_df['train_sharpe'].std(),
            'return_mean': results_df['train_return'].mean(),
            'return_std': results_df['train_return'].std(),
        },
        'test': {
            'sharpe_mean': results_df['test_sharpe'].mean(),
            'sharpe_std': results_df['test_sharpe'].std(),
            'return_mean': results_df['test_return'].mean(),
            'return_std': results_df['test_return'].std(),
        },
        'decay': {
            'sharpe_decay_mean': results_df['sharpe_decay'].mean(),
            'sharpe_decay_std': results_df['sharpe_decay'].std(),
            'return_decay_mean': results_df['return_decay'].mean(),
            'return_decay_std': results_df['return_decay'].std(),
        },
        'statistical_test': {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05 and t_stat > 0
        },
        'window_details': all_results
    }
    
    return result


def main():
    """主函数"""
    print("="*70)
    print("月度组合策略样本外验证 - 完整运行")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    # 1. 固定划分验证
    fixed_result = fixed_split_validation()
    if fixed_result:
        all_results['fixed_split'] = fixed_result
    
    # 2. 滚动窗口验证
    rolling_result = rolling_window_validation()
    if rolling_result:
        all_results['rolling_window'] = rolling_result
    
    # 保存结果
    output_file = '/Users/xinyutan/Documents/量化投资/quant-project/results/oos_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"所有结果已保存: {output_file}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return all_results


if __name__ == '__main__':
    results = main()
