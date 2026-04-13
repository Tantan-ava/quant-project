#!/usr/bin/env python3
"""
月度组合策略样本外验证 - 实用版本

使用现有的MonthlyComboAllMarketBacktester进行样本外验证

验证方案:
1. 固定划分: 训练(2006-2018) / 验证(2019-2022) / 测试(2023-2025)
2. 滚动窗口: 5年训练 + 1年测试

作者: Assistant
日期: 2026-04-13
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


def run_backtest_with_params(start_date, end_date, ep_weight, reversal_months, top_k):
    """
    使用指定参数运行回测
    
    注意: reversal_months参数在当前引擎中固定为6，这里仅作记录
    """
    try:
        backtester = MonthlyComboAllMarketBacktester(
            monthly_returns_path='data/raw/TRD_Mnth.xlsx',
            start_date=start_date,
            end_date=end_date,
            ep_weight=ep_weight,
            reversal_weight=1 - ep_weight,  # 反转权重 = 1 - EP权重
            top_k=top_k,
            winsorize=True
        )
        
        results = backtester.run_backtest()
        
        return {
            'annual_return': results['annual_return'],
            'volatility': results['annual_volatility'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'calmar_ratio': results['calmar_ratio'],
            'total_return': results['total_return'],
        }
    except Exception as e:
        print(f"回测失败: {e}")
        return None


def parameter_search(train_start, train_end, param_grid):
    """在训练集上搜索最优参数"""
    print(f"\n参数搜索: {train_start} ~ {train_end}")
    print(f"参数组合数: {len(list(product(*param_grid.values())))}")
    
    results = []
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for combo in product(*values):
        param_set = dict(zip(keys, combo))
        
        result = run_backtest_with_params(
            train_start, train_end,
            param_set['ep_weight'],
            param_set['reversal_months'],
            param_set['top_k']
        )
        
        if result:
            row = param_set.copy()
            row.update(result)
            results.append(row)
            print(f"  参数{param_set} -> 夏普={result['sharpe_ratio']:.3f}")
    
    results_df = pd.DataFrame(results)
    
    # 选择夏普比率最高的参数
    best_idx = results_df['sharpe_ratio'].idxmax()
    best_params = results_df.loc[best_idx, keys].to_dict()
    
    return best_params, results_df


def fixed_split_validation():
    """固定划分验证"""
    print("="*60)
    print("固定划分样本外验证")
    print("="*60)
    
    # 数据划分
    train_start, train_end = '2006-01-01', '2018-12-31'
    val_start, val_end = '2019-01-01', '2022-12-31'
    test_start, test_end = '2023-01-01', '2025-12-31'
    
    # 参数搜索空间
    param_grid = {
        'ep_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
        'reversal_months': [6],  # 固定为6
        'top_k': [50, 100, 150],
    }
    
    # 1. 训练集参数搜索
    print("\n[1/3] 训练集参数搜索...")
    best_params, train_results_df = parameter_search(train_start, train_end, param_grid)
    print(f"\n最优参数: {best_params}")
    
    # 2. 训练集回测
    print("\n[2/3] 训练集回测...")
    train_perf = run_backtest_with_params(
        train_start, train_end,
        best_params['ep_weight'],
        best_params['reversal_months'],
        best_params['top_k']
    )
    
    # 3. 验证集回测
    print("\n[3/3] 验证集回测...")
    val_perf = run_backtest_with_params(
        val_start, val_end,
        best_params['ep_weight'],
        best_params['reversal_months'],
        best_params['top_k']
    )
    
    # 4. 测试集回测（真正的样本外）
    print("\n[4/4] 测试集样本外验证...")
    test_perf = run_backtest_with_params(
        test_start, test_end,
        best_params['ep_weight'],
        best_params['reversal_months'],
        best_params['top_k']
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("固定划分验证结果")
    print("="*60)
    
    print(f"\n最优参数:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    print(f"\n训练集 ({train_start} ~ {train_end}):")
    print(f"  年化收益: {train_perf['annual_return']:.2%}")
    print(f"  夏普比率: {train_perf['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {train_perf['max_drawdown']:.2%}")
    
    print(f"\n验证集 ({val_start} ~ {val_end}):")
    print(f"  年化收益: {val_perf['annual_return']:.2%}")
    print(f"  夏普比率: {val_perf['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {val_perf['max_drawdown']:.2%}")
    
    print(f"\n测试集 ({test_start} ~ {test_end}) - 样本外:")
    print(f"  年化收益: {test_perf['annual_return']:.2%}")
    print(f"  夏普比率: {test_perf['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {test_perf['max_drawdown']:.2%}")
    
    # 计算衰减
    sharpe_decay = test_perf['sharpe_ratio'] - train_perf['sharpe_ratio']
    return_decay = test_perf['annual_return'] - train_perf['annual_return']
    
    print(f"\n样本外衰减:")
    print(f"  夏普比率衰减: {sharpe_decay:.3f}")
    print(f"  年化收益衰减: {return_decay:.2%}")
    
    # 保存结果
    result = {
        'best_params': best_params,
        'train_performance': train_perf,
        'val_performance': val_perf,
        'test_performance': test_perf,
        'sharpe_decay': sharpe_decay,
        'return_decay': return_decay,
    }
    
    output_file = '/Users/xinyutan/Documents/量化投资/quant-project/results/oos_fixed_split.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n结果已保存: {output_file}")
    
    return result


def rolling_window_validation():
    """滚动窗口验证"""
    print("\n" + "="*60)
    print("滚动窗口样本外验证")
    print("="*60)
    
    # 滚动窗口设置
    window_train_years = 5
    window_test_years = 1
    
    # 生成窗口
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
    
    # 参数搜索空间（简化以加快计算）
    param_grid = {
        'ep_weight': [0.4, 0.5, 0.6],
        'reversal_months': [6],
        'top_k': [50, 100],
    }
    
    all_results = []
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"\n{'#'*60}")
        print(f"# 窗口 {i+1}/{len(windows)}: {train_start} ~ {test_end}")
        print(f"{'#'*60}")
        
        # 参数搜索
        best_params, _ = parameter_search(train_start, train_end, param_grid)
        
        # 训练集回测
        train_perf = run_backtest_with_params(
            train_start, train_end,
            best_params['ep_weight'],
            best_params['reversal_months'],
            best_params['top_k']
        )
        
        # 测试集回测
        test_perf = run_backtest_with_params(
            test_start, test_end,
            best_params['ep_weight'],
            best_params['reversal_months'],
            best_params['top_k']
        )
        
        result = {
            'window': f"{train_start}_{test_end}",
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'best_params': best_params,
            'train_sharpe': train_perf['sharpe_ratio'],
            'test_sharpe': test_perf['sharpe_ratio'],
            'train_return': train_perf['annual_return'],
            'test_return': test_perf['annual_return'],
            'sharpe_decay': test_perf['sharpe_ratio'] - train_perf['sharpe_ratio'],
            'return_decay': test_perf['annual_return'] - train_perf['annual_return'],
        }
        
        all_results.append(result)
        
        print(f"\n窗口 {i+1} 结果:")
        print(f"  最优参数: EP={best_params['ep_weight']}, TopK={best_params['top_k']}")
        print(f"  训练集夏普: {train_perf['sharpe_ratio']:.3f}")
        print(f"  测试集夏普: {test_perf['sharpe_ratio']:.3f}")
        print(f"  夏普衰减: {result['sharpe_decay']:.3f}")
    
    # 汇总结果
    summary_df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("滚动窗口验证汇总")
    print("="*60)
    
    print(f"\n窗口数量: {len(summary_df)}")
    
    print(f"\n训练集表现:")
    print(f"  夏普比率 - 均值: {summary_df['train_sharpe'].mean():.3f}, 标准差: {summary_df['train_sharpe'].std():.3f}")
    print(f"  年化收益 - 均值: {summary_df['train_return'].mean():.2%}, 标准差: {summary_df['train_return'].std():.2%}")
    
    print(f"\n测试集表现 (样本外):")
    print(f"  夏普比率 - 均值: {summary_df['test_sharpe'].mean():.3f}, 标准差: {summary_df['test_sharpe'].std():.3f}")
    print(f"  年化收益 - 均值: {summary_df['test_return'].mean():.2%}, 标准差: {summary_df['test_return'].std():.2%}")
    
    print(f"\n样本外衰减:")
    print(f"  夏普比率衰减 - 均值: {summary_df['sharpe_decay'].mean():.3f}, 标准差: {summary_df['sharpe_decay'].std():.3f}")
    print(f"  年化收益衰减 - 均值: {summary_df['return_decay'].mean():.2%}, 标准差: {summary_df['return_decay'].std():.2%}")
    
    # 统计显著性
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(summary_df['test_sharpe'], 0)
    print(f"\n样本外夏普比率统计检验:")
    print(f"  t统计量: {t_stat:.3f}")
    print(f"  p值: {p_value:.4f}")
    print(f"  是否显著大于0: {'是' if p_value < 0.05 and t_stat > 0 else '否'}")
    
    # 保存结果
    output_file = '/Users/xinyutan/Documents/量化投资/quant-project/results/oos_rolling_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")
    
    return summary_df


def main():
    """主函数"""
    print("="*60)
    print("月度组合策略样本外验证")
    print("="*60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行固定划分验证
    fixed_result = fixed_split_validation()
    
    # 运行滚动窗口验证
    rolling_result = rolling_window_validation()
    
    print("\n" + "="*60)
    print("样本外验证完成")
    print("="*60)


if __name__ == '__main__':
    main()
