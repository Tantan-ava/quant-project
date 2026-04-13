#!/usr/bin/env python3
"""
运行月度组合策略的样本外验证

验证方案:
1. 固定划分验证: 训练(2006-2018) / 验证(2019-2022) / 测试(2023-2025)
2. 滚动窗口验证: 5年训练 + 1年测试，滚动步长1年

作者: Assistant
日期: 2026-04-13
"""

import pandas as pd
import numpy as np
import sys
import json
from datetime import datetime

sys.path.append('/Users/xinyutan/Documents/量化投资/quant-project')

from src.validation.out_of_sample_framework import (
    ValidationConfig, OutOfSampleValidator, 
    create_default_backtest_func, BacktestResult
)


def load_data():
    """加载数据"""
    print("加载数据...")
    
    # 加载月度收益率数据
    returns_file = '/Users/xinyutan/Documents/量化投资/quant-project/data/processed/monthly_returns_wide.csv'
    returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    
    # 加载EP数据（这里简化处理，使用模拟数据或从其他数据源获取）
    # 实际应用中应该从财务数据计算EP
    ep_file = '/Users/xinyutan/Documents/量化投资/quant-project/data/processed/ep_data.csv'
    try:
        ep_df = pd.read_csv(ep_file, index_col=0, parse_dates=True)
    except:
        print("警告: EP数据文件不存在，使用模拟数据")
        # 创建模拟EP数据（实际应用中需要真实数据）
        ep_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        np.random.seed(42)
        for col in ep_df.columns:
            ep_df[col] = np.random.randn(len(ep_df)) * 0.1 + 0.05
    
    print(f"收益率数据: {returns_df.shape}")
    print(f"EP数据: {ep_df.shape}")
    print(f"数据期间: {returns_df.index[0]} ~ {returns_df.index[-1]}")
    
    return returns_df, ep_df


def run_fixed_split_validation(returns_df, ep_df):
    """运行固定划分验证"""
    print("\n" + "="*60)
    print("固定划分样本外验证")
    print("="*60)
    
    config = ValidationConfig(
        train_start='2006-01-01',
        train_end='2018-12-31',
        val_start='2019-01-01',
        val_end='2022-12-31',
        test_start='2023-01-01',
        test_end='2025-12-31',
        param_grid={
            'ep_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
            'reversal_months': [3, 6, 12],
            'top_k': [50, 100, 150],
        }
    )
    
    validator = OutOfSampleValidator(config)
    backtest_func = create_default_backtest_func(returns_df, ep_df)
    
    result = validator.run_fixed_split_validation(backtest_func, scoring='sharpe_ratio')
    
    # 打印结果
    print("\n" + "="*60)
    print("固定划分验证结果")
    print("="*60)
    
    print(f"\n最优参数:")
    for k, v in result['best_params'].items():
        print(f"  {k}: {v}")
    
    print(f"\n训练集 (2006-2018):")
    print(f"  年化收益: {result['train_performance']['annual_return']:.2%}")
    print(f"  夏普比率: {result['train_performance']['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {result['train_performance']['max_drawdown']:.2%}")
    
    print(f"\n验证集 (2019-2022):")
    print(f"  年化收益: {result['val_performance']['annual_return']:.2%}")
    print(f"  夏普比率: {result['val_performance']['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {result['val_performance']['max_drawdown']:.2%}")
    
    print(f"\n测试集 (2023-2025) - 样本外:")
    print(f"  年化收益: {result['test_performance']['annual_return']:.2%}")
    print(f"  夏普比率: {result['test_performance']['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {result['test_performance']['max_drawdown']:.2%}")
    
    # 计算衰减
    sharpe_decay = result['test_performance']['sharpe_ratio'] - result['train_performance']['sharpe_ratio']
    return_decay = result['test_performance']['annual_return'] - result['train_performance']['annual_return']
    
    print(f"\n样本外衰减:")
    print(f"  夏普比率衰减: {sharpe_decay:.3f}")
    print(f"  年化收益衰减: {return_decay:.2%}")
    
    # 保存结果
    output_file = '/Users/xinyutan/Documents/量化投资/quant-project/results/oos_validation_fixed_split.json'
    with open(output_file, 'w') as f:
        # 转换不可序列化的对象
        result_serializable = {
            'best_params': result['best_params'],
            'train_performance': result['train_performance'],
            'val_performance': result['val_performance'],
            'test_performance': result['test_performance'],
        }
        json.dump(result_serializable, f, indent=2, default=str)
    print(f"\n结果已保存: {output_file}")
    
    return result


def run_rolling_validation(returns_df, ep_df):
    """运行滚动窗口验证"""
    print("\n" + "="*60)
    print("滚动窗口样本外验证")
    print("="*60)
    
    config = ValidationConfig(
        train_start='2006-01-01',
        train_end='2025-12-31',
        window_train_years=5,
        window_test_years=1,
        step_years=1,
        param_grid={
            'ep_weight': [0.3, 0.4, 0.5, 0.6],
            'reversal_months': [3, 6, 12],
            'top_k': [50, 100],
        }
    )
    
    validator = OutOfSampleValidator(config)
    backtest_func = create_default_backtest_func(returns_df, ep_df)
    
    summary_df = validator.run_rolling_validation(backtest_func, scoring='sharpe_ratio')
    
    # 打印汇总结果
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
    output_file = '/Users/xinyutan/Documents/量化投资/quant-project/results/oos_validation_rolling.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")
    
    return summary_df


def main():
    """主函数"""
    print("="*60)
    print("月度组合策略样本外验证")
    print("="*60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据
    returns_df, ep_df = load_data()
    
    # 运行固定划分验证
    fixed_result = run_fixed_split_validation(returns_df, ep_df)
    
    # 运行滚动窗口验证
    rolling_result = run_rolling_validation(returns_df, ep_df)
    
    print("\n" + "="*60)
    print("样本外验证完成")
    print("="*60)


if __name__ == '__main__':
    main()
