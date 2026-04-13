# -*- coding: utf-8 -*-
"""
混合频率策略自动调优与稳健性验证 - 全数据版本

使用2022年至2026年4月的全部日度收益数据
"""

import sys
import os
import pandas as pd
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimization.auto_tune import StrategyOptimizer
from optimization.robustness_check import RobustnessValidator
from backtest.hybrid_backtester import HybridBacktester


def main():
    """
    主函数：使用全数据执行参数优化
    """
    
    # 定义搜索空间（36种组合）
    search_space = {
        'ep_weight': [0.4, 0.5, 0.6],
        'osa_threshold_extreme': [-2.5],
        'osa_threshold_panic': [-1.5],
        'osa_threshold_greed': [1.5],
        'scalar_extreme': [1.5, 1.7],
        'scalar_greed': [0.5, 0.7],
        'top_k': [50, 100, 150]
    }
    
    print("="*60)
    print("混合频率策略自动调优与稳健性验证（全数据版本）")
    print("="*60)
    
    # 获取交易日序列（2022年至2026年4月全部数据）
    try:
        daily_returns = pd.read_csv('data/raw/TRD-daily.csv', index_col=0, parse_dates=True)
        # 使用全部可用数据
        trading_days = daily_returns.index[daily_returns.index >= '2022-02-01']
        print(f"\n加载交易日序列: {len(trading_days)} 天")
        print(f"数据期间: {trading_days[0].strftime('%Y-%m-%d')} 至 {trading_days[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # Step 1: 自动调优（交叉验证）
    print("\n[Step 1] 自动参数搜索（3折时间序列交叉验证）...")
    optimizer = StrategyOptimizer(search_space, scoring='sharpe', n_splits=3)
    top_candidates = optimizer.grid_search(HybridBacktester, trading_days)
    
    print(f"\nTop 3 参数组合（验证集夏普）：")
    for i, candidate in enumerate(top_candidates[:3], 1):
        print(f"{i}. 夏普: {candidate['cv_score']:.3f} (±{candidate['cv_std']:.3f})")
        print(f"   参数: EP={candidate['params']['ep_weight']}, "
              f"反转={candidate['params']['reversal_weight']}, "
              f"选股数={candidate['params']['top_k']}, "
              f"超配={candidate['params']['scalar_extreme']}x")
    
    if not top_candidates or top_candidates[0]['cv_score'] < -900:
        print("\n⚠️ 没有找到有效的参数组合")
        return
    
    # Step 2: 稳健性验证（Top 3逐一验证）
    print("\n[Step 2] 稳健性验证（滚动窗口 + Bootstrap）...")
    final_candidates = []
    
    for candidate in top_candidates[:3]:
        params = candidate['params']
        validator = RobustnessValidator(HybridBacktester, params)
        
        # 2.1 滚动窗口稳定性
        print(f"\n验证参数组合: EP={params['ep_weight']}, TopK={params['top_k']}")
        rolling = validator.rolling_window_test(
            full_period=('2022-02-01', trading_days[-1].strftime('%Y-%m-%d')),
            window_months=12,
            step_months=3
        )
        print(f"  滚动窗口CV: {rolling['cv_value']:.3f} "
              f"({'通过' if rolling['is_stable'] else '失败'})")
        
        # 2.2 Bootstrap置信区间
        bootstrap = validator.bootstrap_test(n_bootstrap=500)
        print(f"  Bootstrap夏普: {bootstrap['original_sharpe']:.3f} "
              f"[{bootstrap['ci_lower']:.3f}, {bootstrap['ci_upper']:.3f}]")
        
        final_candidates.append({
            'params': params,
            'cv_score': candidate['cv_score'],
            'rolling': rolling,
            'bootstrap': bootstrap
        })
    
    # 选择最佳候选（按CV夏普排序）
    best_candidate = max(final_candidates, key=lambda x: x['cv_score'])
    
    # Step 3: 全样本回测
    print("\n[Step 3] 全样本回测（2022-2026年全部数据）...")
    validator = RobustnessValidator(HybridBacktester, best_candidate['params'])
    
    try:
        bt = HybridBacktester(**best_candidate['params'])
        full_results = bt.run()
        
        # 计算全样本指标
        returns = full_results['daily_return'].dropna()
        total_return = (full_results['portfolio_value'].iloc[-1] / full_results['portfolio_value'].iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * (252 ** 0.5)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 计算最大回撤
        cummax = full_results['portfolio_value'].cummax()
        drawdown = (full_results['portfolio_value'] - cummax) / cummax
        max_dd = drawdown.min()
        
        print(f"全样本夏普: {sharpe:.3f}")
        print(f"全样本收益: {total_return*100:.2f}%")
        print(f"年化收益: {annual_return*100:.2f}%")
        print(f"最大回撤: {max_dd*100:.2f}%")
        
        full_sample = {
            'sharpe': sharpe,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_dd': max_dd
        }
    except Exception as e:
        print(f"全样本回测失败: {e}")
        full_sample = {'sharpe': -999, 'total_return': 0, 'annual_return': 0, 'max_dd': 0}
    
    # Step 4: 保存最终配置
    final_config = {
        'strategy': {
            'name': 'Hybrid_EP_Reversal_Sentiment',
            'description': 'EP价值 + K=6反转 + 情绪调仓（全数据优化）'
        },
        'optimized_params': best_candidate['params'],
        'performance': {
            'cv_sharpe': float(best_candidate['cv_score']),
            'rolling_cv': float(best_candidate['rolling']['cv_value']),
            'rolling_sharpe_mean': float(best_candidate['rolling'].get('sharpe_mean', 0)),
            'bootstrap_ci': [
                float(best_candidate['bootstrap'].get('ci_lower', -999)),
                float(best_candidate['bootstrap'].get('ci_upper', -999))
            ],
            'bootstrap_mean': float(best_candidate['bootstrap'].get('bootstrap_mean', 0)),
            'full_sample_sharpe': float(full_sample['sharpe']),
            'full_sample_return': float(full_sample['total_return']),
            'full_sample_annual_return': float(full_sample['annual_return']),
            'full_sample_maxdd': float(full_sample['max_dd'])
        },
        'data_info': {
            'start_date': trading_days[0].strftime('%Y-%m-%d'),
            'end_date': trading_days[-1].strftime('%Y-%m-%d'),
            'total_days': len(trading_days)
        },
        'data_paths': {
            'daily_returns': 'data/raw/TRD-daily.csv',
            'sentiment_index': 'data/processed/daily_sentiment_index.csv'
        }
    }
    
    # 保存为JSON
    output_dir = 'config'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'hybrid_strategy_optimized_full.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_config, f, indent=2, ensure_ascii=False)
    
    print(f"\n最优参数已保存至: {output_file}")
    
    # 打印最终参数
    print("\n" + "="*60)
    print("最优参数:")
    for key, value in best_candidate['params'].items():
        print(f"  {key}: {value}")
    print("="*60)
    print(f"\n全样本表现:")
    print(f"  夏普比率: {full_sample['sharpe']:.3f}")
    print(f"  总收益: {full_sample['total_return']*100:.2f}%")
    print(f"  年化收益: {full_sample['annual_return']*100:.2f}%")
    print(f"  最大回撤: {full_sample['max_dd']*100:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()