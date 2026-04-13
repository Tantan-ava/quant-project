# -*- coding: utf-8 -*-
"""
混合频率策略自动调优与稳健性验证主脚本

执行流程：
1. 自动参数搜索（3折时间序列交叉验证）
2. 稳健性验证（滚动窗口 + Bootstrap）
3. 样本外测试（2025年数据）
4. 保存最优参数配置
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
    主函数：执行完整的参数优化和稳健性验证流程
    """
    
    # 定义搜索空间（防过拟合：参数组合<500种）
    # 约束：panic_threshold 必须 > extreme_threshold（因为是负数，-1.0 > -2.0）
    search_space = {
        'ep_weight': [0.3, 0.4, 0.5, 0.6, 0.7],           # 5种
        'osa_threshold_extreme': [-3.0, -2.5],            # 2种（更极端）
        'osa_threshold_panic': [-1.5, -1.0],              # 2种（较温和，必须 > extreme）
        'osa_threshold_greed': [1.5],                     # 1种，固定使用
        'scalar_extreme': [1.3, 1.5, 1.7],                # 3种
        'scalar_greed': [0.3, 0.5, 0.7],                  # 3种
        'top_k': [50, 100, 150]                           # 3种
    }
    # 实际组合数：5×2×2×3×3×3 = 540种（考虑panic>extreme约束后全部有效）
    
    print("="*60)
    print("混合频率策略自动调优与稳健性验证")
    print("="*60)
    
    # 获取交易日序列（用于时间序列分割）
    # 注意：仅使用2022-2024年数据用于调优，2025年保留为样本外
    try:
        daily_returns = pd.read_csv('data/raw/TRD-daily.csv', index_col=0, parse_dates=True)
        trading_days = daily_returns.index[
            (daily_returns.index >= '2022-02-01') & 
            (daily_returns.index <= '2024-12-31')
        ]
        print(f"\n加载交易日序列: {len(trading_days)} 天")
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("使用默认交易日序列...")
        trading_days = pd.date_range('2022-02-01', '2024-12-31', freq='B')
    
    # Step 1: 自动调优（交叉验证）
    print("\n[Step 1] 自动参数搜索（3折时间序列交叉验证）...")
    optimizer = StrategyOptimizer(search_space, scoring='sharpe', n_splits=3)
    top_candidates = optimizer.grid_search(HybridBacktester, trading_days)
    
    print(f"\nTop 3 参数组合（验证集夏普）：")
    for i, candidate in enumerate(top_candidates[:3], 1):
        print(f"{i}. 夏普: {candidate['cv_score']:.3f} (±{candidate['cv_std']:.3f})")
        print(f"   参数: EP={candidate['params']['ep_weight']}, "
              f"OSA阈值={candidate['params']['osa_threshold_extreme']}, "
              f"超配={candidate['params']['scalar_extreme']}x")
    
    # Step 2: 稳健性验证（Top 3逐一验证）
    print("\n[Step 2] 稳健性验证（滚动窗口 + Bootstrap）...")
    final_candidates = []
    
    for candidate in top_candidates[:3]:
        params = candidate['params']
        validator = RobustnessValidator(HybridBacktester, params)
        
        # 2.1 滚动窗口稳定性
        print(f"\n验证参数组合: EP={params['ep_weight']}")
        rolling = validator.rolling_window_test(
            full_period=('2022-02-01', '2024-12-31'),
            window_months=12,
            step_months=3
        )
        print(f"  滚动窗口CV: {rolling['cv_value']:.3f} "
              f"({'通过' if rolling['is_stable'] else '失败'})")
        
        # 2.2 Bootstrap置信区间
        bootstrap = validator.bootstrap_test(n_bootstrap=500)
        print(f"  Bootstrap夏普: {bootstrap['original_sharpe']:.3f} "
              f"[{bootstrap['ci_lower']:.3f}, {bootstrap['ci_upper']:.3f}]")
        
        # 只有通过稳定性检验的才进入下一步
        if rolling['is_stable']:
            final_candidates.append({
                'params': params,
                'cv_score': candidate['cv_score'],
                'rolling': rolling,
                'bootstrap': bootstrap
            })
    
    if not final_candidates:
        print("\n警告：所有参数组合均未通过稳健性检验，建议放宽约束或检查数据")
        return
    
    # Step 3: 样本外测试（2025年）
    print("\n[Step 3] 样本外测试（2025年数据）...")
    best_candidate = final_candidates[0]  # 取第一个通过验证的
    validator = RobustnessValidator(HybridBacktester, best_candidate['params'])
    
    try:
        oos = validator.out_of_sample_test('2025-01-01', '2025-12-31')
        print(f"样本外夏普: {oos['oos_sharpe']:.3f} ({'通过' if oos['is_positive'] else '失败'})")
        print(f"样本外收益: {oos['oos_return']*100:.2f}%, 最大回撤: {oos['oos_maxdd']*100:.2f}%")
    except Exception as e:
        print(f"样本外测试失败: {e}")
        oos = {'is_positive': False, 'oos_sharpe': -999}
    
    # Step 4: 保存最终配置
    final_config = {
        'strategy': {
            'name': 'Hybrid_EP_Reversal_Sentiment',
            'description': 'EP价值 + K=6反转 + 情绪调仓'
        },
        'optimized_params': best_candidate['params'],
        'performance': {
            'cv_sharpe': float(best_candidate['cv_score']),
            'rolling_cv': float(best_candidate['rolling']['cv_value']),
            'rolling_sharpe_mean': float(best_candidate['rolling']['sharpe_mean']),
            'bootstrap_ci': [
                float(best_candidate['bootstrap']['ci_lower']),
                float(best_candidate['bootstrap']['ci_upper'])
            ],
            'bootstrap_mean': float(best_candidate['bootstrap']['bootstrap_mean']),
            'oos_sharpe': float(oos.get('oos_sharpe', -999)),
            'oos_return': float(oos.get('oos_return', 0)),
            'oos_maxdd': float(oos.get('oos_maxdd', 0))
        },
        'data_paths': {
            'daily_returns': 'data/raw/TRD-daily.csv',
            'sentiment_index': 'data/processed/daily_sentiment_index.csv'
        }
    }
    
    # 保存为JSON
    output_dir = 'config'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'hybrid_strategy_optimized.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_config, f, indent=2, ensure_ascii=False)
    
    print(f"\n最优参数已保存至: {output_file}")
    
    if oos.get('is_positive', False):
        print("\n✅ 优化完成！策略通过所有检验")
    else:
        print("\n⚠️ 样本外测试失败，策略可能过拟合，建议调整搜索空间")
    
    # 打印最终参数
    print("\n" + "="*60)
    print("最优参数:")
    for key, value in best_candidate['params'].items():
        print(f"  {key}: {value}")
    print("="*60)


if __name__ == '__main__':
    main()
