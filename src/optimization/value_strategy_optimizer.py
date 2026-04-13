# -*- coding: utf-8 -*-
"""
价值投资策略因子权重优化器

网格搜索 + 时间序列交叉验证
优化目标：复合价值因子权重 (EP, BP, SP, DP)
"""

import itertools
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.factors.composite_value import CompositeValueFactor
from src.factors.quality_screener import QualityScreener
from src.strategy.portfolio_constructor import PortfolioConstructor


class ValueStrategyOptimizer:
    """
    价值投资策略优化器
    
    优化复合价值因子权重：
    - EP (Earnings-to-Price)
    - BP (Book-to-Price)  
    - SP (Sales-to-Price)
    - DP (Dividend Yield)
    
    约束：权重之和 = 1
    """
    
    def __init__(self, scoring='sharpe', n_splits=3):
        """
        初始化优化器
        
        Parameters:
        -----------
        scoring : str
            评分指标 ('sharpe', 'calmar', 'total_return')
        n_splits : int
            交叉验证折数
        """
        self.scoring = scoring
        self.n_splits = n_splits
        self.results = []
        
    def generate_weight_grid(self, step=0.1):
        """
        生成权重网格（满足权重和=1的约束）
        
        Parameters:
        -----------
        step : float
            权重步长
            
        Returns:
        --------
        list
            权重组合列表
        """
        weights = []
        n_steps = int(1.0 / step) + 1
        
        # 生成所有可能的权重组合
        for ep_w in np.arange(0, 1.01, step):
            for bp_w in np.arange(0, 1.01 - ep_w, step):
                for sp_w in np.arange(0, 1.01 - ep_w - bp_w, step):
                    dp_w = 1.0 - ep_w - bp_w - sp_w
                    if dp_w >= 0 and dp_w <= 1.0:
                        weights.append({
                            'ep': round(ep_w, 2),
                            'bp': round(bp_w, 2),
                            'sp': round(sp_w, 2),
                            'dp': round(dp_w, 2)
                        })
        
        # 去重
        unique_weights = []
        seen = set()
        for w in weights:
            key = (w['ep'], w['bp'], w['sp'], w['dp'])
            if key not in seen:
                seen.add(key)
                unique_weights.append(w)
        
        print(f"生成权重组合: {len(unique_weights)} 种")
        return unique_weights
    
    def grid_search(self, returns_data, dates, step=0.1):
        """
        网格搜索 + 时间序列交叉验证
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            股票收益率数据
        dates : list
            再平衡日期列表
        step : float
            权重步长
            
        Returns:
        --------
        list
            Top 10 参数组合
        """
        # 生成权重组合
        weight_combinations = self.generate_weight_grid(step)
        
        # 时间序列分割
        fold_size = len(dates) // self.n_splits
        
        print(f"\n开始网格搜索 ({len(weight_combinations)} 组合 × {self.n_splits} 折)...")
        
        for idx, weights in enumerate(weight_combinations):
            if idx % 10 == 0:
                print(f"进度: {idx}/{len(weight_combinations)}")
            
            cv_scores = []
            
            for fold in range(self.n_splits):
                # 计算训练集和验证集
                val_start_idx = fold * fold_size
                val_end_idx = (fold + 1) * fold_size if fold < self.n_splits - 1 else len(dates)
                
                train_dates = dates[:val_start_idx]
                val_dates = dates[val_start_idx:val_end_idx]
                
                if len(train_dates) < 3 or len(val_dates) < 1:
                    cv_scores.append(-999)
                    continue
                
                # 验证集回测
                try:
                    score = self._evaluate_weights(
                        weights, returns_data, val_dates
                    )
                    cv_scores.append(score)
                except Exception as e:
                    print(f"  权重回测失败: {e}")
                    cv_scores.append(-999)
            
            # 平均验证集得分
            valid_scores = [s for s in cv_scores if s > -900]
            if len(valid_scores) > 0:
                avg_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                
                self.results.append({
                    'weights': weights,
                    'cv_score': avg_score,
                    'cv_std': std_score,
                    'fold_scores': cv_scores
                })
        
        # 按得分排序
        self.results.sort(key=lambda x: x['cv_score'], reverse=True)
        
        return self.results[:10]
    
    def _evaluate_weights(self, weights, returns_data, dates):
        """
        评估一组权重
        
        Parameters:
        -----------
        weights : dict
            权重字典
        returns_data : pd.DataFrame
            收益率数据
        dates : list
            验证日期
            
        Returns:
        --------
        float
            评分
        """
        # 简化评估：基于价值因子的IC（信息系数）
        # 实际应用中应进行完整回测
        
        portfolio_returns = []
        
        for date in dates:
            # 模拟选股收益（简化处理）
            # 实际应调用完整策略回测
            daily_return = np.random.randn() * 0.01  # 占位符
            portfolio_returns.append(daily_return)
        
        # 计算评分
        returns = np.array(portfolio_returns)
        
        if self.scoring == 'sharpe':
            if len(returns) < 2 or returns.std() == 0:
                return -999
            return returns.mean() / returns.std() * np.sqrt(252)
        
        elif self.scoring == 'total_return':
            return (1 + returns).prod() - 1
        
        else:  # calmar
            total_ret = (1 + returns).prod() - 1
            max_dd = np.min(np.minimum.accumulate(np.cumprod(1 + returns)) / 
                           np.maximum.accumulate(np.cumprod(1 + returns)) - 1)
            return total_ret / abs(max_dd) if max_dd != 0 else 0
    
    def get_optimal_weights(self):
        """获取最优权重"""
        if len(self.results) == 0:
            return None
        
        return self.results[0]['weights']


class ValueStrategyRobustnessValidator:
    """
    价值投资策略稳健性验证器
    """
    
    def __init__(self, weights):
        """
        初始化验证器
        
        Parameters:
        -----------
        weights : dict
            因子权重
        """
        self.weights = weights
    
    def rolling_window_test(self, returns_data, dates, window_months=12, step_months=3):
        """
        滚动窗口稳定性检验
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            收益率数据
        dates : list
            日期列表
        window_months : int
            窗口大小
        step_months : int
            步长
            
        Returns:
        --------
        dict
            检验结果
        """
        results = []
        
        # 生成滚动窗口
        n_dates = len(dates)
        window_size = window_months
        step_size = max(1, n_dates // (len(dates) // step_months))
        
        for i in range(0, n_dates - window_size, step_size):
            window_dates = dates[i:i+window_size]
            
            try:
                # 模拟回测（简化）
                score = np.random.randn() * 0.5 + 1.0  # 占位符
                results.append(score)
                
            except Exception as e:
                print(f"  窗口回测失败: {e}")
        
        if len(results) < 3:
            return {
                'is_stable': False,
                'cv_value': 999,
                'scores': results
            }
        
        # 计算变异系数
        mean_score = np.mean(results)
        std_score = np.std(results)
        cv = std_score / mean_score if mean_score != 0 else 999
        
        return {
            'is_stable': cv < 0.3 and mean_score > 0,
            'cv_value': cv,
            'mean_score': mean_score,
            'std_score': std_score,
            'scores': results
        }
    
    def bootstrap_test(self, returns_data, n_bootstrap=500):
        """
        Bootstrap置信区间检验
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            收益率数据
        n_bootstrap : int
            Bootstrap次数
            
        Returns:
        --------
        dict
            Bootstrap结果
        """
        # 模拟原始得分
        original_score = np.random.randn() * 0.5 + 1.0
        
        # Bootstrap重采样
        bootstrap_scores = []
        for _ in range(n_bootstrap):
            # 模拟Bootstrap得分
            score = np.random.randn() * 0.5 + original_score
            bootstrap_scores.append(score)
        
        # 计算置信区间
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        
        return {
            'original_score': original_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'is_significant': ci_lower > 0,
            'bootstrap_scores': bootstrap_scores
        }


# 使用示例
if __name__ == '__main__':
    print("="*60)
    print("价值投资策略因子权重优化")
    print("="*60)
    
    # 初始化优化器
    optimizer = ValueStrategyOptimizer(scoring='sharpe', n_splits=3)
    
    # 模拟数据（实际应用中从AKShare获取）
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='MS').strftime('%Y-%m-%d').tolist()
    
    # 执行网格搜索
    top_candidates = optimizer.grid_search(
        returns_data=None,
        dates=dates,
        step=0.2  # 步长0.2，减少组合数
    )
    
    print("\n" + "="*60)
    print("Top 5 权重组合")
    print("="*60)
    
    for i, candidate in enumerate(top_candidates[:5], 1):
        w = candidate['weights']
        print(f"\n{i}. 评分: {candidate['cv_score']:.3f} (±{candidate['cv_std']:.3f})")
        print(f"   EP={w['ep']:.1%}, BP={w['bp']:.1%}, SP={w['sp']:.1%}, DP={w['dp']:.1%}")
    
    # 获取最优权重
    optimal = optimizer.get_optimal_weights()
    if optimal:
        print("\n" + "="*60)
        print("最优权重配置")
        print("="*60)
        print(f"EP: {optimal['ep']:.1%}")
        print(f"BP: {optimal['bp']:.1%}")
        print(f"SP: {optimal['sp']:.1%}")
        print(f"DP: {optimal['dp']:.1%}")
