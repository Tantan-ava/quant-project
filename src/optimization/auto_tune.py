# -*- coding: utf-8 -*-
"""
策略参数自动调优模块

网格搜索 + 时间序列交叉验证
"""

import itertools
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest.hybrid_backtester import HybridBacktester


class StrategyOptimizer:
    """
    策略参数优化器
    
    使用网格搜索 + 时间序列交叉验证
    强制约束：训练集必须在验证集之前（避免未来函数）
    """
    
    def __init__(self, search_space, scoring='sharpe', n_splits=3):
        """
        初始化优化器
        
        Parameters:
        -----------
        search_space : dict
            参数搜索空间
        scoring : str
            评分指标 ('sharpe' 或 'calmar')
        n_splits : int
            交叉验证折数
        """
        self.search_space = search_space
        self.scoring = scoring
        self.n_splits = n_splits
        self.results = []
        
    def grid_search(self, backtest_engine, trading_days):
        """
        网格搜索 + 时间序列交叉验证
        
        Parameters:
        -----------
        backtest_engine : class
            回测引擎类
        trading_days : pd.DatetimeIndex
            交易日序列
            
        Returns:
        --------
        list
            Top 10 参数组合
        """
        # 生成参数组合
        param_combinations = self._generate_param_grid()
        print(f"总参数组合数: {len(param_combinations)}")
        
        # 时间序列分割（训练集→验证集，无重叠）
        fold_size = len(trading_days) // self.n_splits
        
        for idx, params in enumerate(param_combinations):
            if idx % 10 == 0:
                print(f"进度: {idx}/{len(param_combinations)}")
            
            cv_scores = []
            
            for fold in range(self.n_splits):
                # 计算训练集和验证集索引
                val_start_idx = fold * fold_size
                val_end_idx = (fold + 1) * fold_size if fold < self.n_splits - 1 else len(trading_days)
                
                train_days = trading_days[:val_start_idx]
                val_days = trading_days[val_start_idx:val_end_idx]
                
                if len(train_days) < 60 or len(val_days) < 20:
                    # 数据不足，跳过
                    cv_scores.append(-999)
                    continue
                
                # 验证集回测（用于评分）
                try:
                    score = self._evaluate_single_fold(
                        params, backtest_engine, 
                        train_days[0], val_days[-1], val_days
                    )
                    cv_scores.append(score)
                except Exception as e:
                    print(f"  参数回测失败: {e}")
                    cv_scores.append(-999)
            
            # 平均验证集得分
            valid_scores = [s for s in cv_scores if s > -900]
            avg_score = np.mean(valid_scores) if valid_scores else -999
            
            self.results.append({
                'params': params,
                'cv_score': avg_score,
                'cv_std': np.std(valid_scores) if valid_scores else 0,
                'cv_scores': cv_scores
            })
        
        # 按验证集夏普排序
        self.results.sort(key=lambda x: x['cv_score'], reverse=True)
        return self.results[:10]
    
    def _generate_param_grid(self):
        """
        生成参数网格
        
        约束：
        - ep_weight + reversal_weight = 1.0
        - panic阈值必须比extreme更温和（即 panic > extreme，例如 -1.0 > -2.0）
        """
        grid = []
        for ep in self.search_space['ep_weight']:
            for extreme in self.search_space['osa_threshold_extreme']:
                for panic in self.search_space['osa_threshold_panic']:
                    # 约束：panic必须比extreme更接近0（即 panic > extreme）
                    # 例如：extreme=-2.5, panic=-1.5 是有效的（-1.5 > -2.5）
                    if panic <= extreme:
                        continue
                    for scalar_ext in self.search_space['scalar_extreme']:
                        for scalar_greed in self.search_space['scalar_greed']:
                            for topk in self.search_space['top_k']:
                                grid.append({
                                    'ep_weight': ep,
                                    'reversal_weight': 1.0 - ep,
                                    'osa_threshold_extreme': extreme,
                                    'osa_threshold_panic': panic,
                                    'osa_threshold_greed': self.search_space['osa_threshold_greed'][0],
                                    'scalar_extreme': scalar_ext,
                                    'scalar_greed': scalar_greed,
                                    'top_k': topk
                                })
        return grid
    
    def _evaluate_single_fold(self, params, engine, start, end, val_days):
        """
        评估单折验证集
        
        Parameters:
        -----------
        params : dict
            参数字典
        engine : class
            回测引擎类
        start : pd.Timestamp
            开始日期
        end : pd.Timestamp
            结束日期
        val_days : pd.DatetimeIndex
            验证集交易日
            
        Returns:
        --------
        float
            评分指标
        """
        # 实例化回测引擎
        bt = engine(
            ep_weight=params['ep_weight'],
            reversal_weight=params['reversal_weight'],
            osa_threshold_extreme=params['osa_threshold_extreme'],
            osa_threshold_panic=params['osa_threshold_panic'],
            osa_threshold_greed=params['osa_threshold_greed'],
            scalar_extreme=params['scalar_extreme'],
            scalar_greed=params['scalar_greed'],
            top_k=params['top_k'],
            start_date=start.strftime('%Y-%m-%d'),
            end_date=end.strftime('%Y-%m-%d')
        )
        
        results = bt.run()
        
        # 只在验证集上计算指标
        val_results = results.loc[results.index.intersection(val_days)]
        
        if len(val_results) < 20:
            return -999
        
        if self.scoring == 'sharpe':
            return self._calculate_sharpe(val_results)
        elif self.scoring == 'calmar':
            return self._calculate_calmar(val_results)
        else:
            return -999
    
    def _calculate_sharpe(self, results, annualize=252):
        """
        计算年化夏普比率
        
        Parameters:
        -----------
        results : pd.DataFrame
            回测结果
        annualize : int
            年化因子
            
        Returns:
        --------
        float
            夏普比率
        """
        if len(results) < 20:
            return -999
        
        mean_ret = results['daily_return'].mean() * annualize
        std_ret = results['daily_return'].std() * np.sqrt(annualize)
        
        return mean_ret / std_ret if std_ret > 0 else -999
    
    def _calculate_calmar(self, results, annualize=252):
        """
        计算Calmar比率 = 年化收益 / 最大回撤
        
        Parameters:
        -----------
        results : pd.DataFrame
            回测结果
        annualize : int
            年化因子
            
        Returns:
        --------
        float
            Calmar比率
        """
        if len(results) < 20:
            return -999
        
        # 年化收益
        mean_ret = results['daily_return'].mean() * annualize
        
        # 最大回撤
        cumulative = (1 + results['daily_return']).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = abs(drawdown.min())
        
        return mean_ret / max_dd if max_dd > 0 else -999


# 使用示例
if __name__ == '__main__':
    # 定义搜索空间
    search_space = {
        'ep_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
        'osa_threshold_extreme': [-3.0, -2.5, -2.0],
        'osa_threshold_panic': [-2.0, -1.5, -1.0],
        'osa_threshold_greed': [1.5],
        'scalar_extreme': [1.3, 1.5, 1.7],
        'scalar_greed': [0.3, 0.5, 0.7],
        'top_k': [50, 100, 150]
    }
    
    # 获取交易日序列
    daily_returns = pd.read_csv('data/raw/TRD-daily.csv', index_col=0, parse_dates=True)
    trading_days = daily_returns.index
    
    # 创建优化器
    optimizer = StrategyOptimizer(search_space, scoring='sharpe', n_splits=3)
    
    # 执行网格搜索
    top_results = optimizer.grid_search(HybridBacktester, trading_days)
    
    # 打印结果
    print("\nTop 3 参数组合:")
    for i, result in enumerate(top_results[:3], 1):
        print(f"{i}. 夏普: {result['cv_score']:.3f} (±{result['cv_std']:.3f})")
        print(f"   参数: {result['params']}")