# -*- coding: utf-8 -*-
"""
策略稳健性验证模块

滚动窗口 + Bootstrap 验证
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest.hybrid_backtester import HybridBacktester


class RobustnessValidator:
    """
    策略稳健性验证器
    
    提供滚动窗口检验、Bootstrap验证和样本外测试
    """
    
    def __init__(self, backtest_engine, params):
        """
        初始化验证器
        
        Parameters:
        -----------
        backtest_engine : class
            回测引擎类
        params : dict
            策略参数字典
        """
        self.engine = backtest_engine
        self.params = params
        
    def rolling_window_test(self, full_period, window_months=12, step_months=3):
        """
        滚动窗口稳定性检验
        
        要求：夏普比率变异系数(CV) < 0.3
        
        Parameters:
        -----------
        full_period : tuple
            (start_date, end_date) 全样本期间
        window_months : int
            窗口大小（月）
        step_months : int
            步长（月）
            
        Returns:
        --------
        dict
            检验结果
        """
        start_date, end_date = full_period
        results = []
        
        # 生成滚动窗口
        current = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        window_count = 0
        while current + pd.DateOffset(months=window_months) <= end:
            window_end = current + pd.DateOffset(months=window_months)
            
            try:
                # 回测当前窗口
                bt = self.engine(
                    **self.params,
                    start_date=current.strftime('%Y-%m-%d'),
                    end_date=window_end.strftime('%Y-%m-%d')
                )
                result = bt.run()
                
                sharpe = self._calc_sharpe(result)
                max_dd = self._calc_maxdd(result)
                
                results.append({
                    'window_start': current.strftime('%Y-%m'),
                    'window_end': window_end.strftime('%Y-%m'),
                    'sharpe': sharpe,
                    'max_dd': max_dd
                })
                window_count += 1
                
            except Exception as e:
                print(f"  窗口 {current.strftime('%Y-%m')} 回测失败: {e}")
            
            current += pd.DateOffset(months=step_months)
        
        if len(results) == 0:
            return {
                'is_stable': False,
                'cv_value': 999,
                'sharpe_mean': 0,
                'sharpe_std': 0,
                'details': []
            }
        
        # 稳定性判断
        sharpe_series = pd.Series([r['sharpe'] for r in results])
        mean_sharpe = sharpe_series.mean()
        cv = sharpe_series.std() / abs(mean_sharpe) if mean_sharpe != 0 else 999
        
        return {
            'is_stable': cv < 0.3,
            'cv_value': cv,
            'sharpe_mean': mean_sharpe,
            'sharpe_std': sharpe_series.std(),
            'details': results
        }
    
    def bootstrap_test(self, n_bootstrap=1000, confidence=0.90):
        """
        Bootstrap重采样验证
        
        返回夏普比率的置信区间
        
        Parameters:
        -----------
        n_bootstrap : int
            Bootstrap次数
        confidence : float
            置信水平
            
        Returns:
        --------
        dict
            Bootstrap结果
        """
        # 全样本回测获取原始收益
        try:
            bt = self.engine(**self.params)
            full_results = bt.run()
            returns = full_results['daily_return'].dropna()
        except Exception as e:
            print(f"全样本回测失败: {e}")
            return {
                'original_sharpe': -999,
                'ci_lower': -999,
                'ci_upper': -999,
                'in_interval': False,
                'bootstrap_mean': 0,
                'bootstrap_std': 0
            }
        
        if len(returns) < 50:
            return {
                'original_sharpe': -999,
                'ci_lower': -999,
                'ci_upper': -999,
                'in_interval': False,
                'bootstrap_mean': 0,
                'bootstrap_std': 0
            }
        
        bootstrap_sharpes = []
        for i in range(n_bootstrap):
            if i % 200 == 0:
                print(f"  Bootstrap进度: {i}/{n_bootstrap}")
            
            # 有放回重采样
            sample = returns.sample(n=len(returns), replace=True)
            sharpe = self._calc_sharpe_from_series(sample)
            bootstrap_sharpes.append(sharpe)
        
        # 计算置信区间
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_sharpes, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha/2) * 100)
        
        # 检查原始夏普是否在置信区间内
        original_sharpe = self._calc_sharpe_from_series(returns)
        in_interval = ci_lower <= original_sharpe <= ci_upper
        
        return {
            'original_sharpe': original_sharpe,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'in_interval': in_interval,
            'bootstrap_mean': np.mean(bootstrap_sharpes),
            'bootstrap_std': np.std(bootstrap_sharpes)
        }
    
    def out_of_sample_test(self, oos_start='2025-01-01', oos_end='2025-12-31'):
        """
        样本外测试
        
        要求：夏普 > 0（策略未失效）
        
        Parameters:
        -----------
        oos_start : str
            样本外开始日期
        oos_end : str
            样本外结束日期
            
        Returns:
        --------
        dict
            样本外测试结果
        """
        try:
            bt = self.engine(
                **self.params,
                start_date=oos_start,
                end_date=oos_end
            )
            oos_results = bt.run()
            oos_sharpe = self._calc_sharpe(oos_results)
            
            return {
                'oos_sharpe': oos_sharpe,
                'is_positive': oos_sharpe > 0,
                'oos_return': oos_results['daily_return'].mean() * 252,
                'oos_maxdd': self._calc_maxdd(oos_results)
            }
        except Exception as e:
            print(f"样本外测试失败: {e}")
            return {
                'oos_sharpe': -999,
                'is_positive': False,
                'oos_return': 0,
                'oos_maxdd': 0
            }
    
    def _calc_sharpe(self, results, annualize=252):
        """
        计算夏普比率
        """
        if len(results) < 20:
            return -999
        
        mean_ret = results['daily_return'].mean() * annualize
        std_ret = results['daily_return'].std() * np.sqrt(annualize)
        return mean_ret / std_ret if std_ret > 0 else -999
    
    def _calc_sharpe_from_series(self, returns, annualize=252):
        """
        从收益率序列计算夏普
        """
        mean_ret = returns.mean() * annualize
        std_ret = returns.std() * np.sqrt(annualize)
        return mean_ret / std_ret if std_ret > 0 else -999
    
    def _calc_maxdd(self, results):
        """
        计算最大回撤
        """
        cumulative = (1 + results['daily_return']).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()


# 使用示例
if __name__ == '__main__':
    # 示例参数
    params = {
        'ep_weight': 0.4,
        'reversal_weight': 0.6,
        'top_k': 100,
        'osa_threshold_extreme': -2.0,
        'osa_threshold_panic': -1.0,
        'osa_threshold_greed': 1.5,
        'scalar_extreme': 1.5,
        'scalar_greed': 0.5
    }
    
    # 创建验证器
    validator = RobustnessValidator(HybridBacktester, params)
    
    # 滚动窗口检验
    print("滚动窗口检验...")
    rolling_result = validator.rolling_window_test(
        full_period=('2022-02-01', '2024-12-31'),
        window_months=12,
        step_months=3
    )
    print(f"稳定性: {'通过' if rolling_result['is_stable'] else '失败'}")
    print(f"CV值: {rolling_result['cv_value']:.3f}")
    print(f"平均夏普: {rolling_result['sharpe_mean']:.3f}")
    
    # Bootstrap检验
    print("\nBootstrap检验...")
    bootstrap_result = validator.bootstrap_test(n_bootstrap=500)
    print(f"原始夏普: {bootstrap_result['original_sharpe']:.3f}")
    print(f"90%置信区间: [{bootstrap_result['ci_lower']:.3f}, {bootstrap_result['ci_upper']:.3f}]")
    print(f"在区间内: {'是' if bootstrap_result['in_interval'] else '否'}")
