#!/usr/bin/env python3
"""
样本外验证框架 - Out-of-Sample Validation Framework

设计原则:
1. 时间序列交叉验证 - 训练集必须在验证集之前
2. 滚动窗口回测 - 多期样本外验证
3. 参数优化与选择分离 - 避免数据窥探偏差

作者: Assistant
日期: 2026-04-13
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm
import sys
sys.path.append('/Users/xinyutan/Documents/量化投资/quant-project')


@dataclass
class ValidationConfig:
    """样本外验证配置"""
    # 数据划分
    train_start: str = '2006-01-01'
    train_end: str = '2018-12-31'
    val_start: str = '2019-01-01'
    val_end: str = '2022-12-31'
    test_start: str = '2023-01-01'
    test_end: str = '2025-12-31'
    
    # 滚动窗口参数
    window_train_years: int = 5  # 训练窗口长度
    window_test_years: int = 1   # 测试窗口长度
    step_years: int = 1          # 滚动步长
    
    # 参数搜索空间
    param_grid: Dict = None
    
    def __post_init__(self):
        if self.param_grid is None:
            # 默认参数搜索空间
            self.param_grid = {
                'ep_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
                'reversal_months': [3, 6, 12],
                'top_k': [50, 100, 150, 200],
            }


@dataclass
class BacktestResult:
    """回测结果"""
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    total_return: float
    monthly_returns: pd.Series
    
    def to_dict(self):
        return {
            'annual_return': self.annual_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'total_return': self.total_return,
        }


class OutOfSampleValidator:
    """样本外验证器"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.results = {}
        
    def generate_rolling_windows(self, 
                                  start_date: str, 
                                  end_date: str) -> List[Tuple[str, str, str, str]]:
        """
        生成滚动窗口
        
        Returns:
            List of (train_start, train_end, test_start, test_end)
        """
        windows = []
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        train_days = self.config.window_train_years * 365
        test_days = self.config.window_test_years * 365
        step_days = self.config.step_years * 365
        
        current_start = start
        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days)
            
            if test_end > end:
                break
                
            windows.append((
                train_start.strftime('%Y-%m-%d'),
                train_end.strftime('%Y-%m-%d'),
                test_start.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d')
            ))
            
            current_start = current_start + timedelta(days=step_days)
            
        return windows
    
    def evaluate_params(self,
                       param_set: Dict,
                       train_start: str,
                       train_end: str,
                       backtest_func: Callable) -> BacktestResult:
        """
        在指定时间段内评估参数
        
        Args:
            param_set: 参数字典
            train_start: 训练开始日期
            train_end: 训练结束日期
            backtest_func: 回测函数
            
        Returns:
            BacktestResult
        """
        result = backtest_func(
            start_date=train_start,
            end_date=train_end,
            **param_set
        )
        return result
    
    def parameter_search(self,
                        train_start: str,
                        train_end: str,
                        backtest_func: Callable,
                        scoring: str = 'sharpe_ratio') -> Tuple[Dict, pd.DataFrame]:
        """
        参数搜索 - 在训练集上优化参数
        
        Args:
            train_start: 训练开始日期
            train_end: 训练结束日期
            backtest_func: 回测函数
            scoring: 评分指标
            
        Returns:
            (最优参数, 所有结果DataFrame)
        """
        results = []
        param_grid = self.config.param_grid
        
        # 生成所有参数组合
        from itertools import product
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        print(f"参数搜索: {len(list(product(*values)))} 种组合")
        
        for combo in tqdm(product(*values), total=len(list(product(*values)))):
            param_set = dict(zip(keys, combo))
            
            try:
                result = self.evaluate_params(
                    param_set, train_start, train_end, backtest_func
                )
                
                row = param_set.copy()
                row.update(result.to_dict())
                results.append(row)
                
            except Exception as e:
                print(f"参数 {param_set} 回测失败: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        # 选择最优参数
        if scoring == 'sharpe_ratio':
            best_idx = results_df['sharpe_ratio'].idxmax()
        elif scoring == 'annual_return':
            best_idx = results_df['annual_return'].idxmax()
        elif scoring == 'calmar_ratio':
            best_idx = results_df['calmar_ratio'].idxmax()
        else:
            best_idx = results_df['sharpe_ratio'].idxmax()
            
        best_params = results_df.loc[best_idx, keys].to_dict()
        
        return best_params, results_df
    
    def validate_single_window(self,
                              train_start: str,
                              train_end: str,
                              test_start: str,
                              test_end: str,
                              backtest_func: Callable,
                              scoring: str = 'sharpe_ratio') -> Dict:
        """
        验证单个滚动窗口
        
        Returns:
            包含训练集和测试集结果的字典
        """
        print(f"\n{'='*60}")
        print(f"窗口: {train_start} ~ {test_end}")
        print(f"训练集: {train_start} ~ {train_end}")
        print(f"测试集: {test_start} ~ {test_end}")
        print(f"{'='*60}")
        
        # 1. 在训练集上搜索最优参数
        print("\n[1/3] 训练集参数搜索...")
        best_params, train_results = self.parameter_search(
            train_start, train_end, backtest_func, scoring
        )
        print(f"最优参数: {best_params}")
        
        # 2. 在训练集上评估最优参数
        print("\n[2/3] 训练集回测...")
        train_backtest = self.evaluate_params(
            best_params, train_start, train_end, backtest_func
        )
        
        # 3. 在测试集上评估最优参数（真正的样本外）
        print("\n[3/3] 测试集样本外验证...")
        test_backtest = self.evaluate_params(
            best_params, test_start, test_end, backtest_func
        )
        
        return {
            'window': f"{train_start}_{test_end}",
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'best_params': best_params,
            'train_results': train_results,
            'train_performance': train_backtest.to_dict(),
            'test_performance': test_backtest.to_dict(),
        }
    
    def run_rolling_validation(self,
                              backtest_func: Callable,
                              scoring: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        运行滚动窗口验证
        
        Returns:
            所有窗口的结果DataFrame
        """
        windows = self.generate_rolling_windows(
            self.config.train_start,
            self.config.test_end
        )
        
        print(f"\n总共 {len(windows)} 个滚动窗口")
        
        all_results = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\n{'#'*60}")
            print(f"# 窗口 {i+1}/{len(windows)}")
            print(f"{'#'*60}")
            
            result = self.validate_single_window(
                train_start, train_end, test_start, test_end,
                backtest_func, scoring
            )
            all_results.append(result)
            
        # 汇总结果
        summary = []
        for r in all_results:
            summary.append({
                'window': r['window'],
                'train_sharpe': r['train_performance']['sharpe_ratio'],
                'test_sharpe': r['test_performance']['sharpe_ratio'],
                'train_return': r['train_performance']['annual_return'],
                'test_return': r['test_performance']['annual_return'],
                'train_maxdd': r['train_performance']['max_drawdown'],
                'test_maxdd': r['test_performance']['max_drawdown'],
                'best_ep_weight': r['best_params'].get('ep_weight'),
                'best_reversal_months': r['best_params'].get('reversal_months'),
                'best_top_k': r['best_params'].get('top_k'),
            })
        
        summary_df = pd.DataFrame(summary)
        
        # 计算样本外稳定性指标
        summary_df['sharpe_decay'] = summary_df['test_sharpe'] - summary_df['train_sharpe']
        summary_df['return_decay'] = summary_df['test_return'] - summary_df['train_return']
        
        self.results['rolling_summary'] = summary_df
        self.results['all_windows'] = all_results
        
        return summary_df
    
    def run_fixed_split_validation(self,
                                   backtest_func: Callable,
                                   scoring: str = 'sharpe_ratio') -> Dict:
        """
        运行固定划分验证（训练/验证/测试）
        
        Returns:
            验证结果字典
        """
        print(f"\n{'='*60}")
        print("固定划分样本外验证")
        print(f"{'='*60}")
        
        # 1. 在训练集上搜索最优参数
        print("\n[1/4] 训练集参数搜索...")
        best_params, train_results = self.parameter_search(
            self.config.train_start,
            self.config.train_end,
            backtest_func,
            scoring
        )
        print(f"最优参数: {best_params}")
        
        # 2. 在训练集上评估
        print("\n[2/4] 训练集回测...")
        train_backtest = self.evaluate_params(
            best_params,
            self.config.train_start,
            self.config.train_end,
            backtest_func
        )
        
        # 3. 在验证集上评估（用于模型选择，非最终测试）
        print("\n[3/4] 验证集回测...")
        val_backtest = self.evaluate_params(
            best_params,
            self.config.val_start,
            self.config.val_end,
            backtest_func
        )
        
        # 4. 在测试集上评估（真正的样本外，仅使用一次）
        print("\n[4/4] 测试集样本外验证（最终评估）...")
        test_backtest = self.evaluate_params(
            best_params,
            self.config.test_start,
            self.config.test_end,
            backtest_func
        )
        
        result = {
            'best_params': best_params,
            'train_performance': train_backtest.to_dict(),
            'val_performance': val_backtest.to_dict(),
            'test_performance': test_backtest.to_dict(),
            'train_results': train_results,
        }
        
        self.results['fixed_split'] = result
        
        return result
    
    def generate_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("="*60)
        report.append("样本外验证报告")
        report.append("="*60)
        
        # 固定划分结果
        if 'fixed_split' in self.results:
            r = self.results['fixed_split']
            report.append("\n【固定划分验证结果】")
            report.append(f"最优参数: {r['best_params']}")
            report.append(f"\n训练集 ({self.config.train_start} ~ {self.config.train_end}):")
            report.append(f"  年化收益: {r['train_performance']['annual_return']:.2%}")
            report.append(f"  夏普比率: {r['train_performance']['sharpe_ratio']:.3f}")
            report.append(f"  最大回撤: {r['train_performance']['max_drawdown']:.2%}")
            
            report.append(f"\n验证集 ({self.config.val_start} ~ {self.config.val_end}):")
            report.append(f"  年化收益: {r['val_performance']['annual_return']:.2%}")
            report.append(f"  夏普比率: {r['val_performance']['sharpe_ratio']:.3f}")
            report.append(f"  最大回撤: {r['val_performance']['max_drawdown']:.2%}")
            
            report.append(f"\n测试集 ({self.config.test_start} ~ {self.config.test_end}) - 样本外:")
            report.append(f"  年化收益: {r['test_performance']['annual_return']:.2%}")
            report.append(f"  夏普比率: {r['test_performance']['sharpe_ratio']:.3f}")
            report.append(f"  最大回撤: {r['test_performance']['max_drawdown']:.2%}")
            
            # 计算衰减
            sharpe_decay = r['test_performance']['sharpe_ratio'] - r['train_performance']['sharpe_ratio']
            report.append(f"\n样本外衰减:")
            report.append(f"  夏普比率衰减: {sharpe_decay:.3f}")
        
        # 滚动窗口结果
        if 'rolling_summary' in self.results:
            df = self.results['rolling_summary']
            report.append("\n【滚动窗口验证结果】")
            report.append(f"窗口数量: {len(df)}")
            report.append(f"\n样本外夏普比率:")
            report.append(f"  均值: {df['test_sharpe'].mean():.3f}")
            report.append(f"  标准差: {df['test_sharpe'].std():.3f}")
            report.append(f"  最小值: {df['test_sharpe'].min():.3f}")
            report.append(f"  最大值: {df['test_sharpe'].max():.3f}")
            
            report.append(f"\n样本外年化收益:")
            report.append(f"  均值: {df['test_return'].mean():.2%}")
            report.append(f"  标准差: {df['test_return'].std():.2%}")
            
            report.append(f"\n夏普比率衰减 (测试集 - 训练集):")
            report.append(f"  均值: {df['sharpe_decay'].mean():.3f}")
            report.append(f"  标准差: {df['sharpe_decay'].std():.3f}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


def create_default_backtest_func(monthly_returns_df: pd.DataFrame,
                                 ep_data_df: pd.DataFrame):
    """
    创建默认的回测函数
    
    Args:
        monthly_returns_df: 月度收益率数据
        ep_data_df: EP因子数据
        
    Returns:
        回测函数
    """
    def backtest_func(start_date: str,
                     end_date: str,
                     ep_weight: float = 0.4,
                     reversal_months: int = 6,
                     top_k: int = 100,
                     **kwargs) -> BacktestResult:
        """
        月度组合策略回测函数
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            ep_weight: EP因子权重
            reversal_months: 反转形成期（月）
            top_k: 选股数量
            
        Returns:
            BacktestResult
        """
        # 筛选日期范围内的数据
        mask = (monthly_returns_df.index >= start_date) & (monthly_returns_df.index <= end_date)
        returns_slice = monthly_returns_df.loc[mask]
        
        if len(returns_slice) == 0:
            raise ValueError(f"日期范围 {start_date} ~ {end_date} 无数据")
        
        # 获取完整的月度数据用于计算信号
        all_returns = monthly_returns_df.loc[:end_date]
        
        portfolio_values = [1.0]
        monthly_rets = []
        
        # 从第reversal_months+1个月开始
        start_idx = reversal_months + 1
        
        for i in range(start_idx, len(all_returns)):
            date = all_returns.index[i]
            
            # 只记录在目标区间内的收益
            if date < start_date:
                continue
            if date > end_date:
                break
            
            # 计算信号
            # 1. EP信号（使用当前可获得的最新EP数据）
            ep_signal = ep_data_df.loc[date] if date in ep_data_df.index else pd.Series()
            
            # 2. 反转信号
            past_returns = all_returns.iloc[i-reversal_months:i]
            reversal_signal = -past_returns.mean()  # 负的过去收益 = 反转信号
            
            # 对齐股票
            common_stocks = ep_signal.index.intersection(reversal_signal.index)
            if len(common_stocks) == 0:
                monthly_rets.append(0)
                portfolio_values.append(portfolio_values[-1] * (1 + 0))
                continue
            
            ep_signal = ep_signal.loc[common_stocks]
            reversal_signal = reversal_signal.loc[common_stocks]
            
            # 标准化信号
            ep_signal = (ep_signal - ep_signal.mean()) / ep_signal.std()
            reversal_signal = (reversal_signal - reversal_signal.mean()) / reversal_signal.std()
            
            # 合成信号
            composite_signal = ep_weight * ep_signal + (1 - ep_weight) * reversal_signal
            
            # 选股
            selected = composite_signal.nlargest(top_k)
            
            # 计算当月收益
            month_return = all_returns.iloc[i]
            available_stocks = month_return.index.intersection(selected.index)
            
            if len(available_stocks) == 0:
                monthly_ret = 0
            else:
                # 等权重
                weights = pd.Series(1/len(available_stocks), index=available_stocks)
                monthly_ret = (month_return.loc[available_stocks] * weights).sum()
            
            monthly_rets.append(monthly_ret)
            portfolio_values.append(portfolio_values[-1] * (1 + monthly_ret))
        
        # 计算绩效指标
        monthly_rets = pd.Series(monthly_rets)
        
        if len(monthly_rets) == 0:
            return BacktestResult(
                annual_return=0,
                volatility=0,
                sharpe_ratio=0,
                max_drawdown=0,
                calmar_ratio=0,
                total_return=0,
                monthly_returns=monthly_rets
            )
        
        total_return = portfolio_values[-1] - 1
        annual_return = (1 + total_return) ** (12 / len(monthly_rets)) - 1
        volatility = monthly_rets.std() * np.sqrt(12)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        cumulative = (1 + monthly_rets).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return BacktestResult(
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_return=total_return,
            monthly_returns=monthly_rets
        )
    
    return backtest_func


if __name__ == '__main__':
    # 示例用法
    config = ValidationConfig()
    validator = OutOfSampleValidator(config)
    
    print("样本外验证框架初始化完成")
    print(f"参数搜索空间: {config.param_grid}")
    print(f"\n滚动窗口设置:")
    print(f"  训练窗口: {config.window_train_years}年")
    print(f"  测试窗口: {config.window_test_years}年")
    print(f"  滚动步长: {config.step_years}年")
