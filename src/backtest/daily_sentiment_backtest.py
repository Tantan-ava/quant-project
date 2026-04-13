# -*- coding: utf-8 -*-
"""
纯日度调仓策略回测
基于情绪指数进行仓位调整，不择股（持有市场组合）
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.strategy.daily_tactician import DailyTactician


class DailySentimentBacktester:
    """
    纯日度调仓策略回测器
    
    策略逻辑：
    - 不择股，始终持有市场等权组合
    - 每日根据情绪指数调整仓位系数
    - 无交易成本
    """
    
    def __init__(self,
                 daily_returns_path='data/raw/TRD-daily.csv',
                 sentiment_path='data/processed/daily_sentiment_index.csv',
                 start_date='2022-02-01',
                 end_date='2026-04-10',
                 initial_capital=1e8):
        """
        初始化回测引擎
        
        Parameters:
        -----------
        daily_returns_path : str
            日度收益率数据路径
        sentiment_path : str
            日度情绪指数路径
        start_date : str
            回测开始日期
        end_date : str
            回测结束日期
        initial_capital : float
            初始资金
        """
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        
        # 加载数据
        self._load_data(daily_returns_path, sentiment_path)
        
        # 初始化日度调仓器
        self.daily_tactician = DailyTactician(self.sentiment_data, use_shock_signal=True)
        
    def _load_data(self, daily_returns_path, sentiment_path):
        """加载数据"""
        # 加载日度收益率数据
        daily_returns = pd.read_csv(daily_returns_path, index_col=0, parse_dates=True)
        
        # 计算市场等权收益率（所有股票的平均）
        self.market_returns = daily_returns.mean(axis=1)
        self.market_returns = self.market_returns[self.start_date:self.end_date]
        
        print(f"市场收益率数据: {len(self.market_returns)} 天")
        print(f"期间: {self.market_returns.index.min().strftime('%Y-%m-%d')} 至 {self.market_returns.index.max().strftime('%Y-%m-%d')}")
        
        # 加载情绪数据
        sentiment = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
        self.sentiment_data = sentiment
        
        print(f"情绪指数数据: {len(self.sentiment_data)} 天")
        
    def run(self):
        """
        执行回测
        
        Returns:
        --------
        results : pd.DataFrame
            回测结果，包含每日净值和收益率
        """
        print("\n开始回测...")
        
        # 初始化结果记录
        portfolio_values = []
        daily_returns = []
        position_scalars = []
        signal_types = []
        dates = []
        
        current_value = self.initial_capital
        
        for date in self.market_returns.index:
            # 获取当日市场收益率
            market_return = self.market_returns.loc[date]
            
            # 获取情绪调仓信号
            scalar, signal_type, sentiment_score = self.daily_tactician.get_position_scalar(date)
            
            # 计算策略收益率（仓位系数 × 市场收益率）
            strategy_return = scalar * market_return
            
            # 更新组合价值
            current_value = current_value * (1 + strategy_return)
            
            # 记录结果
            dates.append(date)
            portfolio_values.append(current_value)
            daily_returns.append(strategy_return)
            position_scalars.append(scalar)
            signal_types.append(signal_type)
        
        # 构建结果DataFrame
        self.results = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'daily_return': daily_returns,
            'position_scalar': position_scalars,
            'signal_type': signal_types
        }, index=dates)
        
        print(f"回测完成: {len(self.results)} 天")
        
        return self.results
    
    def get_performance_metrics(self):
        """
        计算绩效指标
        
        Returns:
        --------
        metrics : dict
            绩效指标字典
        """
        if self.results is None:
            raise ValueError("请先执行回测")
        
        returns = self.results['daily_return'].dropna()
        
        # 总收益
        total_return = (self.results['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        
        # 年化收益
        n_days = len(returns)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        # 年化波动
        annual_vol = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        cummax = self.results['portfolio_value'].cummax()
        drawdown = (self.results['portfolio_value'] - cummax) / cummax
        max_dd = drawdown.min()
        
        # Calmar比率
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'avg_position_scalar': self.results['position_scalar'].mean()
        }
        
        return metrics
    
    def get_annual_returns(self):
        """
        计算每年的年化收益
        
        Returns:
        --------
        annual_returns : pd.Series
            每年的收益率
        """
        if self.results is None:
            raise ValueError("请先执行回测")
        
        # 按年份分组计算收益
        self.results['year'] = self.results.index.year
        
        annual_returns = {}
        for year in sorted(self.results['year'].unique()):
            year_data = self.results[self.results['year'] == year]
            
            # 计算该年的总收益
            year_return = (year_data['portfolio_value'].iloc[-1] / year_data['portfolio_value'].iloc[0]) - 1
            
            # 计算该年的交易日数
            n_days = len(year_data)
            
            # 年化收益
            annual_return = (1 + year_return) ** (252 / n_days) - 1
            
            annual_returns[year] = {
                'total_return': year_return,
                'annualized_return': annual_return,
                'trading_days': n_days
            }
        
        return annual_returns


def main():
    """主函数"""
    print("="*60)
    print("纯日度调仓策略回测（基于情绪指数）")
    print("="*60)
    
    # 创建回测引擎
    backtester = DailySentimentBacktester(
        daily_returns_path='data/raw/TRD-daily.csv',
        sentiment_path='data/processed/daily_sentiment_index.csv',
        start_date='2022-02-01',
        end_date='2026-04-10',
        initial_capital=1e8
    )
    
    # 执行回测
    results = backtester.run()
    
    # 获取绩效指标
    metrics = backtester.get_performance_metrics()
    
    print("\n" + "="*60)
    print("整体绩效指标")
    print("="*60)
    print(f"总收益率: {metrics['total_return']*100:.2f}%")
    print(f"年化收益率: {metrics['annual_return']*100:.2f}%")
    print(f"年化波动率: {metrics['annual_volatility']*100:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
    print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"Calmar比率: {metrics['calmar_ratio']:.3f}")
    print(f"平均仓位系数: {metrics['avg_position_scalar']:.4f}")
    
    # 获取每年收益
    annual_returns = backtester.get_annual_returns()
    
    print("\n" + "="*60)
    print("各年度收益")
    print("="*60)
    print(f"{'年份':<10} {'总收益':<12} {'年化收益':<12} {'交易日':<10}")
    print("-"*50)
    
    for year, data in annual_returns.items():
        print(f"{year:<10} {data['total_return']*100:>+10.2f}% {data['annualized_return']*100:>+10.2f}% {data['trading_days']:<10}")
    
    # 保存结果
    output_dir = 'results/tables'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存日度结果
    output_file = os.path.join(output_dir, 'daily_sentiment_only_returns.csv')
    results.to_csv(output_file)
    print(f"\n结果已保存: {output_file}")
    
    return backtester, metrics, annual_returns


if __name__ == '__main__':
    backtester, metrics, annual_returns = main()
