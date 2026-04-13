# -*- coding: utf-8 -*-
"""
混合频率回测引擎

月度选股 + 日度调仓（情绪驱动）
无交易成本
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from strategy.monthly_selector import MonthlySelector
from strategy.daily_tactician import DailyTactician


class HybridBacktester:
    """
    混合频率回测引擎
    
    策略逻辑：
    - 月度选股：EP价值 + K=6反转，每月第1个交易日执行
    - 日度调仓：基于情绪指数调整仓位系数
    - 无交易成本
    """
    
    def __init__(self,
                 daily_returns_path='data/raw/TRD-daily.csv',
                 sentiment_path='data/processed/daily_sentiment_index.csv',
                 start_date='2022-02-01',
                 end_date='2025-12-31',
                 initial_capital=1e8,
                 ep_weight=0.4,
                 reversal_weight=0.6,
                 top_k=100,
                 osa_threshold_extreme=-2.0,
                 osa_threshold_panic=-1.0,
                 osa_threshold_greed=1.5,
                 scalar_extreme=1.5,
                 scalar_greed=0.5):
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
        ep_weight : float
            EP价值因子权重
        reversal_weight : float
            反转因子权重
        top_k : int
            选股数量
        osa_threshold_extreme : float
            极度恐慌阈值
        osa_threshold_panic : float
            恐慌阈值
        osa_threshold_greed : float
            贪婪阈值
        scalar_extreme : float
            极度恐慌时的仓位系数
        scalar_greed : float
            贪婪时的仓位系数
        """
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.ep_weight = ep_weight
        self.reversal_weight = reversal_weight
        self.top_k = top_k
        
        # OSA阈值参数
        self.osa_threshold_extreme = osa_threshold_extreme
        self.osa_threshold_panic = osa_threshold_panic
        self.osa_threshold_greed = osa_threshold_greed
        self.scalar_extreme = scalar_extreme
        self.scalar_greed = scalar_greed
        
        # 加载数据
        self._load_data(daily_returns_path, sentiment_path)
        
        # 初始化策略模块
        self.selector = MonthlySelector(
            ep_weight=ep_weight,
            reversal_weight=reversal_weight,
            top_k=top_k,
            winsorize=True
        )
        
        # 初始化日度调仓器（使用自定义阈值）
        self.tactician = self._create_tactician()
    
    def _create_tactician(self):
        """
        创建日度调仓器，使用自定义OSA阈值
        """
        tactician = DailyTactician(self.sentiment_data, use_shock_signal=True)
        
        # 更新阈值
        tactician.thresholds['extreme_panic'] = self.osa_threshold_extreme
        tactician.thresholds['panic'] = self.osa_threshold_panic
        tactician.thresholds['greed'] = self.osa_threshold_greed
        
        return tactician
        
    def _load_data(self, daily_ret_path, sentiment_path):
        """
        加载数据
        
        Parameters:
        -----------
        daily_ret_path : str
            日度收益率文件路径
        sentiment_path : str
            情绪指数文件路径
        """
        # 加载日度收益率数据
        self.daily_returns = pd.read_csv(daily_ret_path, index_col=0, parse_dates=True)
        self.daily_returns.index = pd.to_datetime(self.daily_returns.index)
        print(f"日度收益率数据: {len(self.daily_returns)} 天, "
              f"{len(self.daily_returns.columns)} 只股票")
        
        # 加载情绪指数数据
        self.sentiment_data = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
        self.sentiment_data.index = pd.to_datetime(self.sentiment_data.index)
        print(f"情绪指数数据: {len(self.sentiment_data)} 天")
        
        # 确定交易日范围
        self.trading_days = self.daily_returns.index[
            (self.daily_returns.index >= self.start_date) &
            (self.daily_returns.index <= self.end_date)
        ]
        print(f"回测交易日: {len(self.trading_days)} 天")
        
    def run(self):
        """
        执行回测
        
        Returns:
        --------
        pd.DataFrame
            回测结果，包含每日组合价值和收益
        """
        portfolio_value = self.capital
        current_weights = pd.Series(dtype=float)
        base_weights = None
        last_month = None
        
        records = []
        
        print("\n开始回测...")
        
        for i, date in enumerate(self.trading_days):
            # 月度选股：每月第1个交易日执行
            current_month = date.to_period('M')
            if current_month != last_month:
                # 获取上月末的收益率数据用于计算反转信号
                month_start = date.replace(day=1)
                last_month_end = month_start - pd.Timedelta(days=1)
                
                # 使用日度数据计算月度反转信号（过去120个交易日 ≈ 6个月）
                base_weights = self._calculate_monthly_weights(date)
                last_month = current_month
                
                if i % 50 == 0:
                    print(f"  [{date.strftime('%Y-%m-%d')}] 月度选股: {len(base_weights)} 只股票")
            
            # 日度调仓：基于情绪调整仓位
            target_weights, scalar, signal_type, sentiment_score = self.tactician.adjust_weights(
                base_weights, date
            )
            
            # 对齐股票代码（只保留有收益率数据的股票）
            available_stocks = self.daily_returns.columns.intersection(target_weights.index)
            target_weights = target_weights.loc[available_stocks]
            
            # 归一化权重
            if target_weights.sum() > 0:
                target_weights = target_weights / target_weights.sum()
            
            current_weights = target_weights
            
            # 计算当日收益
            if i > 0:
                prev_date = self.trading_days[i-1]
                # 当日收益率 = 当日收盘价 / 前一日收盘价 - 1
                daily_ret = self.daily_returns.loc[date]
            else:
                daily_ret = pd.Series(0, index=self.daily_returns.columns)
            
            # 计算组合收益
            portfolio_ret = (daily_ret.loc[current_weights.index] * current_weights).sum()
            portfolio_value *= (1 + portfolio_ret)
            
            records.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'daily_return': portfolio_ret,
                'position_scalar': scalar,
                'signal_type': signal_type,
                'sentiment_score': sentiment_score,
                'num_holding': len(current_weights)
            })
        
        self.results = pd.DataFrame(records).set_index('date')
        self._save_results()
        self._print_summary()
        
        return self.results
    
    def _calculate_monthly_weights(self, date):
        """
        计算月度选股权重
        
        使用日度数据计算过去6个月的反转信号
        
        Parameters:
        -----------
        date : pd.Timestamp
            当前日期
            
        Returns:
        --------
        pd.Series
            选股权重
        """
        # 获取过去120个交易日（约6个月）的数据
        date_idx = self.daily_returns.index.searchsorted(date)
        if date_idx < 120:
            # 数据不足，返回等权重
            all_stocks = self.daily_returns.columns
            return pd.Series(1.0/len(all_stocks), index=all_stocks)
        
        # 计算过去6个月的累积收益
        past_returns = self.daily_returns.iloc[date_idx-120:date_idx]
        cumulative_return = (1 + past_returns).prod() - 1
        
        # 反转信号 = -累积收益
        reversal_signal = -cumulative_return
        
        # 去除NaN
        reversal_signal = reversal_signal.dropna()
        
        # Winsorization标准化
        lower = reversal_signal.quantile(0.05)
        upper = reversal_signal.quantile(0.95)
        reversal_signal = reversal_signal.clip(lower, upper)
        
        # 选Top K
        selected = reversal_signal.nlargest(self.selector.top_k)
        
        # 等权重
        base_weights = pd.Series(
            1.0 / len(selected),
            index=selected.index,
            name='base_weight'
        )
        
        return base_weights
    
    def _save_results(self):
        """
        保存回测结果
        """
        # 确保目录存在
        os.makedirs('results/tables', exist_ok=True)
        
        # 保存日度结果
        self.results.to_csv('results/tables/hybrid_daily_returns.csv')
        print("\n日度回测结果已保存至: results/tables/hybrid_daily_returns.csv")
        
        # 计算月度收益
        monthly = self.results.resample('M').agg({
            'daily_return': lambda x: (1 + x).prod() - 1,
            'portfolio_value': 'last',
            'position_scalar': 'mean',
            'sentiment_score': 'mean'
        })
        monthly.to_csv('results/tables/hybrid_monthly_returns.csv')
        print("月度回测结果已保存至: results/tables/hybrid_monthly_returns.csv")
        
        self.monthly_results = monthly
    
    def _print_summary(self):
        """
        打印回测摘要
        """
        if not hasattr(self, 'monthly_results'):
            return
            
        ret = self.monthly_results['daily_return']
        
        # 年化收益
        ann_ret = (1 + ret.mean()) ** 12 - 1
        
        # 年化波动
        ann_vol = ret.std() * np.sqrt(12)
        
        # 夏普比率
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # 最大回撤
        cum_ret = (1 + ret).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        # 卡玛比率
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        print("\n" + "="*60)
        print("回测结果摘要")
        print("="*60)
        print(f"回测期间: {self.start_date} 至 {self.end_date}")
        print(f"初始资金: {self.initial_capital/1e8:.1f}亿")
        print(f"最终资金: {self.results['portfolio_value'].iloc[-1]/1e8:.2f}亿")
        print(f"总收益率: {(self.results['portfolio_value'].iloc[-1]/self.initial_capital - 1)*100:.2f}%")
        print(f"年化收益: {ann_ret*100:.2f}%")
        print(f"年化波动: {ann_vol*100:.2f}%")
        print(f"夏普比率: {sharpe:.2f}")
        print(f"最大回撤: {max_dd*100:.2f}%")
        print(f"卡玛比率: {calmar:.2f}")
        print(f"平均仓位系数: {self.results['position_scalar'].mean():.2f}")
        print("="*60)
    
    def get_performance_metrics(self):
        """
        获取绩效指标
        
        Returns:
        --------
        dict
            绩效指标字典
        """
        if self.results is None:
            return {}
        
        ret = self.monthly_results['daily_return']
        
        # 年化收益
        ann_ret = (1 + ret.mean()) ** 12 - 1
        
        # 年化波动
        ann_vol = ret.std() * np.sqrt(12)
        
        # 夏普比率
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # 最大回撤
        cum_ret = (1 + ret).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'annual_return': ann_ret,
            'annual_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_return': self.results['portfolio_value'].iloc[-1] / self.initial_capital - 1,
            'avg_position_scalar': self.results['position_scalar'].mean(),
            'avg_sentiment': self.results['sentiment_score'].mean()
        }


# 使用示例
if __name__ == '__main__':
    # 创建回测引擎
    backtester = HybridBacktester(
        daily_returns_path='data/raw/TRD-daily.csv',
        sentiment_path='data/processed/daily_sentiment_index.csv',
        start_date='2022-02-01',
        end_date='2024-12-31',
        initial_capital=1e8,
        ep_weight=0.4,
        reversal_weight=0.6,
        top_k=100
    )
    
    # 执行回测
    results = backtester.run()
    
    # 获取绩效指标
    metrics = backtester.get_performance_metrics()
    print("\n绩效指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
