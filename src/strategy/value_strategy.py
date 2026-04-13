# -*- coding: utf-8 -*-
"""
价值投资策略主类

策略流程：
1. 月度再平衡（每月第1个交易日）
2. 复合价值因子选股
3. 质量筛选
4. 组合构建（行业分散+市值分层）
5. 日度收益计算，输出净值曲线
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.factors.composite_value import CompositeValueFactor
from src.factors.quality_screener import QualityScreener
from src.strategy.portfolio_constructor import PortfolioConstructor


class ValueStrategy:
    """
    价值投资策略
    
    策略逻辑：
    - 月度再平衡（每月第1个交易日）
    - 复合价值因子 → 质量筛选 → 组合构建
    - 日度收益计算
    """
    
    def __init__(self, 
                 initial_capital=1e8,
                 cache_dir='data/cache',
                 sleep_time=0.5):
        """
        初始化价值投资策略
        
        Parameters:
        -----------
        initial_capital : float
            初始资金
        cache_dir : str
            缓存目录
        sleep_time : float
            AKShare API调用间隔
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # 初始化各模块
        self.value_factor = CompositeValueFactor(cache_dir, sleep_time)
        self.screener = QualityScreener(cache_dir, sleep_time)
        self.constructor = PortfolioConstructor()
        
        # 持仓记录
        self.positions = {}  # 当前持仓
        self.trade_history = []  # 交易记录
        self.nav_history = []  # 净值记录
        
    def run_backtest(self, start_date, end_date, price_data):
        """
        执行回测
        
        Parameters:
        -----------
        start_date : str
            开始日期，格式'2022-01-01'
        end_date : str
            结束日期，格式'2026-04-01'
        price_data : pd.DataFrame
            日度价格数据，columns=['date', 'stock_code', 'close']
            
        Returns:
        --------
        pd.DataFrame
            回测结果
        """
        print("="*60)
        print("价值投资策略回测")
        print("="*60)
        print(f"回测期间: {start_date} 至 {end_date}")
        print(f"初始资金: {self.initial_capital:,.0f}")
        
        # 生成交易日历
        trading_days = self._generate_trading_days(start_date, end_date)
        
        # 初始化
        current_portfolio = None
        
        for i, date in enumerate(trading_days):
            date_str = date.strftime('%Y-%m-%d')
            
            # 判断是否为月初第一个交易日
            is_rebalance_day = self._is_first_trading_day_of_month(date, trading_days)
            
            if is_rebalance_day:
                print(f"\n[{date_str}] 月度再平衡")
                
                # 1. 复合价值因子选股
                value_stocks = self.value_factor.get_top_value_stocks(
                    date_str, n=100, industry_neutral=True
                )
                
                if len(value_stocks) == 0:
                    print(f"  警告: {date_str} 无价值股数据")
                    continue
                
                print(f"  价值股候选: {len(value_stocks)} 只")
                
                # 2. 质量筛选
                stock_list = value_stocks['stock_code'].tolist()
                quality_stocks = self.screener.screen(stock_list, date_str)
                
                if len(quality_stocks) == 0:
                    print(f"  警告: {date_str} 无股票通过质量筛选")
                    continue
                
                print(f"  通过质量筛选: {len(quality_stocks)} 只")
                
                # 3. 组合构建
                # 获取市值数据
                market_caps = self._get_market_caps(quality_stocks['stock_code'].tolist())
                
                # 合并价值得分
                quality_stocks = quality_stocks.merge(
                    value_stocks[['stock_code', 'composite_score', 'industry_code']],
                    on='stock_code',
                    how='left'
                )
                
                current_portfolio = self.constructor.construct_portfolio(
                    quality_stocks, market_caps
                )
                
                print(f"  最终持仓: {len(current_portfolio)} 只")
                
                # 记录交易
                self._record_rebalance(date_str, current_portfolio)
            
            # 计算日度收益
            if current_portfolio is not None and len(current_portfolio) > 0:
                daily_return = self._calculate_daily_return(
                    date_str, current_portfolio, price_data
                )
                
                # 更新资金
                self.capital = self.capital * (1 + daily_return)
                
                # 记录净值
                self.nav_history.append({
                    'date': date_str,
                    'nav': self.capital,
                    'daily_return': daily_return,
                    'stock_count': len(current_portfolio)
                })
                
                # 每30天打印进度
                if i % 30 == 0:
                    print(f"  {date_str} NAV: {self.capital:,.0f} ({daily_return*100:+.2f}%)")
        
        # 生成回测结果
        results = pd.DataFrame(self.nav_history)
        
        print("\n" + "="*60)
        print("回测完成")
        print("="*60)
        
        return results
    
    def _generate_trading_days(self, start_date, end_date):
        """生成交易日历（简化版，使用所有日期）"""
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
        return dates
    
    def _is_first_trading_day_of_month(self, date, trading_days):
        """判断是否为月初第一个交易日"""
        # 获取当月第一个交易日
        month_start = date.replace(day=1)
        month_days = [d for d in trading_days if d.year == date.year and d.month == date.month]
        
        if len(month_days) == 0:
            return False
        
        first_day = min(month_days)
        return date == first_day
    
    def _get_market_caps(self, stock_codes):
        """获取市值数据"""
        try:
            import akshare as ak
            import time
            
            spot_df = ak.stock_zh_a_spot_em()
            time.sleep(0.5)
            
            result = pd.DataFrame()
            result['stock_code'] = spot_df['代码']
            result['market_cap'] = pd.to_numeric(spot_df.get('总市值', 0), errors='coerce')
            
            # 过滤指定股票
            result = result[result['stock_code'].isin(stock_codes)]
            
            return result
        except Exception as e:
            print(f"获取市值数据失败: {e}")
            # 返回默认市值
            return pd.DataFrame({
                'stock_code': stock_codes,
                'market_cap': [100e8] * len(stock_codes)  # 默认100亿
            })
    
    def _record_rebalance(self, date, portfolio):
        """记录再平衡交易"""
        for _, row in portfolio.iterrows():
            self.trade_history.append({
                'date': date,
                'stock_code': row['stock_code'],
                'action': 'buy',
                'weight': row['weight']
            })
    
    def _calculate_daily_return(self, date, portfolio, price_data):
        """计算日度收益"""
        # 获取当日价格数据
        day_prices = price_data[price_data['date'] == date]
        
        if len(day_prices) == 0:
            return 0.0
        
        # 计算组合收益（等权）
        portfolio_codes = portfolio['stock_code'].tolist()
        portfolio_prices = day_prices[day_prices['stock_code'].isin(portfolio_codes)]
        
        if len(portfolio_prices) == 0:
            return 0.0
        
        # 简化处理：假设等权持有，收益为平均收益率
        # 实际应用中需要计算每只股票的收益率
        avg_return = portfolio_prices['daily_return'].mean() if 'daily_return' in portfolio_prices.columns else 0.0
        
        return avg_return
    
    def get_performance_metrics(self, results):
        """
        计算绩效指标
        
        Parameters:
        -----------
        results : pd.DataFrame
            回测结果
            
        Returns:
        --------
        dict
            绩效指标
        """
        if len(results) == 0:
            return {}
        
        # 计算收益率
        results['nav'] = results['nav'].astype(float)
        total_return = (results['nav'].iloc[-1] / self.initial_capital) - 1
        
        # 年化收益率
        n_days = len(results)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        # 年化波动率
        daily_returns = results['daily_return'].dropna()
        annual_vol = daily_returns.std() * np.sqrt(252)
        
        # 夏普比率（假设无风险利率3%）
        risk_free_rate = 0.03
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        cummax = results['nav'].cummax()
        drawdown = (results['nav'] - cummax) / cummax
        max_dd = drawdown.min()
        
        # 卡玛比率
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'trading_days': n_days
        }
        
        return metrics


# 使用示例
if __name__ == '__main__':
    print("="*60)
    print("价值投资策略测试")
    print("="*60)
    
    # 初始化策略
    strategy = ValueStrategy(initial_capital=1e8)
    
    # 创建模拟价格数据（实际应用中从AKShare获取）
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
    
    price_data = []
    for date in dates:
        for i in range(50):
            price_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'stock_code': f'SH{i:06d}',
                'close': 10 + np.random.randn(),
                'daily_return': np.random.randn() * 0.02
            })
    
    price_df = pd.DataFrame(price_data)
    
    # 执行回测
    results = strategy.run_backtest('2024-01-01', '2024-03-31', price_df)
    
    # 计算绩效
    metrics = strategy.get_performance_metrics(results)
    
    print("\n" + "="*60)
    print("绩效指标")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'return' in key or 'drawdown' in key:
                print(f"  {key}: {value*100:.2f}%")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
