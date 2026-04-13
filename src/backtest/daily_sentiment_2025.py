# -*- coding: utf-8 -*-
"""
2025年纯日度调仓策略回测 + CH3归因
基于情绪指数进行仓位调整，不择股（持有市场组合）
"""

import pandas as pd
import numpy as np
import os
import sys
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.strategy.daily_tactician import DailyTactician


class DailySentimentBacktester2025:
    """2025年纯日度调仓策略回测器"""
    
    def __init__(self,
                 daily_returns_path='data/raw/TRD-daily.csv',
                 sentiment_path='data/processed/daily_sentiment_index.csv',
                 start_date='2025-01-01',
                 end_date='2025-12-31',
                 initial_capital=1e8):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        
        self._load_data(daily_returns_path, sentiment_path)
        self.daily_tactician = DailyTactician(self.sentiment_data, use_shock_signal=True)
        
    def _load_data(self, daily_returns_path, sentiment_path):
        daily_returns = pd.read_csv(daily_returns_path, index_col=0, parse_dates=True)
        self.market_returns = daily_returns.mean(axis=1)
        self.market_returns = self.market_returns[self.start_date:self.end_date]
        
        sentiment = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
        self.sentiment_data = sentiment
        
        print(f"市场收益率数据: {len(self.market_returns)} 天")
        print(f"期间: {self.market_returns.index.min().strftime('%Y-%m-%d')} 至 {self.market_returns.index.max().strftime('%Y-%m-%d')}")
        
    def run(self):
        print("\n开始回测...")
        
        portfolio_values = []
        daily_returns = []
        position_scalars = []
        signal_types = []
        dates = []
        
        current_value = self.initial_capital
        
        for date in self.market_returns.index:
            market_return = self.market_returns.loc[date]
            scalar, signal_type, sentiment_score = self.daily_tactician.get_position_scalar(date)
            strategy_return = scalar * market_return
            current_value = current_value * (1 + strategy_return)
            
            dates.append(date)
            portfolio_values.append(current_value)
            daily_returns.append(strategy_return)
            position_scalars.append(scalar)
            signal_types.append(signal_type)
        
        self.results = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'daily_return': daily_returns,
            'position_scalar': position_scalars,
            'signal_type': signal_types
        }, index=dates)
        
        print(f"回测完成: {len(self.results)} 天")
        
        return self.results
    
    def get_monthly_returns(self):
        """转换为月度收益率"""
        daily_df = pd.DataFrame({
            'date': self.results.index,
            'daily_return': self.results['daily_return'].values
        })
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df['month'] = daily_df['date'].dt.to_period('M')
        
        monthly_returns = daily_df.groupby('month').apply(
            lambda x: (1 + x['daily_return']).prod() - 1
        )
        
        month_end_dates = daily_df.groupby('month')['date'].last()
        
        output_df = pd.DataFrame({
            'date': month_end_dates.values,
            'return_0cost': monthly_returns.values,
            'return_001': monthly_returns.values,
            'return_002': monthly_returns.values
        })
        output_df['date'] = pd.to_datetime(output_df['date'])
        
        return output_df


def ch3_attribution(monthly_returns):
    """CH3因子归因分析"""
    print("\n" + "="*60)
    print("CH3因子归因分析 - 2025年纯情绪策略")
    print("="*60)
    
    # 加载CH3因子数据
    ch3_factors = pd.read_excel('data/processed/CH3_factors_monthly_202602.xlsx')
    ch3_factors['date'] = pd.to_datetime(ch3_factors['mnthdt'], format='%Y%m%d')
    ch3_factors = ch3_factors.set_index('date')
    ch3_factors = ch3_factors.rename(columns={'rf_mon': 'rf', 'mktrf': 'MKT'})
    
    # 合并数据
    merged = monthly_returns.set_index('date').join(ch3_factors, how='inner')
    print(f"合并后数据: {len(merged)} 个月")
    
    if len(merged) == 0:
        print("数据对齐失败，请检查日期格式")
        return None
    
    # CH3回归
    y = merged['return_0cost'] - merged['rf']
    X = merged[['MKT', 'SMB', 'VMG']]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit(cov_type='HC0')
    
    print("\n" + "="*60)
    print("CH3因子回归结果")
    print("="*60)
    print(model.summary())
    
    # 提取关键指标
    alpha = model.params['const']
    alpha_t = model.tvalues['const']
    beta_mkt = model.params['MKT']
    beta_smb = model.params['SMB']
    beta_vmg = model.params['VMG']
    r_squared = model.rsquared
    
    print("\n" + "="*60)
    print("关键指标摘要")
    print("="*60)
    print(f"\nAlpha (月度): {alpha*100:.4f}%")
    print(f"Alpha t统计量: {alpha_t:.3f}")
    print(f"Alpha显著性: {'显著' if abs(alpha_t) > 2 else '不显著'} (|t| > 2)")
    
    print(f"\n因子暴露:")
    print(f"  MKT Beta: {beta_mkt:.4f} (t={model.tvalues['MKT']:.3f})")
    print(f"  SMB Beta: {beta_smb:.4f} (t={model.tvalues['SMB']:.3f})")
    print(f"  VMG Beta: {beta_vmg:.4f} (t={model.tvalues['VMG']:.3f})")
    
    print(f"\n模型拟合:")
    print(f"  R²: {r_squared:.4f} ({r_squared*100:.2f}%收益被因子解释)")
    
    annual_alpha = (1 + alpha)**12 - 1
    print(f"\n年化Alpha: {annual_alpha*100:.2f}%")
    
    return {
        'alpha': alpha,
        'alpha_t': alpha_t,
        'beta_mkt': beta_mkt,
        'beta_smb': beta_smb,
        'beta_vmg': beta_vmg,
        'r_squared': r_squared,
        'model': model
    }


def main():
    print("="*60)
    print("2025年纯日度调仓策略回测 + CH3归因")
    print("="*60)
    
    # 回测
    backtester = DailySentimentBacktester2025(
        daily_returns_path='data/raw/TRD-daily.csv',
        sentiment_path='data/processed/daily_sentiment_index.csv',
        start_date='2025-01-01',
        end_date='2025-12-31',
        initial_capital=1e8
    )
    
    results = backtester.run()
    
    # 计算绩效
    total_return = (results['portfolio_value'].iloc[-1] / backtester.initial_capital) - 1
    n_days = len(results)
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    annual_vol = results['daily_return'].std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    cummax = results['portfolio_value'].cummax()
    drawdown = (results['portfolio_value'] - cummax) / cummax
    max_dd = drawdown.min()
    
    print("\n" + "="*60)
    print("2025年绩效指标")
    print("="*60)
    print(f"总收益率: {total_return*100:.2f}%")
    print(f"年化收益率: {annual_return*100:.2f}%")
    print(f"年化波动率: {annual_vol*100:.2f}%")
    print(f"夏普比率: {sharpe:.3f}")
    print(f"最大回撤: {max_dd*100:.2f}%")
    
    # 生成月度收益率
    monthly_returns = backtester.get_monthly_returns()
    
    print("\n" + "="*60)
    print("2025年月度收益率")
    print("="*60)
    for _, row in monthly_returns.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['return_0cost']*100:+.2f}%")
    
    # 保存月度收益率
    output_dir = 'results/tables'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'daily_sentiment_2025_monthly.csv')
    monthly_returns.to_csv(output_file, index=False)
    print(f"\n月度收益率已保存: {output_file}")
    
    # CH3归因
    attribution = ch3_attribution(monthly_returns)
    
    return backtester, monthly_returns, attribution


if __name__ == '__main__':
    backtester, monthly_returns, attribution = main()
