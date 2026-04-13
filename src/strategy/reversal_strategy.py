#!/usr/bin/env python3
"""
反转策略实现
基于过去K个月累积收益的负值构建反转信号
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, '/Users/xinyutan/Documents/量化投资/quant-project')


class ReversalStrategy:
    """
    反转策略
    
    策略逻辑:
    - 形成期: K个月
    - 信号: 过去K个月累积收益的负值
    - 选股: 选择信号最高的TopK只股票
    - 权重: 等权重
    """
    
    def __init__(self,
                 monthly_returns_path='data/raw/TRD_Mnth.xlsx',
                 start_date='2005-01-01',
                 end_date='2025-12-31',
                 formation_period=6,
                 top_k=50,
                 winsorize=True):
        """
        初始化反转策略
        
        Parameters:
        -----------
        monthly_returns_path : str
            月度收益率数据路径
        start_date : str
            回测开始日期
        end_date : str
            回测结束日期
        formation_period : int
            反转形成期(月)，默认6个月
        top_k : int
            选股数量
        winsorize : bool
            是否进行Winsorization处理
        """
        self.monthly_returns_path = monthly_returns_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.formation_period = formation_period
        self.top_k = top_k
        self.winsorize = winsorize
        
        # 加载数据
        self.monthly_returns = self._load_data(monthly_returns_path)
        self.trading_months = self.monthly_returns.index
        
        # 回测记录
        self.records = []
        self.results = None
        self.metrics = {}
        
    def _load_data(self, filepath):
        """加载月度收益率数据"""
        print(f"加载月度收益率数据: {filepath}")
        
        try:
            # 加载Excel文件
            df = pd.read_excel(filepath, engine='openpyxl')
            
            # 第一列是日期
            if 'Trdmnt' in df.columns:
                df['date'] = pd.to_datetime(df['Trdmnt'].astype(str), format='%Y-%m')
                df = df.drop(columns=['Trdmnt'])
            else:
                df['date'] = pd.to_datetime(df.iloc[:, 0].astype(str), format='%Y-%m')
                df = df.iloc[:, 1:]
            
            df = df.set_index('date')
            df = df.sort_index()
            
            # 过滤日期范围
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            
            # 转换为数值类型
            df = df.apply(pd.to_numeric, errors='coerce')
            
            print(f"  加载完成: {len(df)} 个月, {len(df.columns)} 只股票")
            print(f"  日期范围: {df.index.min()} 至 {df.index.max()}")
            
            return df
            
        except Exception as e:
            print(f"  加载失败: {e}")
            raise
    
    def calculate_reversal_signal(self, date_idx):
        """
        计算反转信号
        
        公式: Signal_t = -Return_{t-K:t-1}
        
        Parameters:
        -----------
        date_idx : int
            当前日期索引
            
        Returns:
        --------
        pd.Series
            反转信号序列
        """
        k = self.formation_period
        
        if date_idx < k:
            # 数据不足
            return pd.Series(dtype=float)
        
        # 计算K期累积收益
        period_returns = self.monthly_returns.iloc[date_idx-k:date_idx]
        cumulative_return = (1 + period_returns).prod() - 1
        
        # 反转信号 = -累积收益
        reversal_signal = -cumulative_return
        
        return reversal_signal.dropna()
    
    def _select_stocks(self, date_idx):
        """
        选股逻辑
        
        选择反转信号最高的TopK只股票
        """
        # 计算反转信号
        signal = self.calculate_reversal_signal(date_idx)
        
        if len(signal) == 0:
            return pd.Series()
        
        # Winsorization处理
        if self.winsorize:
            lower = signal.quantile(0.05)
            upper = signal.quantile(0.95)
            signal = signal.clip(lower, upper)
        
        # 选Top K
        signal = signal.dropna()
        if len(signal) == 0:
            return pd.Series()
        
        selected = signal.nlargest(self.top_k)
        
        # 等权重
        weights = pd.Series(1/len(selected), index=selected.index)
        
        return weights
    
    def run_backtest(self):
        """
        执行回测
        
        Returns:
        --------
        pd.DataFrame
            回测结果
        """
        print("\n" + "="*60)
        print("反转策略回测")
        print("="*60)
        print(f"配置: 形成期={self.formation_period}个月, TopK={self.top_k}")
        print(f"期间: {self.start_date.date()} 至 {self.end_date.date()}")
        
        portfolio_value = 1.0
        current_weights = None
        
        # 从第K个月开始
        start_idx = self.formation_period
        
        for i in range(start_idx, len(self.trading_months)):
            date = self.trading_months[i]
            
            # 每月选股
            current_weights = self._select_stocks(i)
            
            if len(current_weights) == 0:
                print(f"  警告: {date} 未选出股票")
                monthly_return = 0
            else:
                # 计算当月收益
                month_return = self.monthly_returns.iloc[i]
                
                # 对齐股票
                available_stocks = month_return.index.intersection(current_weights.index)
                if len(available_stocks) == 0:
                    monthly_return = 0
                else:
                    weights = current_weights.loc[available_stocks]
                    weights = weights / weights.sum()
                    returns = month_return.loc[available_stocks]
                    
                    # 计算组合收益
                    monthly_return = (weights * returns).sum()
            
            # 更新净值
            portfolio_value *= (1 + monthly_return)
            
            # 记录
            self.records.append({
                'date': date,
                'monthly_return': monthly_return,
                'nav': portfolio_value,
                'num_stocks': len(current_weights) if current_weights is not None else 0
            })
            
            # 打印进度
            if i % 50 == 0 or i == len(self.trading_months) - 1:
                print(f"  [{date.strftime('%Y-%m')}] NAV: {portfolio_value:.4f} ({monthly_return*100:+.2f}%) [持仓{len(current_weights) if current_weights is not None else 0}只]")
        
        # 生成结果
        results = pd.DataFrame(self.records)
        results['date'] = pd.to_datetime(results['date'])
        results['year'] = results['date'].dt.year
        
        self.results = results
        self._calculate_metrics()
        
        return results
    
    def _calculate_metrics(self):
        """计算绩效指标"""
        if len(self.results) == 0:
            return
        
        ret = self.results['monthly_return']
        
        # 年化收益
        n_months = len(ret)
        total_return = (1 + ret).prod() - 1
        ann_ret = (1 + total_return) ** (12 / n_months) - 1
        
        # 年化波动
        ann_vol = ret.std() * np.sqrt(12)
        
        # 夏普比率
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # 最大回撤
        cum_ret = self.results['nav']
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        # 卡玛比率
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        print("\n" + "="*60)
        print("回测结果摘要")
        print("="*60)
        print(f"回测期间: {self.results['date'].min().date()} 至 {self.results['date'].max().date()}")
        print(f"总收益率: {total_return*100:.2f}%")
        print(f"年化收益率: {ann_ret*100:.2f}%")
        print(f"年化波动率: {ann_vol*100:.2f}%")
        print(f"夏普比率: {sharpe:.3f}")
        print(f"最大回撤: {max_dd*100:.2f}%")
        print(f"卡玛比率: {calmar:.3f}")
        print(f"交易月数: {n_months}")
        print("="*60)
        
        self.metrics = {
            'total_return': total_return,
            'annual_return': ann_ret,
            'annual_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'n_months': n_months
        }
    
    def save_results(self, output_dir='results/tables'):
        """保存回测结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存收益序列
        output_file = f'{output_dir}/reversal_strategy_K{self.formation_period}_Top{self.top_k}.csv'
        self.results.to_csv(output_file, index=False)
        print(f"\n收益序列已保存: {output_file}")
        
        # 保存年度收益
        yearly_returns = self.results.groupby('year')['monthly_return'].apply(
            lambda x: (1 + x).prod() - 1
        )
        yearly_file = f'{output_dir}/reversal_strategy_K{self.formation_period}_Top{self.top_k}_yearly.csv'
        yearly_returns.to_csv(yearly_file, header=['annual_return'])
        print(f"年度收益已保存: {yearly_file}")


def main():
    """主函数 - 演示反转策略"""
    print("="*60)
    print("反转策略演示")
    print("="*60)
    
    # 创建策略实例
    strategy = ReversalStrategy(
        monthly_returns_path='data/raw/TRD_Mnth.xlsx',
        start_date='2006-01-01',
        end_date='2025-12-31',
        formation_period=6,  # 6个月形成期
        top_k=50,            # 选50只股票
        winsorize=True
    )
    
    # 运行回测
    results = strategy.run_backtest()
    
    # 保存结果
    strategy.save_results()
    
    print("\n" + "="*60)
    print("反转策略演示完成")
    print("="*60)
    
    return results


if __name__ == '__main__':
    results = main()
