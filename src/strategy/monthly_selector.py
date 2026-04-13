# -*- coding: utf-8 -*-
"""
月度选股模块：EP价值 + K=6反转

基于run_experiments.py中最优策略参数：
- K=6, TopK=100, Winsorization, 月度再平衡
- EP价值因子权重0.4，反转因子权重0.6

每月第1个交易日开盘前执行，确定基础股票池
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


class MonthlySelector:
    """
    月度选股：EP价值 + K=6反转
    
    策略配置：
    - 形成期 K=6（6个月累积收益）
    - 选股数量 TopK=100
    - EP价值权重 0.4
    - 反转权重 0.6
    - 月度再平衡
    """
    
    def __init__(self, ep_weight=0.4, reversal_weight=0.6, top_k=100, winsorize=True):
        """
        初始化月度选股器
        
        Parameters:
        -----------
        ep_weight : float
            EP价值因子权重，默认0.4
        reversal_weight : float
            反转因子权重，默认0.6
        top_k : int
            选股数量，默认100（与最优策略一致）
        winsorize : bool
            是否使用Winsorization标准化，默认True
        """
        self.ep_weight = ep_weight
        self.reversal_weight = reversal_weight
        self.top_k = top_k
        self.winsorize = winsorize
        self.logs = []
        
    def select(self, trade_date, ep_data, returns_data):
        """
        生成月度基础股票池
        
        Parameters:
        -----------
        trade_date : str or pd.Timestamp
            当前交易日（用于确定使用哪个月末数据）
        ep_data : pd.DataFrame
            EP价值因子数据，index=stkcd, columns=date, values=ep_zscore
        returns_data : pd.DataFrame
            月度收益率数据，index=date, columns=stkcd
            
        Returns:
        --------
        base_weights : pd.Series
            基础权重，index=stkcd, values=weight (sum=1.0)
        """
        # 获取上月末日期
        last_month_end = self._get_last_month_end(trade_date)
        
        # 检查日期是否在数据中
        available_dates = returns_data.index
        if last_month_end not in available_dates:
            # 找到最近的可用日期
            last_month_end = available_dates[available_dates <= last_month_end][-1]
        
        # 计算K=6反转信号（6个月累积收益的负值）
        reversal_signal = self._calculate_reversal_signal(returns_data, last_month_end, k=6)
        
        # 提取上月末EP因子值
        if last_month_end in ep_data.columns:
            ep_signal = ep_data[last_month_end]
        else:
            # 如果EP数据没有该日期，使用空序列
            ep_signal = pd.Series(dtype=float)
        
        # 对齐股票代码
        common_stocks = reversal_signal.index.intersection(ep_signal.index)
        
        if len(common_stocks) == 0:
            print(f"警告: {trade_date} 无共同股票，仅使用反转因子")
            # 仅使用反转因子
            composite_score = reversal_signal
            common_stocks = reversal_signal.index
        else:
            ep_signal = ep_signal.loc[common_stocks]
            reversal_signal = reversal_signal.loc[common_stocks]
            
            # 缺失值处理
            ep_signal = ep_signal.fillna(ep_signal.median())
            reversal_signal = reversal_signal.fillna(0)
            
            # 标准化
            ep_z = self._standardize(ep_signal)
            reversal_z = self._standardize(reversal_signal)
            
            # 合成得分
            composite_score = (
                self.ep_weight * ep_z +
                self.reversal_weight * reversal_z
            )
        
        # 选Top K
        selected = composite_score.nlargest(self.top_k)
        
        # 等权重分配
        base_weights = pd.Series(
            1.0 / len(selected),
            index=selected.index,
            name='base_weight'
        )
        
        # 记录选股日志
        self._log_selection(trade_date, last_month_end, selected, composite_score)
        
        return base_weights
    
    def _calculate_reversal_signal(self, returns_data, date, k=6):
        """
        计算K期反转信号
        
        Signal = -Return_{t-k:t-1} (过去k个月累积收益的负值)
        
        Parameters:
        -----------
        returns_data : pd.DataFrame
            月度收益率数据
        date : pd.Timestamp
            当前日期
        k : int
            形成期，默认6
            
        Returns:
        --------
        reversal_signal : pd.Series
            反转信号，值越大表示越应该买入（过去表现越差）
        """
        # 找到date在数据中的位置
        try:
            idx = returns_data.index.get_loc(date)
        except KeyError:
            # 如果找不到精确匹配，找最近的
            idx = returns_data.index.searchsorted(date) - 1
            if idx < 0:
                idx = 0
        
        # 确保有足够的历史数据
        start_idx = max(0, idx - k + 1)
        
        # 计算k期累积收益
        period_returns = returns_data.iloc[start_idx:idx+1]
        cumulative_return = (1 + period_returns).prod() - 1
        
        # 反转信号 = -累积收益
        reversal_signal = -cumulative_return
        
        return reversal_signal
    
    def _standardize(self, signal):
        """
        信号标准化
        
        使用Winsorization（5%-95%截尾）+ Z-score标准化
        """
        if self.winsorize:
            # Winsorization: 5%-95%截尾
            lower = signal.quantile(0.05)
            upper = signal.quantile(0.95)
            signal = signal.clip(lower, upper)
        
        # Z-score标准化
        mean = signal.mean()
        std = signal.std()
        if std > 0:
            signal_z = (signal - mean) / std
        else:
            signal_z = signal - mean
            
        return signal_z
    
    def _get_last_month_end(self, date):
        """
        获取上月最后一个交易日
        
        上月末 = 本月1日 - 1天
        """
        if isinstance(date, str):
            date = pd.Timestamp(date)
        
        # 获取本月第一天
        month_start = date.replace(day=1)
        # 上月最后一天
        last_month_end = month_start - pd.Timedelta(days=1)
        
        return last_month_end
    
    def _log_selection(self, trade_date, factor_date, selected, scores):
        """
        记录选股决策
        """
        log = {
            'trade_date': trade_date,
            'factor_date': factor_date,
            'num_selected': len(selected),
            'top10_stocks': selected.head(10).index.tolist(),
            'score_mean': scores.mean(),
            'score_std': scores.std(),
            'score_min': scores.min(),
            'score_max': scores.max()
        }
        self.logs.append(log)
    
    def save_logs(self, filepath='results/logs/monthly_selection.csv'):
        """
        保存选股日志到文件
        """
        if len(self.logs) == 0:
            print("无日志可保存")
            return
            
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df_logs = pd.DataFrame(self.logs)
        
        # 处理top10_stocks列（列表转字符串）
        df_logs['top10_stocks'] = df_logs['top10_stocks'].apply(lambda x: ','.join(map(str, x)))
        
        df_logs.to_csv(filepath, index=False)
        print(f"选股日志已保存至: {filepath}")


def load_ep_factor(filepath='data/processed/CH3_factors_monthly_202602.xlsx'):
    """
    从CH3因子文件加载EP价值因子
    
    CH3因子文件中VMG列代表EP价值因子
    """
    try:
        df = pd.read_excel(filepath, index_col=0)
        # VMG是EP价值因子
        if 'VMG' in df.columns:
            ep_series = df['VMG']
            # 转换为宽格式（如果数据是长格式）
            return ep_series
        else:
            print(f"警告: 文件 {filepath} 中未找到VMG列")
            return None
    except Exception as e:
        print(f"加载EP因子失败: {e}")
        return None


def load_returns_data(filepath='data/processed/TRD_Mnth.xlsx'):
    """
    加载月度收益率数据
    """
    try:
        df = pd.read_excel(filepath, index_col=0, engine='openpyxl')
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"加载收益率数据失败: {e}")
        return None


# 使用示例
if __name__ == '__main__':
    # 初始化选股器
    selector = MonthlySelector(
        ep_weight=0.4,
        reversal_weight=0.6,
        top_k=100,
        winsorize=True
    )
    
    # 加载数据（示例）
    # returns_data = load_returns_data()
    # ep_data = load_ep_factor()
    
    # 执行选股
    # trade_date = '2024-01-02'
    # weights = selector.select(trade_date, ep_data, returns_data)
    # print(f"选股结果: {len(weights)} 只股票")
    # print(weights.head(10))
    
    print("月度选股模块已加载")
    print("策略配置: K=6, TopK=100, EP权重0.4, 反转权重0.6, Winsorization标准化")
