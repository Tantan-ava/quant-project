# -*- coding: utf-8 -*-
"""
EP因子构造模块

基于AKShare数据源构建EP（市盈率倒数）价值因子
支持行业中性化处理
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.data_pipeline import DataPipeline


class EPFactor:
    """
    EP因子构造器
    
    从AKShare获取PE数据，计算EP=1/PE
    支持行业中性化处理
    """
    
    def __init__(self, cache_dir='data/cache', sleep_time=0.5):
        """
        初始化EP因子构造器
        
        Parameters:
        -----------
        cache_dir : str
            缓存目录路径
        sleep_time : float
            AKShare API调用间隔
        """
        self.pipeline = DataPipeline(cache_dir=cache_dir, sleep_time=sleep_time)
    
    def get_ep_factor(self, date, industry_neutral=True):
        """
        获取EP因子
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
        industry_neutral : bool
            是否进行行业中性化处理
            
        Returns:
        --------
        pd.DataFrame
            columns: ['stock_code', 'ep', 'ep_neutral', 'industry_code']
        """
        # 从数据管道获取月度数据（EP + 行业）
        data = self.pipeline.get_monthly_data(date)
        
        if len(data) == 0:
            print(f"警告: {date} 无EP数据")
            return pd.DataFrame()
        
        # 过滤无效数据
        data = data[data['ep'].notna()]
        data = data[data['ep'] != np.inf]
        data = data[data['ep'] > 0]
        
        if len(data) == 0:
            print(f"警告: {date} 无有效EP数据")
            return pd.DataFrame()
        
        # 行业中性化处理
        if industry_neutral:
            data['ep_neutral'] = self._industry_neutralize(data)
        else:
            data['ep_neutral'] = data['ep']
        
        # 标准化输出列
        result = data[['stock_code', 'ep', 'ep_neutral', 'industry_code', 'industry_name']].copy()
        
        return result
    
    def _industry_neutralize(self, data):
        """
        行业中性化处理
        
        对每个行业的EP因子进行标准化（z-score）
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含EP和行业分类的数据
            
        Returns:
        --------
        pd.Series
            行业中性化后的EP因子
        """
        # 按行业分组标准化
        def neutralize_group(group):
            if len(group) < 3:  # 行业股票数太少，不做标准化
                return group['ep']
            
            mean = group['ep'].mean()
            std = group['ep'].std()
            
            if std == 0 or pd.isna(std):
                return group['ep'] - mean
            
            return (group['ep'] - mean) / std
        
        # 分组处理
        neutralized = data.groupby('industry_code').apply(neutralize_group)
        
        # 展平多级索引
        if isinstance(neutralized.index, pd.MultiIndex):
            neutralized = neutralized.reset_index(level=0, drop=True)
        
        return neutralized
    
    def get_ep_factor_panel(self, start_date, end_date, freq='M'):
        """
        获取多期EP因子面板数据
        
        Parameters:
        -----------
        start_date : str
            开始日期，格式'2024-01-01'
        end_date : str
            结束日期，格式'2024-12-31'
        freq : str
            频率，'M'为月度
            
        Returns:
        --------
        pd.DataFrame
            面板数据，包含多期EP因子
        """
        # 生成日期列表
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        all_data = []
        
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            print(f"获取EP因子: {date_str}")
            
            ep_data = self.get_ep_factor(date_str)
            
            if len(ep_data) > 0:
                ep_data['date'] = date_str
                all_data.append(ep_data)
        
        if len(all_data) == 0:
            return pd.DataFrame()
        
        # 合并所有数据
        panel_data = pd.concat(all_data, ignore_index=True)
        
        return panel_data
    
    def get_top_ep_stocks(self, date, n=50, industry_neutral=True):
        """
        获取EP最高的N只股票（价值股）
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
        n : int
            选取股票数量
        industry_neutral : bool
            是否使用行业中性化后的EP
            
        Returns:
        --------
        pd.DataFrame
            前N只价值股
        """
        ep_data = self.get_ep_factor(date, industry_neutral=industry_neutral)
        
        if len(ep_data) == 0:
            return pd.DataFrame()
        
        # 根据是否行业中性化选择排序列
        sort_col = 'ep_neutral' if industry_neutral else 'ep'
        
        # 排序并取前N
        top_stocks = ep_data.nlargest(n, sort_col)
        
        return top_stocks
    
    def get_ep_distribution(self, date):
        """
        获取EP因子分布统计
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
            
        Returns:
        --------
        dict
            EP分布统计信息
        """
        ep_data = self.get_ep_factor(date, industry_neutral=False)
        
        if len(ep_data) == 0:
            return {}
        
        ep_series = ep_data['ep']
        
        stats = {
            'count': len(ep_series),
            'mean': ep_series.mean(),
            'median': ep_series.median(),
            'std': ep_series.std(),
            'min': ep_series.min(),
            'max': ep_series.max(),
            'p5': ep_series.quantile(0.05),
            'p25': ep_series.quantile(0.25),
            'p75': ep_series.quantile(0.75),
            'p95': ep_series.quantile(0.95),
        }
        
        return stats
    
    def check_data_quality(self, date):
        """
        检查数据质量
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
            
        Returns:
        --------
        dict
            数据质量报告
        """
        ep_data = self.get_ep_factor(date)
        
        if len(ep_data) == 0:
            return {'status': 'error', 'message': '无数据'}
        
        # 行业覆盖
        industry_coverage = ep_data['industry_code'].notna().sum() / len(ep_data)
        
        # EP有效性
        valid_ep = ep_data['ep'].notna().sum() / len(ep_data)
        valid_ep_neutral = ep_data['ep_neutral'].notna().sum() / len(ep_data)
        
        # 异常值检查
        ep_mean = ep_data['ep'].mean()
        ep_std = ep_data['ep'].std()
        outliers = len(ep_data[abs(ep_data['ep'] - ep_mean) > 3 * ep_std])
        
        report = {
            'status': 'ok',
            'total_stocks': len(ep_data),
            'industry_coverage': industry_coverage,
            'valid_ep_ratio': valid_ep,
            'valid_ep_neutral_ratio': valid_ep_neutral,
            'outliers': outliers,
            'outlier_ratio': outliers / len(ep_data)
        }
        
        return report


# 使用示例
if __name__ == '__main__':
    # 初始化EP因子构造器
    ep_factor = EPFactor()
    
    # 测试单期EP因子获取
    print("="*60)
    print("测试EP因子获取")
    print("="*60)
    
    date = '2024-01-31'
    ep_data = ep_factor.get_ep_factor(date, industry_neutral=True)
    
    print(f"\n日期: {date}")
    print(f"股票数量: {len(ep_data)}")
    print("\n前10只价值股（EP最高）:")
    print(ep_data.nlargest(10, 'ep_neutral')[['stock_code', 'ep', 'ep_neutral', 'industry_name']])
    
    # 数据质量检查
    print("\n" + "="*60)
    print("数据质量报告")
    print("="*60)
    quality = ep_factor.check_data_quality(date)
    for key, value in quality.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if value < 1 else f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # EP分布统计
    print("\n" + "="*60)
    print("EP分布统计")
    print("="*60)
    dist = ep_factor.get_ep_distribution(date)
    for key, value in dist.items():
        print(f"  {key}: {value:.4f}")
