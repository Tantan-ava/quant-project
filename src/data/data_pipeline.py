# -*- coding: utf-8 -*-
"""
数据流管道

整合AKShare数据获取和本地缓存
优先检查缓存，不存在则调用AKShare，获取后保存缓存
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data.akshare_fetcher import AKShareFetcher
from src.data.data_cache import DataCache


class DataPipeline:
    """
    数据流管道
    
    提供统一的数据获取接口，自动处理缓存逻辑
    """
    
    def __init__(self, cache_dir='data/cache', sleep_time=0.5):
        """
        初始化数据管道
        
        Parameters:
        -----------
        cache_dir : str
            缓存目录路径
        sleep_time : float
            AKShare API调用间隔（秒）
        """
        self.fetcher = AKShareFetcher(sleep_time=sleep_time)
        self.cache = DataCache(cache_dir=cache_dir)
    
    def get_ep_data(self, date, use_cache=True):
        """
        获取EP因子数据
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
        use_cache : bool
            是否使用缓存
            
        Returns:
        --------
        pd.DataFrame
            EP因子数据
        """
        date_str = date.replace('-', '')[:6]  # 转换为202401格式
        
        # 检查缓存
        if use_cache:
            cached_data = self.cache.load('ep', date_str)
            if cached_data is not None:
                # 过滤指定日期
                cached_data = cached_data[cached_data['trade_date'] == date]
                if len(cached_data) > 0:
                    return cached_data
        
        # 从AKShare获取
        print(f"从AKShare获取EP数据: {date}")
        data = self.fetcher.get_ep_from_pe(date)
        
        # 保存缓存
        if use_cache and len(data) > 0:
            self.cache.save('ep', date_str, data)
        
        return data
    
    def get_industry_data(self, use_cache=True):
        """
        获取行业分类数据
        
        Parameters:
        -----------
        use_cache : bool
            是否使用缓存
            
        Returns:
        --------
        pd.DataFrame
            行业分类数据
        """
        date_str = datetime.now().strftime('%Y%m')  # 使用当前月份
        
        # 检查缓存
        if use_cache:
            cached_data = self.cache.load('industry', date_str)
            if cached_data is not None:
                return cached_data
        
        # 从AKShare获取
        print("从AKShare获取行业分类数据...")
        data = self.fetcher.get_industry_classification()
        
        # 保存缓存
        if use_cache and len(data) > 0:
            self.cache.save('industry', date_str, data)
        
        return data
    
    def get_financial_data(self, date, use_cache=True):
        """
        获取财务指标数据
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
        use_cache : bool
            是否使用缓存
            
        Returns:
        --------
        pd.DataFrame
            财务指标数据
        """
        date_str = date.replace('-', '')[:6]
        
        # 检查缓存
        if use_cache:
            cached_data = self.cache.load('financial', date_str)
            if cached_data is not None:
                cached_data = cached_data[cached_data['report_date'] == date]
                if len(cached_data) > 0:
                    return cached_data
        
        # 从AKShare获取
        print(f"从AKShare获取财务数据: {date}")
        data = self.fetcher.get_financial_indicators(date)
        
        # 保存缓存
        if use_cache and len(data) > 0:
            self.cache.save('financial', date_str, data)
        
        return data
    
    def get_price_data(self, start_date, end_date, use_cache=True):
        """
        获取日度行情数据
        
        Parameters:
        -----------
        start_date : str
            开始日期，格式'2022-01-01'
        end_date : str
            结束日期，格式'2026-12-31'
        use_cache : bool
            是否使用缓存
            
        Returns:
        --------
        pd.DataFrame
            日度行情数据
        """
        # 转换日期格式
        start_str = start_date.replace('-', '')
        end_str = end_date.replace('-', '')
        cache_key = f"{start_str}_{end_str}"
        
        # 检查缓存
        if use_cache:
            cached_data = self.cache.load('prices', cache_key)
            if cached_data is not None:
                return cached_data
        
        # 从AKShare获取
        print(f"从AKShare获取行情数据: {start_date} 至 {end_date}")
        print("注意: 首次运行需要15-20分钟下载全市场数据...")
        data = self.fetcher.get_daily_prices(start_str, end_str)
        
        # 保存缓存
        if use_cache and len(data) > 0:
            self.cache.save('prices', cache_key, data)
        
        return data
    
    def get_monthly_data(self, date):
        """
        获取月度数据（EP + 行业）
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
            
        Returns:
        --------
        pd.DataFrame
            包含EP和行业分类的数据
        """
        # 获取EP数据
        ep_data = self.get_ep_data(date)
        
        # 获取行业数据
        industry_data = self.get_industry_data()
        
        # 合并数据
        if len(ep_data) > 0 and len(industry_data) > 0:
            merged = ep_data.merge(
                industry_data[['stock_code', 'industry_name', 'industry_code']],
                on='stock_code',
                how='left'
            )
            # 缺失的行业设为"综合"
            merged['industry_name'] = merged['industry_name'].fillna('综合')
            merged['industry_code'] = merged['industry_code'].fillna('综合')
            return merged
        
        return ep_data
    
    def check_industry_coverage(self):
        """
        检查行业分类覆盖率
        
        Returns:
        --------
        dict
            覆盖率统计
        """
        industry_data = self.get_industry_data()
        
        if len(industry_data) == 0:
            return {'coverage': 0, 'total': 0, 'covered': 0}
        
        total = len(industry_data)
        covered = len(industry_data[industry_data['industry_code'] != '综合'])
        coverage = covered / total if total > 0 else 0
        
        return {
            'coverage': coverage,
            'total': total,
            'covered': covered
        }
    
    def get_cache_status(self):
        """
        获取缓存状态
        
        Returns:
        --------
        dict
            缓存统计信息
        """
        return self.cache.get_cache_info()
    
    def clear_cache(self, data_type=None, before_date=None):
        """
        清理缓存
        
        Parameters:
        -----------
        data_type : str, optional
            数据类型
        before_date : str, optional
            清理指定日期之前的缓存
        """
        self.cache.clear(data_type=data_type, before_date=before_date)


# 使用示例
if __name__ == '__main__':
    pipeline = DataPipeline()
    
    # 检查缓存状态
    print("缓存状态:")
    status = pipeline.get_cache_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # 测试EP数据获取
    print("\n测试EP数据获取...")
    ep_data = pipeline.get_ep_data('2024-01-31')
    print(f"获取到 {len(ep_data)} 条EP数据")
    print(ep_data.head())
    
    # 检查行业覆盖率
    print("\n检查行业覆盖率...")
    coverage = pipeline.check_industry_coverage()
    print(f"行业覆盖: {coverage['coverage']*100:.1f}% ({coverage['covered']}/{coverage['total']})")
