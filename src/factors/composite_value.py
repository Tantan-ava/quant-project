# -*- coding: utf-8 -*-
"""
复合价值因子模块

基于AKShare的PE/PB数据，计算复合价值得分：
- EP (Earnings-to-Price): 40%
- BP (Book-to-Price): 30%
- SP (Sales-to-Price): 20%
- DP (Dividend Yield): 10%

全部指标进行行业中性化处理（申万一级分组Z-score）
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.data_pipeline import DataPipeline


class CompositeValueFactor:
    """
    复合价值因子构造器
    
    权重配置（优化后）：
    - EP: 80% (盈利收益率)
    - BP: 0% (账面市值比)
    - SP: 0% (营收市值比)
    - DP: 20% (股息率)
    """
    
    # 权重配置（优化后：EP=80%, DP=20%）
    WEIGHTS = {
        'ep': 0.80,
        'bp': 0.00,
        'sp': 0.00,
        'dp': 0.20
    }
    
    def __init__(self, cache_dir='data/cache', sleep_time=0.5):
        """
        初始化复合价值因子构造器
        
        Parameters:
        -----------
        cache_dir : str
            缓存目录路径
        sleep_time : float
            AKShare API调用间隔
        """
        self.pipeline = DataPipeline(cache_dir=cache_dir, sleep_time=sleep_time)
    
    def get_composite_value(self, date, industry_neutral=True):
        """
        获取复合价值因子
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
        industry_neutral : bool
            是否进行行业中性化处理
            
        Returns:
        --------
        pd.DataFrame
            columns: ['stock_code', 'ep', 'bp', 'sp', 'dp', 
                     'ep_neutral', 'bp_neutral', 'sp_neutral', 'dp_neutral',
                     'composite_score', 'industry_code']
        """
        # 获取基础数据（包含EP和行业信息）
        base_data = self.pipeline.get_monthly_data(date)
        
        if len(base_data) == 0:
            print(f"警告: {date} 无基础数据")
            return pd.DataFrame()
        
        # 计算各价值指标
        value_data = self._calculate_value_metrics(base_data)
        
        if len(value_data) == 0:
            print(f"警告: {date} 无法计算价值指标")
            return pd.DataFrame()
        
        # 行业中性化处理
        if industry_neutral:
            value_data = self._industry_neutralize_all(value_data)
        
        # 计算复合价值得分
        value_data['composite_score'] = self._calculate_composite_score(value_data)
        
        return value_data
    
    def _calculate_value_metrics(self, data):
        """
        计算各价值指标
        
        Parameters:
        -----------
        data : pd.DataFrame
            基础数据
            
        Returns:
        --------
        pd.DataFrame
            包含EP、BP、SP、DP的数据
        """
        result = data.copy()
        
        # EP (Earnings-to-Price) = 1/PE
        if 'ep' in result.columns:
            result['ep'] = pd.to_numeric(result['ep'], errors='coerce')
        else:
            result['ep'] = np.nan
        
        # BP (Book-to-Price) = 1/PB
        # 从实时行情获取PB数据
        try:
            spot_data = self._get_spot_data()
            if spot_data is not None and 'pb' in spot_data.columns:
                result = result.merge(spot_data[['stock_code', 'pb']], on='stock_code', how='left')
                result['bp'] = 1 / pd.to_numeric(result['pb'], errors='coerce')
            else:
                result['bp'] = np.nan
        except:
            result['bp'] = np.nan
        
        # SP (Sales-to-Price) - 需要营收数据，暂用EP近似
        # 实际应用中应从财务报表获取
        result['sp'] = result['ep'] * 0.8  # 简化处理
        
        # DP (Dividend Yield) - 股息率
        try:
            if spot_data is not None and 'dividend_yield' in spot_data.columns:
                result = result.merge(spot_data[['stock_code', 'dividend_yield']], on='stock_code', how='left')
                result['dp'] = pd.to_numeric(result['dividend_yield'], errors='coerce')
            else:
                result['dp'] = np.nan
        except:
            result['dp'] = np.nan
        
        # 过滤无效数据
        result = result[result['ep'].notna()]
        result = result[result['ep'] > 0]
        result = result[result['ep'] != np.inf]
        
        return result
    
    def _get_spot_data(self):
        """获取实时行情数据（用于PB、股息率等）"""
        try:
            import akshare as ak
            import time
            spot_df = ak.stock_zh_a_spot_em()
            time.sleep(0.5)
            
            result = pd.DataFrame()
            result['stock_code'] = spot_df['代码']
            result['pb'] = pd.to_numeric(spot_df.get('市净率', spot_df.get('PB')), errors='coerce')
            result['dividend_yield'] = pd.to_numeric(spot_df.get('股息率', spot_df.get('DP')), errors='coerce')
            result['market_cap'] = pd.to_numeric(spot_df.get('总市值', spot_df.get('market_cap')), errors='coerce')
            
            return result
        except Exception as e:
            print(f"获取实时行情数据失败: {e}")
            return None
    
    def _industry_neutralize_all(self, data):
        """
        对所有价值指标进行行业中性化
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含价值指标的数据
            
        Returns:
        --------
        pd.DataFrame
            行业中性化后的数据
        """
        metrics = ['ep', 'bp', 'sp', 'dp']
        
        for metric in metrics:
            if metric in data.columns:
                data[f'{metric}_neutral'] = self._industry_neutralize(data, metric)
        
        return data
    
    def _industry_neutralize(self, data, metric):
        """
        对单一指标进行行业中性化
        
        Parameters:
        -----------
        data : pd.DataFrame
            数据
        metric : str
            指标名称
            
        Returns:
        --------
        pd.Series
            中性化后的指标
        """
        def neutralize_group(group):
            if len(group) < 3:
                return group[metric].fillna(0)
            
            mean = group[metric].mean()
            std = group[metric].std()
            
            if std == 0 or pd.isna(std):
                return group[metric] - mean
            
            return (group[metric] - mean) / std
        
        neutralized = data.groupby('industry_code').apply(neutralize_group)
        
        if isinstance(neutralized.index, pd.MultiIndex):
            neutralized = neutralized.reset_index(level=0, drop=True)
        
        return neutralized.fillna(0)
    
    def _calculate_composite_score(self, data):
        """
        计算复合价值得分
        
        Parameters:
        -----------
        data : pd.DataFrame
            包含中性化指标的数据
            
        Returns:
        --------
        pd.Series
            复合价值得分
        """
        score = pd.Series(0.0, index=data.index)
        
        for metric, weight in self.WEIGHTS.items():
            neutral_col = f'{metric}_neutral'
            if neutral_col in data.columns:
                score += data[neutral_col].fillna(0) * weight
        
        return score
    
    def get_top_value_stocks(self, date, n=50, industry_neutral=True):
        """
        获取复合价值得分最高的N只股票
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
        n : int
            选取股票数量
        industry_neutral : bool
            是否使用行业中性化后的得分
            
        Returns:
        --------
        pd.DataFrame
            前N只价值股
        """
        value_data = self.get_composite_value(date, industry_neutral=industry_neutral)
        
        if len(value_data) == 0:
            return pd.DataFrame()
        
        # 按复合价值得分排序
        top_stocks = value_data.nlargest(n, 'composite_score')
        
        return top_stocks
    
    def get_value_distribution(self, date):
        """
        获取复合价值因子分布统计
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
            
        Returns:
        --------
        dict
            分布统计信息
        """
        value_data = self.get_composite_value(date, industry_neutral=True)
        
        if len(value_data) == 0:
            return {}
        
        score = value_data['composite_score']
        
        stats = {
            'count': len(score),
            'mean': score.mean(),
            'median': score.median(),
            'std': score.std(),
            'min': score.min(),
            'max': score.max(),
            'p5': score.quantile(0.05),
            'p25': score.quantile(0.25),
            'p75': score.quantile(0.75),
            'p95': score.quantile(0.95),
        }
        
        return stats


# 使用示例
if __name__ == '__main__':
    # 初始化复合价值因子构造器
    cvf = CompositeValueFactor()
    
    # 测试单期复合价值因子获取
    print("="*60)
    print("测试复合价值因子获取")
    print("="*60)
    
    date = '2024-01-31'
    value_data = cvf.get_composite_value(date, industry_neutral=True)
    
    print(f"\n日期: {date}")
    print(f"股票数量: {len(value_data)}")
    
    if len(value_data) > 0:
        print("\n各指标权重:")
        for metric, weight in cvf.WEIGHTS.items():
            print(f"  {metric.upper()}: {weight*100:.0f}%")
        
        print("\n前10只价值股（复合得分最高）:")
        display_cols = ['stock_code', 'ep', 'bp', 'composite_score', 'industry_name']
        print(value_data.nlargest(10, 'composite_score')[display_cols])
        
        # 分布统计
        print("\n" + "="*60)
        print("复合价值得分分布统计")
        print("="*60)
        dist = cvf.get_value_distribution(date)
        for key, value in dist.items():
            print(f"  {key}: {value:.4f}")
