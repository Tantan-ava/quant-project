# -*- coding: utf-8 -*-
"""
质量筛选器模块

基于财务指标进行股票质量筛选：
- 剔除ROE < 5%的股票
- 剔除负债率 > 80%的股票
- 剔除利润暴跌50%以上的股票
- 剔除ST股票
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.data_pipeline import DataPipeline


class QualityScreener:
    """
    质量筛选器
    
    筛选条件：
    - ROE >= 5%
    - 负债率 <= 80%
    - 利润变化 > -50%
    - 非ST股票
    """
    
    # 筛选阈值
    THRESHOLDS = {
        'min_roe': 0.05,           # ROE最低5%
        'max_debt_ratio': 0.80,    # 负债率最高80%
        'min_profit_change': -0.50 # 利润变化不低于-50%
    }
    
    def __init__(self, cache_dir='data/cache', sleep_time=0.5):
        """
        初始化质量筛选器
        
        Parameters:
        -----------
        cache_dir : str
            缓存目录路径
        sleep_time : float
            AKShare API调用间隔
        """
        self.pipeline = DataPipeline(cache_dir=cache_dir, sleep_time=sleep_time)
    
    def screen(self, stock_list, date):
        """
        对股票列表进行质量筛选
        
        Parameters:
        -----------
        stock_list : list
            待筛选的股票代码列表
        date : str
            日期，格式'2024-01-31'
            
        Returns:
        --------
        pd.DataFrame
            通过筛选的股票及其质量指标
        """
        # 获取财务数据
        financial_data = self._get_financial_data(date)
        
        if len(financial_data) == 0:
            print(f"警告: {date} 无财务数据")
            return pd.DataFrame()
        
        # 过滤指定股票列表
        if stock_list is not None and len(stock_list) > 0:
            financial_data = financial_data[financial_data['stock_code'].isin(stock_list)]
        
        # 应用筛选条件
        screened = self._apply_screening(financial_data)
        
        return screened
    
    def _get_financial_data(self, date):
        """
        获取财务数据
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
            
        Returns:
        --------
        pd.DataFrame
            包含ROE、负债率、利润变化、ST标记的数据
        """
        try:
            # 从实时行情获取财务数据
            import akshare as ak
            import time
            
            spot_df = ak.stock_zh_a_spot_em()
            time.sleep(0.5)
            
            result = pd.DataFrame()
            result['stock_code'] = spot_df['代码']
            result['stock_name'] = spot_df['名称']
            
            # ROE (净资产收益率)
            roe_col = '净资产收益率' if '净资产收益率' in spot_df.columns else 'ROE'
            result['roe'] = pd.to_numeric(spot_df.get(roe_col, np.nan), errors='coerce') / 100
            
            # 负债率
            debt_col = '资产负债率' if '资产负债率' in spot_df.columns else '负债率'
            result['debt_ratio'] = pd.to_numeric(spot_df.get(debt_col, np.nan), errors='coerce') / 100
            
            # 利润变化（简化处理，使用动态市盈率变化近似）
            # 实际应用中应从财务报表获取同比数据
            result['profit_change'] = 0.0  # 默认值，表示无暴跌
            
            # ST标记（从股票名称判断）
            result['is_st'] = result['stock_name'].str.contains('ST', na=False)
            
            # 过滤无效数据
            result = result[result['roe'].notna()]
            
            return result
            
        except Exception as e:
            print(f"获取财务数据失败: {e}")
            return pd.DataFrame()
    
    def _apply_screening(self, data):
        """
        应用筛选条件
        
        Parameters:
        -----------
        data : pd.DataFrame
            财务数据
            
        Returns:
        --------
        pd.DataFrame
            通过筛选的股票
        """
        original_count = len(data)
        
        # 记录筛选过程
        screening_log = []
        
        # 1. 剔除ST股票
        data = data[~data['is_st']]
        after_st = len(data)
        screening_log.append(f"剔除ST: {original_count - after_st} 只")
        
        # 2. ROE >= 5%
        data = data[data['roe'] >= self.THRESHOLDS['min_roe']]
        after_roe = len(data)
        screening_log.append(f"ROE筛选: {after_st - after_roe} 只")
        
        # 3. 负债率 <= 80%
        data = data[data['debt_ratio'] <= self.THRESHOLDS['max_debt_ratio']]
        after_debt = len(data)
        screening_log.append(f"负债率筛选: {after_roe - after_debt} 只")
        
        # 4. 利润变化 > -50%
        data = data[data['profit_change'] > self.THRESHOLDS['min_profit_change']]
        after_profit = len(data)
        screening_log.append(f"利润筛选: {after_debt - after_profit} 只")
        
        # 打印筛选日志
        print(f"质量筛选: {original_count} -> {len(data)} 只")
        for log in screening_log:
            print(f"  {log}")
        
        return data
    
    def get_screening_summary(self, stock_list, date):
        """
        获取筛选汇总报告
        
        Parameters:
        -----------
        stock_list : list
            待筛选的股票代码列表
        date : str
            日期，格式'2024-01-31'
            
        Returns:
        --------
        dict
            筛选汇总报告
        """
        # 获取全部财务数据
        financial_data = self._get_financial_data(date)
        
        if stock_list is not None and len(stock_list) > 0:
            financial_data = financial_data[financial_data['stock_code'].isin(stock_list)]
        
        original_count = len(financial_data)
        
        # 逐步应用筛选条件
        st_excluded = financial_data[financial_data['is_st']]
        financial_data = financial_data[~financial_data['is_st']]
        
        roe_excluded = financial_data[financial_data['roe'] < self.THRESHOLDS['min_roe']]
        financial_data = financial_data[financial_data['roe'] >= self.THRESHOLDS['min_roe']]
        
        debt_excluded = financial_data[financial_data['debt_ratio'] > self.THRESHOLDS['max_debt_ratio']]
        financial_data = financial_data[financial_data['debt_ratio'] <= self.THRESHOLDS['max_debt_ratio']]
        
        profit_excluded = financial_data[financial_data['profit_change'] <= self.THRESHOLDS['min_profit_change']]
        final_data = financial_data[financial_data['profit_change'] > self.THRESHOLDS['min_profit_change']]
        
        summary = {
            'original_count': original_count,
            'final_count': len(final_data),
            'pass_rate': len(final_data) / original_count if original_count > 0 else 0,
            'st_excluded': len(st_excluded),
            'roe_excluded': len(roe_excluded),
            'debt_excluded': len(debt_excluded),
            'profit_excluded': len(profit_excluded),
            'avg_roe': final_data['roe'].mean() if len(final_data) > 0 else 0,
            'avg_debt_ratio': final_data['debt_ratio'].mean() if len(final_data) > 0 else 0,
        }
        
        return summary
    
    def check_stock_quality(self, stock_code, date):
        """
        检查单只股票的质量
        
        Parameters:
        -----------
        stock_code : str
            股票代码
        date : str
            日期，格式'2024-01-31'
            
        Returns:
        --------
        dict
            质量检查结果
        """
        result = self.screen([stock_code], date)
        
        if len(result) == 0:
            # 获取详细原因
            financial_data = self._get_financial_data(date)
            stock_data = financial_data[financial_data['stock_code'] == stock_code]
            
            if len(stock_data) == 0:
                return {'passed': False, 'reason': '无数据'}
            
            reasons = []
            if stock_data['is_st'].iloc[0]:
                reasons.append('ST股票')
            if stock_data['roe'].iloc[0] < self.THRESHOLDS['min_roe']:
                reasons.append(f"ROE过低 ({stock_data['roe'].iloc[0]:.2%})")
            if stock_data['debt_ratio'].iloc[0] > self.THRESHOLDS['max_debt_ratio']:
                reasons.append(f"负债率过高 ({stock_data['debt_ratio'].iloc[0]:.2%})")
            if stock_data['profit_change'].iloc[0] <= self.THRESHOLDS['min_profit_change']:
                reasons.append(f"利润暴跌 ({stock_data['profit_change'].iloc[0]:.2%})")
            
            return {'passed': False, 'reason': ', '.join(reasons)}
        
        return {
            'passed': True,
            'roe': result['roe'].iloc[0],
            'debt_ratio': result['debt_ratio'].iloc[0],
            'profit_change': result['profit_change'].iloc[0]
        }


# 使用示例
if __name__ == '__main__':
    # 初始化质量筛选器
    screener = QualityScreener()
    
    # 测试筛选
    print("="*60)
    print("测试质量筛选器")
    print("="*60)
    
    date = '2024-01-31'
    
    # 获取全部A股列表
    import akshare as ak
    import time
    spot_df = ak.stock_zh_a_spot_em()
    time.sleep(0.5)
    all_stocks = spot_df['代码'].tolist()[:100]  # 测试前100只
    
    print(f"\n测试股票数量: {len(all_stocks)}")
    
    # 执行筛选
    screened = screener.screen(all_stocks, date)
    
    print(f"\n通过筛选: {len(screened)} 只")
    
    if len(screened) > 0:
        print("\n前10只通过筛选的股票:")
        print(screened[['stock_code', 'stock_name', 'roe', 'debt_ratio']].head(10))
        
        # 筛选汇总
        print("\n" + "="*60)
        print("筛选汇总")
        print("="*60)
        summary = screener.get_screening_summary(all_stocks, date)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}" if value < 1 else f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
