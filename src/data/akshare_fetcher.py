# -*- coding: utf-8 -*-
"""
AKShare数据获取模块

提供从AKShare获取A股数据的接口：
- EP因子（从PE计算）
- 行业分类
- 财务指标（ROE、负债率）
- 日度行情
"""

import pandas as pd
import numpy as np
import akshare as ak
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AKShareFetcher:
    """
    AKShare数据获取器
    
    所有方法都包含频率限制（sleep 0.5秒）
    """
    
    def __init__(self, sleep_time=0.5):
        """
        初始化
        
        Parameters:
        -----------
        sleep_time : float
            每次API调用后的休眠时间（秒），默认0.5秒
        """
        self.sleep_time = sleep_time
        
    def _sleep(self):
        """频率限制"""
        time.sleep(self.sleep_time)
    
    def get_ep_from_pe(self, date):
        """
        从PE数据计算EP因子
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
            
        Returns:
        --------
        pd.DataFrame
            columns: ['stock_code', 'ep', 'pe', 'trade_date']
        """
        try:
            # 获取A股实时行情数据（包含市盈率）
            pe_df = ak.stock_zh_a_spot_em()
            self._sleep()
            
            # 处理数据
            if pe_df is None or len(pe_df) == 0:
                print(f"警告: {date} PE数据为空")
                return pd.DataFrame()
            
            # 提取PE列（市盈率-动态）
            pe_col = '市盈率-动态' if '市盈率-动态' in pe_df.columns else '市盈率'
            
            # 创建结果DataFrame
            result = pd.DataFrame()
            result['stock_code'] = pe_df['代码']
            result['pe'] = pd.to_numeric(pe_df[pe_col], errors='coerce')
            result['trade_date'] = date
            
            # 过滤无效PE值
            result = result[result['pe'] > 0]
            result = result[result['pe'] != np.inf]
            
            # 计算EP = 1/PE
            result['ep'] = 1 / result['pe']
            
            return result[['stock_code', 'ep', 'pe', 'trade_date']]
            
        except Exception as e:
            print(f"获取EP数据失败 {date}: {e}")
            return pd.DataFrame()
    
    def get_industry_classification(self):
        """
        获取行业分类（映射到申万一级行业）
        
        Returns:
        --------
        pd.DataFrame
            columns: ['stock_code', 'industry_name', 'industry_code']
        """
        try:
            # 从实时行情数据获取行业信息
            spot_df = ak.stock_zh_a_spot_em()
            self._sleep()
            
            if spot_df is None or len(spot_df) == 0:
                print("警告: 行业分类数据为空")
                return pd.DataFrame()
            
            # 标准化列名
            result = pd.DataFrame()
            result['stock_code'] = spot_df['代码']
            result['industry_name'] = spot_df['所属行业'] if '所属行业' in spot_df.columns else '综合'
            
            # 映射到申万一级行业代码（简化映射）
            sw_mapping = self._get_sw_mapping()
            result['industry_code'] = result['industry_name'].map(sw_mapping)
            
            # 缺失的行业设为"综合"
            result['industry_code'] = result['industry_code'].fillna('综合')
            result['industry_name'] = result['industry_name'].fillna('综合')
            
            return result
            
        except Exception as e:
            print(f"获取行业分类失败: {e}")
            return pd.DataFrame()
    
    def _get_sw_mapping(self):
        """
        获取申万一级行业映射字典
        
        Returns:
        --------
        dict
            行业名称到申万代码的映射
        """
        # 申万一级行业代码（31个行业）
        sw_mapping = {
            '农林牧渔': '801010',
            '基础化工': '801030',
            '钢铁': '801040',
            '有色金属': '801050',
            '电子': '801080',
            '家用电器': '801110',
            '食品饮料': '801120',
            '纺织服饰': '801130',
            '轻工制造': '801140',
            '医药生物': '801150',
            '公用事业': '801160',
            '交通运输': '801170',
            '房地产': '801180',
            '商贸零售': '801200',
            '社会服务': '801210',
            '银行': '801780',
            '非银金融': '801790',
            '综合': '801230',
            '建筑材料': '801710',
            '建筑装饰': '801720',
            '电力设备': '801730',
            '机械设备': '801890',
            '国防军工': '801740',
            '计算机': '801750',
            '传媒': '801760',
            '通信': '801770',
            '煤炭': '801950',
            '石油石化': '801960',
            '环保': '801970',
            '美容护理': '801980',
        }
        return sw_mapping
    
    def get_financial_indicators(self, date):
        """
        获取财务指标（ROE、负债率）
        
        Parameters:
        -----------
        date : str
            日期，格式'2024-01-31'
            
        Returns:
        --------
        pd.DataFrame
            columns: ['stock_code', 'roe', 'debt_ratio', 'report_date']
        """
        try:
            # 从实时行情数据获取财务指标
            spot_df = ak.stock_zh_a_spot_em()
            self._sleep()
            
            if spot_df is None or len(spot_df) == 0:
                print(f"警告: {date} 财务数据为空")
                return pd.DataFrame()
            
            # 标准化列名
            result = pd.DataFrame()
            result['stock_code'] = spot_df['代码']
            
            # 提取ROE和负债率（根据实际列名调整）
            roe_col = '净资产收益率' if '净资产收益率' in spot_df.columns else 'ROE'
            debt_col = '资产负债率' if '资产负债率' in spot_df.columns else '负债率'
            
            result['roe'] = pd.to_numeric(spot_df.get(roe_col, np.nan), errors='coerce')
            result['debt_ratio'] = pd.to_numeric(spot_df.get(debt_col, np.nan), errors='coerce')
            result['report_date'] = date
            
            return result
            
        except Exception as e:
            print(f"获取财务数据失败 {date}: {e}")
            return pd.DataFrame()
    
    def get_daily_prices(self, start_date, end_date):
        """
        获取日度行情数据
        
        Parameters:
        -----------
        start_date : str
            开始日期，格式'20220101'
        end_date : str
            结束日期，格式'20261231'
            
        Returns:
        --------
        pd.DataFrame
            columns: ['stock_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
        """
        all_data = []
        
        try:
            # 获取A股所有股票列表
            stock_list = ak.stock_zh_a_spot_em()
            self._sleep()
            
            stock_codes = stock_list['代码'].tolist()
            total = len(stock_codes)
            
            print(f"开始获取 {total} 只股票的日度行情数据...")
            
            for i, code in enumerate(stock_codes):
                try:
                    # 获取单只股票历史行情
                    df = ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"  # 前复权
                    )
                    self._sleep()
                    
                    if df is not None and len(df) > 0:
                        df['stock_code'] = code
                        all_data.append(df)
                    
                    # 每10只股票打印进度
                    if (i + 1) % 10 == 0:
                        print(f"进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
                        
                except Exception as e:
                    print(f"获取 {code} 数据失败: {e}")
                    continue
            
            if len(all_data) == 0:
                print("警告: 未获取到任何行情数据")
                return pd.DataFrame()
            
            # 合并所有数据
            result = pd.concat(all_data, ignore_index=True)
            
            # 标准化列名
            column_mapping = {
                '日期': 'trade_date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
            }
            result = result.rename(columns=column_mapping)
            
            # 转换日期格式
            result['trade_date'] = pd.to_datetime(result['trade_date'])
            
            return result[['stock_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"获取行情数据失败: {e}")
            return pd.DataFrame()
    
    def get_stock_list(self):
        """
        获取A股股票列表
        
        Returns:
        --------
        pd.DataFrame
            columns: ['stock_code', 'stock_name']
        """
        try:
            df = ak.stock_zh_a_spot_em()
            self._sleep()
            
            result = pd.DataFrame({
                'stock_code': df['代码'],
                'stock_name': df['名称']
            })
            
            return result
            
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            return pd.DataFrame()


# 使用示例
if __name__ == '__main__':
    fetcher = AKShareFetcher()
    
    # 测试EP获取
    print("测试EP获取...")
    ep_data = fetcher.get_ep_from_pe('2024-01-31')
    print(f"获取到 {len(ep_data)} 条EP数据")
    print(ep_data.head())
    
    # 测试行业分类
    print("\n测试行业分类...")
    industry_data = fetcher.get_industry_classification()
    print(f"获取到 {len(industry_data)} 条行业数据")
    print(industry_data.head())
    
    # 检查行业覆盖
    if len(industry_data) > 0:
        coverage = len(industry_data[industry_data['industry_code'] != '综合']) / len(industry_data)
        print(f"行业覆盖: {coverage*100:.1f}%")
