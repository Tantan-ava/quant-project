"""
从 akshare 下载并处理量化因子数据

对应 CSMAR 数据库表：
- TRD_Mnth: 月度交易数据（市值）
- FI_T2: 财务指标（ROE、盈利）
- FS_Combas: 资产负债表（股东权益）
- FS_Comins: 利润表（净利润）
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def download_stock_list() -> pd.DataFrame:
    """
    下载 A 股股票列表
    
    Returns:
        股票列表 DataFrame
    """
    print("获取 A 股股票列表...")
    stock_list = ak.stock_info_a_code_name()
    
    # 筛选沪深 A 股
    stock_list = stock_list[
        stock_list['code'].str.startswith(('6', '0', '3'))
    ].reset_index(drop=True)
    
    print(f"共获取 {len(stock_list)} 只股票")
    return stock_list


def download_monthly_returns(stock_code: str, 
                            start_date: str = '20100101',
                            end_date: str = None) -> pd.DataFrame:
    """
    下载个股月度收益率数据（类似 TRD_Mnth 表）
    
    Args:
        stock_code: 股票代码（6 位数字）
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        月度收益率数据
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    try:
        # 获取日行情数据
        daily_df = ak.stock_zh_a_daily(
            symbol=stock_code,
            start_date=start_date,
            end_date=end_date,
            adjust='qfq'
        )
        
        if daily_df.empty:
            return pd.DataFrame()
        
        # 转换为月度数据
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df['month'] = daily_df['date'].dt.to_period('M')
        
        # 计算月度收益率
        monthly = daily_df.groupby('month').agg({
            'close': 'last',
            'open': 'first',
            'volume': 'sum'
        }).reset_index()
        
        monthly['month_end'] = monthly['month'].dt.end_time
        monthly['Mnthret'] = monthly['close'] / monthly['open'] - 1
        
        monthly['Stkcd'] = stock_code
        monthly = monthly[['Stkcd', 'month_end', 'Mnthret', 'close', 'volume']]
        monthly.columns = ['Stkcd', 'Trdmnt', 'Mnthret', 'Clsprc', 'Vol']
        
        return monthly
        
    except Exception as e:
        print(f"下载 {stock_code} 月度数据失败：{e}")
        return pd.DataFrame()


def download_market_value(stock_code: str) -> pd.DataFrame:
    """
    下载个股市值数据（类似 TRD_Mnth 表中的市值字段）
    
    Args:
        stock_code: 股票代码
        
    Returns:
        市值数据 DataFrame
    """
    try:
        # 获取实时行情
        realtime_df = ak.stock_zh_a_spot_em()
        
        stock_data = realtime_df[realtime_df['代码'] == stock_code]
        
        if stock_data.empty:
            return pd.DataFrame()
        
        result = pd.DataFrame({
            'Stkcd': [stock_code],
            'Trdmnt': [datetime.now()],
            'MarketValue': [stock_data['总市值'].values[0]],  # 总市值
            'CirculatingMarketValue': [stock_data['流通市值'].values[0]]
        })
        
        return result
        
    except Exception as e:
        print(f"下载 {stock_code} 市值失败：{e}")
        return pd.DataFrame()


def download_financial_data(stock_code: str) -> tuple:
    """
    下载个股财务数据
    
    Args:
        stock_code: 股票代码
        
    Returns:
        (FI_T2 数据，FS_Comins 数据，FS_Combas 数据)
    """
    try:
        # 添加后缀
        if stock_code.startswith('6'):
            symbol = f"{stock_code}.SH"
        else:
            symbol = f"{stock_code}.SZ"
        
        # 1. 获取财务指标（类似 FI_T2）
        print(f"  下载 {stock_code} 财务指标...")
        fi_t2 = ak.stock_financial_analysis_indicator_em(
            symbol=symbol,
            indicator="按报告期"
        )
        fi_t2['Stkcd'] = stock_code
        
        # 2. 获取利润表（类似 FS_Comins）
        print(f"  下载 {stock_code} 利润表...")
        fs_comins = ak.stock_financial_report_sina(
            symbol=symbol,
            report_type="利润表"
        )
        fs_comins['Stkcd'] = stock_code
        
        # 3. 获取资产负债表（类似 FS_Combas）
        print(f"  下载 {stock_code} 资产负债表...")
        fs_combas = ak.stock_financial_report_sina(
            symbol=symbol,
            report_type="资产负债表"
        )
        fs_combas['Stkcd'] = stock_code
        
        return fi_t2, fs_comins, fs_combas
        
    except Exception as e:
        print(f"下载 {stock_code} 财务数据失败：{e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def process_factors(fi_t2_all: pd.DataFrame,
                   fs_comins_all: pd.DataFrame,
                   fs_combas_all: pd.DataFrame) -> pd.DataFrame:
    """
    处理因子数据
    
    提取关键因子：
    - 盈利：扣除非经常性损益后的净利润
    - ROE：加权平均净资产收益率
    - 账面价值：股东权益
    
    Args:
        fi_t2_all: 财务指标数据
        fs_comins_all: 利润表数据
        fs_combas_all: 资产负债表数据
        
    Returns:
        因子数据 DataFrame
    """
    print("\n处理因子数据...")
    
    # 1. 从 FI_T2 提取 ROE 和盈利
    if not fi_t2_all.empty:
        factor_df = fi_t2[[
            'Stkcd',
            '报告期',
            '净资产收益率 (%)',  # ROE
            '扣非净利润 (元)',  # 扣除非经常性损益后的净利润
        ]].copy()
        
        factor_df.columns = ['Stkcd', 'report_date', 'ROE', 'NetProfit_Deducted']
    
    # 2. 从 FS_Comins 提取净利润（备选）
    if not fs_comins_all.empty and '净利润' in fs_comins_all.columns:
        profit_df = fs_comins_all[[
            'Stkcd',
            '报告期',
            '净利润'
        ]].copy()
        profit_df.columns = ['Stkcd', 'report_date', 'NetProfit']
        
        factor_df = factor_df.merge(profit_df, on=['Stkcd', 'report_date'], how='left')
    
    # 3. 从 FS_Combas 提取股东权益（账面价值）
    if not fs_combas_all.empty:
        # 查找股东权益相关列
        equity_cols = [col for col in fs_combas_all.columns 
                      if '股东权益' in col or '所有者权益' in col]
        
        if equity_cols:
            equity_df = fs_combas_all[[
                'Stkcd',
                '报告期'
            ] + equity_cols].copy()
            
            equity_df.columns = ['Stkcd', 'report_date'] + [f'Equity_{i}' 
                                                            for i in range(len(equity_cols))]
            
            factor_df = factor_df.merge(equity_df, on=['Stkcd', 'report_date'], how='left')
    
    # 4. 数据清洗
    factor_df['report_date'] = pd.to_datetime(factor_df['report_date'])
    factor_df['month'] = factor_df['report_date'].dt.to_period('M')
    
    # 5. 计算 B/M（账面市值比）
    # 需要合并市值数据
    factor_df['BM'] = np.nan  # 暂时留空，需要市值数据
    
    return factor_df


def batch_download_all(max_stocks: int = 500):
    """
    批量下载所有股票数据
    
    Args:
        max_stocks: 最大下载股票数量
    """
    print("=" * 50)
    print("批量下载股票数据")
    print("=" * 50)
    
    # 1. 获取股票列表
    stock_list = download_stock_list()
    stock_codes = stock_list['code'].tolist()[:max_stocks]
    
    # 2. 批量下载
    all_fi_t2 = []
    all_fs_comins = []
    all_fs_combas = []
    
    for i, code in enumerate(stock_codes):
        print(f"\n[{i+1}/{len(stock_codes)}] 下载 {code}")
        
        fi_t2, fs_comins, fs_combas = download_financial_data(code)
        
        if not fi_t2.empty:
            all_fi_t2.append(fi_t2)
        if not fs_comins.empty:
            all_fs_comins.append(fs_comins)
        if not fs_combas.empty:
            all_fs_combas.append(fs_combas)
        
        # 每 50 只股票保存一次
        if (i + 1) % 50 == 0:
            print(f"\n已下载 {i+1} 只股票，保存临时文件...")
            if all_fi_t2:
                pd.concat(all_fi_t2).to_csv('../data/raw/fi_t2_temp.csv', index=False)
    
    # 3. 合并并保存
    print("\n合并数据...")
    
    if all_fi_t2:
        fi_t2_all = pd.concat(all_fi_t2, ignore_index=True)
        fi_t2_all.to_csv('../data/raw/fi_t2.csv', index=False)
        print(f"FI_T2 数据：{len(fi_t2_all)} 条")
    
    if all_fs_comins:
        fs_comins_all = pd.concat(all_fs_comins, ignore_index=True)
        fs_comins_all.to_csv('../data/raw/fs_comins.csv', index=False)
        print(f"FS_Comins 数据：{len(fs_comins_all)} 条")
    
    if all_fs_combas:
        fs_combas_all = pd.concat(all_fs_combas, ignore_index=True)
        fs_combas_all.to_csv('../data/raw/fs_combas.csv', index=False)
        print(f"FS_Combas 数据：{len(fs_combas_all)} 条")
    
    # 4. 处理因子
    if all_fi_t2:
        factor_data = process_factors(
            pd.concat(all_fi_t2, ignore_index=True),
            pd.concat(all_fs_comins, ignore_index=True) if all_fs_comins else pd.DataFrame(),
            pd.concat(all_fs_combas, ignore_index=True) if all_fs_combas else pd.DataFrame()
        )
        
        factor_data.to_csv('../data/processed/factor_data.csv', index=False)
        print(f"\n因子数据已保存：{len(factor_data)} 条")


if __name__ == '__main__':
    # 下载数据（示例：前 100 只股票）
    batch_download_all(max_stocks=100)
