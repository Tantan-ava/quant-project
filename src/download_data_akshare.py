"""
使用 akshare 下载量化因子数据
包括：市值、盈利、ROE、账面价值等
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class DataDownloader:
    """数据下载器"""
    
    def __init__(self, save_dir: str = '../data/raw'):
        """
        初始化数据下载器
        
        Args:
            save_dir: 数据保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def get_all_stock_codes(self) -> pd.DataFrame:
        """
        获取所有 A 股股票代码
        
        Returns:
            股票代码 DataFrame
        """
        print("获取所有 A 股股票代码...")
        
        # 获取 A 股列表
        stock_info = ak.stock_info_a_code_name()
        
        # 筛选沪深 A 股
        stock_info = stock_info[
            stock_info['code'].str.startswith(('6', '0', '3'))
        ]
        
        print(f"共获取 {len(stock_info)} 只股票")
        return stock_info
    
    def download_market_data(self, stock_code: str, 
                            start_date: str = '20100101',
                            end_date: str = None) -> pd.DataFrame:
        """
        下载个股市场数据（日行情）
        
        Args:
            stock_code: 股票代码（6 位数字）
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD
            
        Returns:
            日行情数据 DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        try:
            # 获取日行情数据（前复权）
            df = ak.stock_zh_a_daily(
                symbol=stock_code,
                start_date=start_date,
                end_date=end_date,
                adjust='qfq'
            )
            
            df['股票代码'] = stock_code
            return df
            
        except Exception as e:
            print(f"下载 {stock_code} 失败：{e}")
            return pd.DataFrame()
    
    def download_financial_indicators(self, stock_code: str) -> pd.DataFrame:
        """
        下载个股财务指标（从东方财富）
        
        包括：ROE、毛利率、净利润等
        
        Args:
            stock_code: 股票代码（6 位数字，带后缀如 600519.SH）
            
        Returns:
            财务指标 DataFrame
        """
        try:
            # 添加后缀
            if stock_code.startswith('6'):
                symbol = f"{stock_code}.SH"
            else:
                symbol = f"{stock_code}.SZ"
            
            # 获取财务指标
            df = ak.stock_financial_analysis_indicator_em(
                symbol=symbol,
                indicator="按报告期"
            )
            
            df['股票代码'] = stock_code
            return df
            
        except Exception as e:
            print(f"下载 {stock_code} 财务指标失败：{e}")
            return pd.DataFrame()
    
    def download_balance_sheet(self, stock_code: str) -> pd.DataFrame:
        """
        下载个股资产负债表
        
        Args:
            stock_code: 股票代码
            
        Returns:
            资产负债表 DataFrame
        """
        try:
            if stock_code.startswith('6'):
                symbol = f"{stock_code}.SH"
            else:
                symbol = f"{stock_code}.SZ"
            
            # 获取资产负债表
            df = ak.stock_financial_report_sina(
                symbol=symbol,
                report_type="资产负债表"
            )
            
            df['股票代码'] = stock_code
            return df
            
        except Exception as e:
            print(f"下载 {stock_code} 资产负债表失败：{e}")
            return pd.DataFrame()
    
    def download_income_statement(self, stock_code: str) -> pd.DataFrame:
        """
        下载个股利润表
        
        Args:
            stock_code: 股票代码
            
        Returns:
            利润表 DataFrame
        """
        try:
            if stock_code.startswith('6'):
                symbol = f"{stock_code}.SH"
            else:
                symbol = f"{stock_code}.SZ"
            
            # 获取利润表
            df = ak.stock_financial_report_sina(
                symbol=symbol,
                report_type="利润表"
            )
            
            df['股票代码'] = stock_code
            return df
            
        except Exception as e:
            print(f"下载 {stock_code} 利润表失败：{e}")
            return pd.DataFrame()
    
    def batch_download(self, stock_codes: list, 
                      max_count: int = 100,
                      save_interval: int = 10):
        """
        批量下载股票数据
        
        Args:
            stock_codes: 股票代码列表
            max_count: 最大下载数量
            save_interval: 每下载多少只股票保存一次
        """
        print(f"\n开始批量下载，计划下载 {min(max_count, len(stock_codes))} 只股票")
        
        all_market_data = []
        all_financial_data = []
        
        for i, code in enumerate(stock_codes[:max_count]):
            print(f"\n[{i+1}/{max_count}] 下载 {code}")
            
            # 下载市场数据
            market_df = self.download_market_data(code)
            if not market_df.empty:
                all_market_data.append(market_df)
            
            # 下载财务数据
            financial_df = self.download_financial_indicators(code)
            if not financial_df.empty:
                all_financial_data.append(financial_df)
            
            # 每 save_interval 只股票保存一次
            if (i + 1) % save_interval == 0:
                print(f"已下载 {i+1} 只股票，保存临时文件...")
                
                if all_market_data:
                    pd.concat(all_market_data).to_csv(
                        self.save_dir / f'market_data_temp_{i+1}.csv',
                        index=False
                    )
                
                if all_financial_data:
                    pd.concat(all_financial_data).to_csv(
                        self.save_dir / f'financial_data_temp_{i+1}.csv',
                        index=False
                    )
        
        # 合并并保存最终文件
        print("\n合并并保存数据...")
        
        if all_market_data:
            market_all = pd.concat(all_market_data, ignore_index=True)
            market_all.to_csv(
                self.save_dir / 'market_data_all.csv',
                index=False
            )
            print(f"市场数据已保存：{len(market_all)} 条记录")
        
        if all_financial_data:
            financial_all = pd.concat(all_financial_data, ignore_index=True)
            financial_all.to_csv(
                self.save_dir / 'financial_data_all.csv',
                index=False
            )
            print(f"财务数据已保存：{len(financial_all)} 条记录")


def process_factor_data(market_file: str, 
                       financial_file: str,
                       save_dir: str = '../data/processed') -> pd.DataFrame:
    """
    处理因子数据
    
    提取：
    - 市值：总市值
    - 盈利：扣除非经常性损益后的净利润
    - ROE：加权平均净资产收益率
    - 账面价值：股东权益
    
    Args:
        market_file: 市场数据文件路径
        financial_file: 财务数据文件路径
        save_dir: 处理后数据保存目录
        
    Returns:
        因子数据 DataFrame
    """
    print("\n处理因子数据...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取数据
    market_df = pd.read_csv(market_file)
    financial_df = pd.read_csv(financial_file)
    
    # 1. 处理市场数据，提取市值相关信息
    # 注意：akshare 的日行情数据不直接包含总市值，需要通过其他方式获取
    # 这里使用收盘价 * 总股本估算（需要额外获取股本数据）
    
    # 2. 处理财务数据，提取关键指标
    factor_data = financial_df[[
        '股票代码', 
        '报告期', 
        '净资产收益率 (%)',  # ROE
        '扣非净利润 (元)',  # 扣除非经常性损益后的净利润
        '基本每股收益 (元)',  # 可用于估算
        '每股净资产 (元)'  # 每股账面价值
    ]].copy()
    
    factor_data.columns = [
        'stkcd', 
        'report_date', 
        'roe', 
        'net_profit_deducted',
        'eps',
        'bvps'
    ]
    
    # 转换日期格式
    factor_data['report_date'] = pd.to_datetime(factor_data['report_date'])
    factor_data['month'] = factor_data['report_date'].dt.to_period('M')
    
    # 计算衍生指标
    # 市净率（P/B）需要市值数据，这里先留空
    factor_data['bm'] = 1 / factor_data['bvps'] if 'bvps' in factor_data.columns else None
    
    # 保存处理后的数据
    factor_data.to_csv(
        save_dir / 'factor_data.csv',
        index=False
    )
    
    print(f"因子数据已保存：{len(factor_data)} 条记录")
    print(f"字段包括：{list(factor_data.columns)}")
    
    return factor_data


def main():
    """主函数"""
    print("=" * 50)
    print("使用 akshare 下载量化因子数据")
    print("=" * 50)
    
    downloader = DataDownloader()
    
    # 1. 获取所有股票代码
    stock_codes_df = downloader.get_all_stock_codes()
    stock_codes = stock_codes_df['code'].tolist()
    
    # 2. 批量下载（示例：前 100 只股票）
    # 实际使用时可以调整 max_count 参数
    downloader.batch_download(
        stock_codes=stock_codes,
        max_count=100,  # 先下载 100 只作为示例
        save_interval=20
    )
    
    # 3. 处理因子数据
    process_factor_data(
        market_file=downloader.save_dir / 'market_data_all.csv',
        financial_file=downloader.save_dir / 'financial_data_all.csv'
    )
    
    print("\n数据下载完成！")


if __name__ == '__main__':
    main()
