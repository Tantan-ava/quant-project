# -*- coding: utf-8 -*-
"""
价值投资策略运行入口

执行完整的价值投资策略回测：
1. 加载AKShare缓存的日度收益率数据
2. 初始化策略
3. 执行回测
4. 输出绩效指标
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.strategy.value_strategy import ValueStrategy


def load_price_data(data_path='data/raw/TRD-daily.csv'):
    """
    加载日度价格数据
    
    Parameters:
    -----------
    data_path : str
        价格数据路径
        
    Returns:
    --------
    pd.DataFrame
        价格数据
    """
    print(f"加载价格数据: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"警告: 数据文件不存在 {data_path}")
        print("将使用模拟数据进行测试...")
        return None
    
    try:
        # 加载数据
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # 转换为长格式
        df_long = df.stack().reset_index()
        df_long.columns = ['date', 'stock_code', 'daily_return']
        df_long['date'] = df_long['date'].dt.strftime('%Y-%m-%d')
        
        print(f"加载完成: {len(df_long)} 条记录")
        print(f"日期范围: {df_long['date'].min()} 至 {df_long['date'].max()}")
        print(f"股票数量: {df_long['stock_code'].nunique()}")
        
        return df_long
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None


def generate_mock_data(start_date='2022-01-01', end_date='2025-12-31', n_stocks=500):
    """
    生成模拟数据（用于测试）
    
    Parameters:
    -----------
    start_date : str
        开始日期
    end_date : str
        结束日期
    n_stocks : int
        股票数量
        
    Returns:
    --------
    pd.DataFrame
        模拟价格数据
    """
    print("生成模拟数据...")
    
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    data = []
    for date in dates:
        for i in range(n_stocks):
            # 生成股票代码
            if i < 300:
                code = f'SH{600000 + i:06d}'
            else:
                code = f'SZ{300000 + i - 300:06d}'
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'stock_code': code,
                'daily_return': np.random.randn() * 0.02  # 日收益率
            })
    
    df = pd.DataFrame(data)
    
    print(f"生成完成: {len(df)} 条记录")
    print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"股票数量: {df['stock_code'].nunique()}")
    
    return df


def run_backtest(start_date='2022-01-01', end_date='2025-12-31', initial_capital=1e8):
    """
    执行回测
    
    Parameters:
    -----------
    start_date : str
        开始日期
    end_date : str
        结束日期
    initial_capital : float
        初始资金
        
    Returns:
    --------
    pd.DataFrame
        回测结果
    """
    print("\n" + "="*60)
    print("价值投资策略回测")
    print("="*60)
    print(f"回测期间: {start_date} 至 {end_date}")
    print(f"初始资金: {initial_capital:,.0f}")
    
    # 加载数据
    price_data = load_price_data()
    
    if price_data is None:
        price_data = generate_mock_data(start_date, end_date)
    
    # 初始化策略
    strategy = ValueStrategy(
        initial_capital=initial_capital,
        cache_dir='data/cache',
        sleep_time=0.5
    )
    
    # 执行回测
    results = strategy.run_backtest(start_date, end_date, price_data)
    
    return results, strategy


def calculate_metrics(results, strategy):
    """
    计算并输出绩效指标
    
    Parameters:
    -----------
    results : pd.DataFrame
        回测结果
    strategy : ValueStrategy
        策略实例
    """
    print("\n" + "="*60)
    print("绩效指标")
    print("="*60)
    
    metrics = strategy.get_performance_metrics(results)
    
    if len(metrics) == 0:
        print("无绩效数据")
        return
    
    print(f"总收益率: {metrics['total_return']*100:+.2f}%")
    print(f"年化收益率: {metrics['annual_return']*100:+.2f}%")
    print(f"年化波动率: {metrics['annual_volatility']*100:.2f}%")
    print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
    print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"卡玛比率: {metrics['calmar_ratio']:.3f}")
    print(f"交易日数: {metrics['trading_days']}")
    
    # 验证指标
    print("\n" + "="*60)
    print("指标验证")
    print("="*60)
    
    checks = {
        '年化收益 > 10%': metrics['annual_return'] > 0.10,
        '夏普比率 > 0.5': metrics['sharpe_ratio'] > 0.5,
        '最大回撤 < 25%': abs(metrics['max_drawdown']) < 0.25
    }
    
    for check, passed in checks.items():
        status = "✓ 通过" if passed else "✗ 未通过"
        print(f"  {status}: {check}")
    
    return metrics


def save_results(results, output_path='results/tables/value_strategy_returns.csv'):
    """
    保存回测结果
    
    Parameters:
    -----------
    results : pd.DataFrame
        回测结果
    output_path : str
        输出路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存结果
    results.to_csv(output_path, index=False)
    print(f"\n结果已保存: {output_path}")
    
    # 显示最后10行
    print("\n最后10个交易日净值:")
    print(results.tail(10)[['date', 'nav', 'daily_return', 'stock_count']])


def main():
    """主函数"""
    # 回测参数
    start_date = '2022-01-01'
    end_date = '2025-12-31'
    initial_capital = 1e8
    
    # 执行回测
    results, strategy = run_backtest(start_date, end_date, initial_capital)
    
    if len(results) == 0:
        print("回测失败: 无结果数据")
        return
    
    # 计算绩效
    metrics = calculate_metrics(results, strategy)
    
    # 保存结果
    save_results(results)
    
    print("\n" + "="*60)
    print("回测完成")
    print("="*60)


if __name__ == '__main__':
    main()
