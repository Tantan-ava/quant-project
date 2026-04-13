# -*- coding: utf-8 -*-
"""
组合构建器模块

构建分散化的价值投资组合：
- 行业分散：单行业不超过15%
- 市值分层：大盘40%、中盘30%、小盘30%
- 最终等权持有Top 50
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class PortfolioConstructor:
    """
    组合构建器
    
    构建规则：
    1. 行业分散：单行业权重不超过15%
    2. 市值分层：大盘(>500亿)40%、中盘(100-500亿)30%、小盘(<100亿)30%
    3. 等权配置：最终持仓等权重
    4. 目标数量：Top 50只股票
    """
    
    # 配置参数
    CONFIG = {
        'max_industry_weight': 0.15,    # 单行业最大权重15%
        'large_cap_weight': 0.40,        # 大盘权重40%
        'mid_cap_weight': 0.30,          # 中盘权重30%
        'small_cap_weight': 0.30,        # 小盘权重30%
        'large_cap_threshold': 50e8,     # 大盘阈值500亿
        'mid_cap_threshold': 10e8,       # 中盘阈值100亿
        'target_stock_count': 50         # 目标持股数量
    }
    
    def __init__(self):
        """初始化组合构建器"""
        pass
    
    def construct_portfolio(self, value_stocks, market_caps):
        """
        构建投资组合
        
        Parameters:
        -----------
        value_stocks : pd.DataFrame
            价值股列表，包含['stock_code', 'composite_score', 'industry_code']
        market_caps : pd.DataFrame
            市值数据，包含['stock_code', 'market_cap']
            
        Returns:
        --------
        pd.DataFrame
            组合持仓，包含['stock_code', 'industry_code', 'market_cap', 
                          'cap_tier', 'weight']
        """
        if len(value_stocks) == 0:
            print("警告: 无价值股数据")
            return pd.DataFrame()
        
        # 合并市值数据
        portfolio = value_stocks.merge(market_caps, on='stock_code', how='left')
        
        # 填充缺失市值
        portfolio['market_cap'] = portfolio['market_cap'].fillna(0)
        
        # 市值分层
        portfolio['cap_tier'] = portfolio['market_cap'].apply(self._classify_cap_tier)
        
        # 应用行业分散约束
        portfolio = self._apply_industry_constraint(portfolio)
        
        # 应用市值分层约束
        portfolio = self._apply_cap_constraint(portfolio)
        
        # 等权配置
        portfolio['weight'] = 1.0 / len(portfolio) if len(portfolio) > 0 else 0
        
        return portfolio
    
    def _classify_cap_tier(self, market_cap):
        """
        市值分层
        
        Parameters:
        -----------
        market_cap : float
            市值（元）
            
        Returns:
        --------
        str
            'large', 'mid', 'small'
        """
        if market_cap >= self.CONFIG['large_cap_threshold']:
            return 'large'
        elif market_cap >= self.CONFIG['mid_cap_threshold']:
            return 'mid'
        else:
            return 'small'
    
    def _apply_industry_constraint(self, portfolio):
        """
        应用行业分散约束
        
        单行业不超过15%，超出的行业剔除低分股票
        
        Parameters:
        -----------
        portfolio : pd.DataFrame
            组合数据
            
        Returns:
        --------
        pd.DataFrame
            应用约束后的组合
        """
        max_per_industry = int(self.CONFIG['target_stock_count'] * 
                               self.CONFIG['max_industry_weight'])
        
        result = []
        
        for industry, group in portfolio.groupby('industry_code'):
            # 按价值得分排序
            group = group.sort_values('composite_score', ascending=False)
            
            # 限制行业股票数量
            if len(group) > max_per_industry:
                group = group.head(max_per_industry)
                print(f"  行业 {industry}: 限制为 {max_per_industry} 只")
            
            result.append(group)
        
        if len(result) == 0:
            return pd.DataFrame()
        
        return pd.concat(result, ignore_index=True)
    
    def _apply_cap_constraint(self, portfolio):
        """
        应用市值分层约束
        
        大盘40%、中盘30%、小盘30%
        
        Parameters:
        -----------
        portfolio : pd.DataFrame
            组合数据
            
        Returns:
        --------
        pd.DataFrame
            应用约束后的组合
        """
        target_count = self.CONFIG['target_stock_count']
        
        large_target = int(target_count * self.CONFIG['large_cap_weight'])
        mid_target = int(target_count * self.CONFIG['mid_cap_weight'])
        small_target = target_count - large_target - mid_target
        
        large_caps = portfolio[portfolio['cap_tier'] == 'large'].sort_values(
            'composite_score', ascending=False).head(large_target)
        mid_caps = portfolio[portfolio['cap_tier'] == 'mid'].sort_values(
            'composite_score', ascending=False).head(mid_target)
        small_caps = portfolio[portfolio['cap_tier'] == 'small'].sort_values(
            'composite_score', ascending=False).head(small_target)
        
        result = pd.concat([large_caps, mid_caps, small_caps], ignore_index=True)
        
        # 如果总数不足，从剩余股票中补充
        if len(result) < target_count:
            remaining = portfolio[~portfolio['stock_code'].isin(result['stock_code'])]
            remaining = remaining.sort_values('composite_score', ascending=False)
            supplement = remaining.head(target_count - len(result))
            result = pd.concat([result, supplement], ignore_index=True)
        
        return result
    
    def get_portfolio_summary(self, portfolio):
        """
        获取组合汇总信息
        
        Parameters:
        -----------
        portfolio : pd.DataFrame
            组合持仓
            
        Returns:
        --------
        dict
            组合汇总信息
        """
        if len(portfolio) == 0:
            return {}
        
        summary = {
            'total_stocks': len(portfolio),
            'industry_count': portfolio['industry_code'].nunique(),
            'avg_composite_score': portfolio['composite_score'].mean(),
            'industry_weights': portfolio.groupby('industry_code').size() / len(portfolio),
            'cap_distribution': portfolio.groupby('cap_tier').size() / len(portfolio),
            'total_market_cap': portfolio['market_cap'].sum()
        }
        
        return summary
    
    def check_constraints(self, portfolio):
        """
        检查组合是否满足约束条件
        
        Parameters:
        -----------
        portfolio : pd.DataFrame
            组合持仓
            
        Returns:
        --------
        dict
            约束检查结果
        """
        checks = {
            'stock_count_ok': len(portfolio) <= self.CONFIG['target_stock_count'],
            'industry_constraint_ok': True,
            'cap_constraint_ok': True
        }
        
        # 检查行业约束
        industry_weights = portfolio.groupby('industry_code').size() / len(portfolio)
        max_industry_weight = industry_weights.max() if len(industry_weights) > 0 else 0
        checks['industry_constraint_ok'] = max_industry_weight <= self.CONFIG['max_industry_weight']
        checks['max_industry_weight'] = max_industry_weight
        
        # 检查市值分层
        cap_dist = portfolio.groupby('cap_tier').size() / len(portfolio)
        checks['large_cap_ratio'] = cap_dist.get('large', 0)
        checks['mid_cap_ratio'] = cap_dist.get('mid', 0)
        checks['small_cap_ratio'] = cap_dist.get('small', 0)
        
        return checks


# 使用示例
if __name__ == '__main__':
    # 初始化组合构建器
    constructor = PortfolioConstructor()
    
    print("="*60)
    print("测试组合构建器")
    print("="*60)
    
    # 创建测试数据
    np.random.seed(42)
    n_stocks = 100
    
    test_stocks = pd.DataFrame({
        'stock_code': [f'SH{i:06d}' for i in range(n_stocks)],
        'composite_score': np.random.randn(n_stocks),
        'industry_code': np.random.choice(['801010', '801020', '801030', '801040', '801050'], n_stocks)
    })
    
    test_caps = pd.DataFrame({
        'stock_code': test_stocks['stock_code'],
        'market_cap': np.random.choice([80e8, 200e8, 600e8], n_stocks)  # 小、中、大盘
    })
    
    # 构建组合
    portfolio = constructor.construct_portfolio(test_stocks, test_caps)
    
    print(f"\n组合构建完成:")
    print(f"  持股数量: {len(portfolio)}")
    print(f"  行业数量: {portfolio['industry_code'].nunique()}")
    
    # 汇总信息
    print("\n" + "="*60)
    print("组合汇总")
    print("="*60)
    summary = constructor.get_portfolio_summary(portfolio)
    print(f"  平均价值得分: {summary['avg_composite_score']:.4f}")
    print(f"  行业分布:")
    for industry, weight in summary['industry_weights'].items():
        print(f"    {industry}: {weight*100:.1f}%")
    print(f"  市值分层:")
    for tier, ratio in summary['cap_distribution'].items():
        print(f"    {tier}: {ratio*100:.1f}%")
    
    # 约束检查
    print("\n" + "="*60)
    print("约束检查")
    print("="*60)
    checks = constructor.check_constraints(portfolio)
    for key, value in checks.items():
        if isinstance(value, bool):
            status = "✓" if value else "✗"
            print(f"  {status} {key}")
        else:
            print(f"    {key}: {value:.2%}" if value < 1 else f"    {key}: {value:.4f}")
