# -*- coding: utf-8 -*-
"""
数据缓存模块

使用parquet格式本地缓存，避免重复调用AKShare
缓存命名规范：
- ep_202401.parquet: EP因子数据
- industry_202401.parquet: 行业分类数据
- financial_202401.parquet: 财务指标数据
- prices_202201_202612.parquet: 日度行情数据
"""

import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataCache:
    """
    数据缓存管理器
    
    自动管理缓存文件的读写
    """
    
    def __init__(self, cache_dir='data/cache'):
        """
        初始化缓存管理器
        
        Parameters:
        -----------
        cache_dir : str
            缓存目录路径
        """
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"创建缓存目录: {self.cache_dir}")
    
    def _get_cache_path(self, data_type, date_str):
        """
        获取缓存文件路径
        
        Parameters:
        -----------
        data_type : str
            数据类型 ('ep', 'industry', 'financial', 'prices')
        date_str : str
            日期字符串
            
        Returns:
        --------
        str
            缓存文件完整路径
        """
        filename = f"{data_type}_{date_str}.parquet"
        return os.path.join(self.cache_dir, filename)
    
    def exists(self, data_type, date_str):
        """
        检查缓存是否存在
        
        Parameters:
        -----------
        data_type : str
            数据类型
        date_str : str
            日期字符串
            
        Returns:
        --------
        bool
            缓存是否存在
        """
        cache_path = self._get_cache_path(data_type, date_str)
        return os.path.exists(cache_path)
    
    def load(self, data_type, date_str):
        """
        从缓存加载数据
        
        Parameters:
        -----------
        data_type : str
            数据类型
        date_str : str
            日期字符串
            
        Returns:
        --------
        pd.DataFrame or None
            缓存的数据，不存在则返回None
        """
        cache_path = self._get_cache_path(data_type, date_str)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            print(f"从缓存加载: {cache_path}")
            return df
        except Exception as e:
            print(f"加载缓存失败 {cache_path}: {e}")
            return None
    
    def save(self, data_type, date_str, df):
        """
        保存数据到缓存
        
        Parameters:
        -----------
        data_type : str
            数据类型
        date_str : str
            日期字符串
        df : pd.DataFrame
            要缓存的数据
            
        Returns:
        --------
        bool
            是否保存成功
        """
        if df is None or len(df) == 0:
            print(f"警告: 数据为空，跳过缓存 {data_type}_{date_str}")
            return False
        
        cache_path = self._get_cache_path(data_type, date_str)
        
        try:
            df.to_parquet(cache_path, index=False)
            print(f"保存到缓存: {cache_path}")
            return True
        except Exception as e:
            print(f"保存缓存失败 {cache_path}: {e}")
            return False
    
    def clear(self, data_type=None, before_date=None):
        """
        清理缓存
        
        Parameters:
        -----------
        data_type : str, optional
            指定数据类型，None则清理所有
        before_date : str, optional
            清理指定日期之前的缓存
        """
        if not os.path.exists(self.cache_dir):
            return
        
        files = os.listdir(self.cache_dir)
        
        for file in files:
            if not file.endswith('.parquet'):
                continue
            
            # 检查数据类型
            if data_type and not file.startswith(f"{data_type}_"):
                continue
            
            # 检查日期
            if before_date:
                # 从文件名提取日期
                try:
                    file_date = file.split('_')[1].split('.')[0]
                    if file_date >= before_date:
                        continue
                except:
                    pass
            
            # 删除文件
            file_path = os.path.join(self.cache_dir, file)
            try:
                os.remove(file_path)
                print(f"删除缓存: {file}")
            except Exception as e:
                print(f"删除缓存失败 {file}: {e}")
    
    def list_cache(self):
        """
        列出所有缓存文件
        
        Returns:
        --------
        list
            缓存文件列表
        """
        if not os.path.exists(self.cache_dir):
            return []
        
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.parquet')]
        return sorted(files)
    
    def get_cache_info(self):
        """
        获取缓存统计信息
        
        Returns:
        --------
        dict
            缓存统计信息
        """
        files = self.list_cache()
        
        info = {
            'total_files': len(files),
            'ep_files': len([f for f in files if f.startswith('ep_')]),
            'industry_files': len([f for f in files if f.startswith('industry_')]),
            'financial_files': len([f for f in files if f.startswith('financial_')]),
            'prices_files': len([f for f in files if f.startswith('prices_')]),
        }
        
        # 计算总大小
        total_size = 0
        for file in files:
            file_path = os.path.join(self.cache_dir, file)
            total_size += os.path.getsize(file_path)
        
        info['total_size_mb'] = total_size / (1024 * 1024)
        
        return info


# 使用示例
if __name__ == '__main__':
    cache = DataCache()
    
    # 检查缓存状态
    print("缓存统计:")
    info = cache.get_cache_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 列出所有缓存
    print("\n缓存文件列表:")
    files = cache.list_cache()
    for file in files[:10]:  # 只显示前10个
        print(f"  {file}")
    if len(files) > 10:
        print(f"  ... 还有 {len(files)-10} 个文件")
