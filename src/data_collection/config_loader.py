# -*- coding: utf-8 -*-
"""
配置加载器
加载YAML配置文件
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any


class ConfigLoader:
    """加载和管理配置文件"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置加载器
        
        Parameters
        ----------
        config_path : str, optional
            配置文件路径，默认为项目根目录下的config/policy_keywords.yaml
        """
        if config_path is None:
            # 默认路径：项目根目录/config/policy_keywords.yaml
            project_root = Path(__file__).resolve().parent.parent.parent
            config_path = project_root / 'config' / 'policy_keywords.yaml'
        
        self.config_path = Path(config_path)
        self._config = None
    
    def load(self) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Returns
        -------
        Dict[str, Any]
            配置字典
        """
        if self._config is None:
            if not self.config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        
        return self._config
    
    def get_vader_lexicon(self) -> Dict[str, float]:
        """获取VADER自定义词典"""
        config = self.load()
        return config.get('vader_lexicon', {})
    
    def get_rss_sources(self) -> List[str]:
        """获取RSS源列表"""
        config = self.load()
        return config.get('rss_sources', [])
    
    def get_filters(self) -> Dict[str, Any]:
        """获取过滤参数"""
        config = self.load()
        return config.get('filters', {})
    
    def get_sentiment_params(self) -> Dict[str, Any]:
        """获取情绪计算参数"""
        config = self.load()
        return config.get('sentiment_params', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """获取输出配置"""
        config = self.load()
        return config.get('output', {})
    
    def get_shock_thresholds(self) -> Dict[str, float]:
        """获取冲击信号阈值"""
        output_config = self.get_output_config()
        return output_config.get('shock_thresholds', {
            'severe_negative': -0.5,
            'moderate_negative': -0.3,
            'neutral': 0.0,
            'moderate_positive': 0.3,
            'severe_positive': 0.5
        })


# 全局配置实例
_config_loader = None


def get_config_loader(config_path: str = None) -> ConfigLoader:
    """
    获取全局配置加载器实例（单例模式）
    
    Parameters
    ----------
    config_path : str, optional
        配置文件路径
    
    Returns
    -------
    ConfigLoader
        配置加载器实例
    """
    global _config_loader
    if _config_loader is None or config_path is not None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader
