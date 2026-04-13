# -*- coding: utf-8 -*-
"""
Truth Social RSS抓取器
从Truth Social RSS源抓取帖子数据
"""

import feedparser
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Optional
import re
import time
import warnings
warnings.filterwarnings('ignore')

from .config_loader import get_config_loader


class TruthSocialFetcher:
    """
    Truth Social RSS数据抓取器
    
    支持从多个RSS源抓取帖子，并进行关键词过滤
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化抓取器
        
        Parameters
        ----------
        config_path : str, optional
            配置文件路径
        """
        self.config_loader = get_config_loader(config_path)
        self.rss_sources = self.config_loader.get_rss_sources()
        self.filters = self.config_loader.get_filters()
        self.keywords_required = set(self.filters.get('keywords_required', []))
        self.min_favourites = self.filters.get('min_favourites', 10)
        
    def fetch_rss_feed(self, rss_url: str, max_retries: int = 3) -> List[Dict]:
        """
        抓取单个RSS源
        
        Parameters
        ----------
        rss_url : str
            RSS源URL
        max_retries : int
            最大重试次数
        
        Returns
        -------
        List[Dict]
            帖子列表
        """
        posts = []
        
        for attempt in range(max_retries):
            try:
                feed = feedparser.parse(rss_url)
                
                if feed.bozo and hasattr(feed, 'bozo_exception'):
                    print(f"警告: RSS解析问题 ({rss_url}): {feed.bozo_exception}")
                
                for entry in feed.entries:
                    post = self._parse_entry(entry)
                    if post:
                        posts.append(post)
                
                break  # 成功获取，跳出重试循环
                
            except Exception as e:
                print(f"抓取失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    print(f"无法抓取RSS源: {rss_url}")
        
        return posts
    
    def _parse_entry(self, entry) -> Optional[Dict]:
        """
        解析RSS条目
        
        Parameters
        ----------
        entry : feedparser.FeedParserDict
            RSS条目
        
        Returns
        -------
        Optional[Dict]
            解析后的帖子数据，如果解析失败返回None
        """
        try:
            # 提取内容
            content = entry.get('content', [{}])[0].get('value', '') if 'content' in entry else entry.get('summary', '')
            content = self._clean_html(content)
            
            # 提取发布时间
            published = entry.get('published', '')
            created_at = self._parse_date(published)
            
            # 提取点赞数 (从内容或扩展字段)
            favourites_count = self._extract_favourites(entry)
            
            # 提取作者
            author = entry.get('author', 'Unknown')
            
            # 提取链接
            link = entry.get('link', '')
            
            return {
                'content': content,
                'created_at': created_at,
                'favourites_count': favourites_count,
                'author': author,
                'link': link,
                'source': 'truth_social'
            }
            
        except Exception as e:
            print(f"解析条目失败: {e}")
            return None
    
    def _clean_html(self, text: str) -> str:
        """清理HTML标签"""
        if not text:
            return ''
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 解码HTML实体
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'")
        return text.strip()
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """解析日期字符串"""
        if not date_str:
            return None
        
        try:
            # 尝试多种日期格式
            formats = [
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    # 转换为UTC
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc)
                except ValueError:
                    continue
            
            # 使用feedparser的parsed日期
            return None
            
        except Exception:
            return None
    
    def _extract_favourites(self, entry) -> int:
        """提取点赞数"""
        # 尝试从扩展字段获取
        if 'truth_social_favourites' in entry:
            try:
                return int(entry['truth_social_favourites'])
            except:
                pass
        
        # 尝试从内容中提取
        content = entry.get('content', [{}])[0].get('value', '') if 'content' in entry else entry.get('summary', '')
        match = re.search(r'(\d+)\s*(?:favourites?|likes?)', content, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return 0
    
    def fetch_all_posts(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        抓取所有RSS源的帖子
        
        Parameters
        ----------
        start_date : datetime, optional
            开始日期
        end_date : datetime, optional
            结束日期
        
        Returns
        -------
        pd.DataFrame
            帖子数据框
        """
        all_posts = []
        
        for rss_url in self.rss_sources:
            print(f"正在抓取: {rss_url}")
            posts = self.fetch_rss_feed(rss_url)
            all_posts.extend(posts)
            time.sleep(1)  # 礼貌性延迟
        
        if not all_posts:
            print("警告: 未获取到任何帖子")
            return pd.DataFrame(columns=['content', 'created_at', 'favourites_count', 'author', 'link', 'source'])
        
        df = pd.DataFrame(all_posts)
        
        # 日期过滤
        if start_date:
            df = df[df['created_at'] >= start_date]
        if end_date:
            df = df[df['created_at'] <= end_date]
        
        # 点赞数过滤
        df = df[df['favourites_count'] >= self.min_favourites]
        
        # 关键词过滤
        df = self._filter_by_keywords(df)
        
        # 去重
        df = df.drop_duplicates(subset=['content', 'created_at'])
        
        # 排序
        df = df.sort_values('created_at').reset_index(drop=True)
        
        print(f"共获取 {len(df)} 条帖子")
        return df
    
    def _filter_by_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据关键词过滤帖子
        
        Parameters
        ----------
        df : pd.DataFrame
            帖子数据框
        
        Returns
        -------
        pd.DataFrame
            过滤后的数据框
        """
        if not self.keywords_required or df.empty:
            return df
        
        def contains_keyword(text):
            if pd.isna(text):
                return False
            text_lower = str(text).lower()
            return any(kw.lower() in text_lower for kw in self.keywords_required)
        
        mask = df['content'].apply(contains_keyword)
        return df[mask].copy()
    
    def get_mock_data(self, start_date: str = '2022-02-01', end_date: str = None, n_samples: int = 100) -> pd.DataFrame:
        """
        生成模拟数据（用于测试和历史回溯）
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str, optional
            结束日期，默认为今天
        n_samples : int
            样本数量
        
        Returns
        -------
        pd.DataFrame
            模拟帖子数据
        """
        import numpy as np
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 生成模拟帖子
        mock_posts = []
        keywords_positive = ['deal', 'agreement', 'win', 'winning', 'chicken out', 'postpone']
        keywords_negative = ['tariff', 'sanctions', 'trade war', 'escalate', 'retaliate', 'disaster']
        
        for i in range(n_samples):
            date = np.random.choice(date_range)
            
            # 随机选择情绪倾向
            sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=[0.3, 0.4, 0.3])
            
            if sentiment_type == 'positive':
                keyword = np.random.choice(keywords_positive)
                content = f"Great progress on {keyword}! America is winning big!"
                favourites = np.random.randint(50, 500)
            elif sentiment_type == 'negative':
                keyword = np.random.choice(keywords_negative)
                content = f"We will impose {keyword} on China. They must pay!"
                favourites = np.random.randint(100, 1000)
            else:
                content = "Making America Great Again!"
                favourites = np.random.randint(20, 200)
            
            mock_posts.append({
                'content': content,
                'created_at': date,
                'favourites_count': favourites,
                'author': 'realDonaldTrump',
                'link': f'https://truthsocial.com/@realDonaldTrump/posts/{i}',
                'source': 'truth_social_mock'
            })
        
        df = pd.DataFrame(mock_posts)
        df = df.sort_values('created_at').reset_index(drop=True)
        
        print(f"生成 {len(df)} 条模拟帖子")
        return df
