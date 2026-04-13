# -*- coding: utf-8 -*-
"""
检查CH3因子文件结构
"""

import pandas as pd

# 读取CH3因子数据
ch3_factors = pd.read_excel('data/processed/CH3_factors_monthly_202602.xlsx')

print("CH3因子文件列名:")
print(ch3_factors.columns.tolist())
print("\n前5行数据:")
print(ch3_factors.head())
