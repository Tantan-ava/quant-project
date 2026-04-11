# -*- coding: utf-8 -*-
"""
CH4因子统计特征分析
"""

import pandas as pd
import numpy as np

print('='*80)
print('CH4因子统计特征分析')
print('='*80)

# 读取CH4因子数据
df_ch4 = pd.read_excel('/Users/xinyutan/Documents/量化投资/quant-project/data/processed/CH4_factors_monthly_202602.xlsx')
df_ch4['mnthdt'] = pd.to_datetime(df_ch4['mnthdt'], format='%Y%m%d')

# 因子列表
factors = ['rf_mon', 'mktrf', 'VMG', 'SMB', 'PMO']
factor_names = {
    'rf_mon': '无风险利率',
    'mktrf': '市场超额收益',
    'VMG': '价值因子',
    'SMB': '规模因子',
    'PMO': '动量因子'
}

print(f'\n数据期间: {df_ch4["mnthdt"].min().strftime("%Y-%m")} 至 {df_ch4["mnthdt"].max().strftime("%Y-%m")}')
print(f'样本量: {len(df_ch4)} 个月')

# 统计特征
print('\n' + '='*80)
print('月度统计特征')
print('='*80)

stats_data = []
for f in factors:
    row = [
        df_ch4[f].mean() * 100,      # 均值 (%)
        df_ch4[f].std() * 100,       # 标准差 (%)
        df_ch4[f].min() * 100,       # 最小值 (%)
        df_ch4[f].max() * 100,       # 最大值 (%)
        df_ch4[f].mean() * 12 * 100, # 年化均值 (%)
        df_ch4[f].std() * np.sqrt(12) * 100,  # 年化波动 (%)
        df_ch4[f].mean() / df_ch4[f].std() * np.sqrt(12) if df_ch4[f].std() != 0 else 0,  # 年化夏普
    ]
    stats_data.append(row)

stats = pd.DataFrame(stats_data, 
                     index=[factor_names[f] for f in factors],
                     columns=['月度均值(%)', '月度标准差(%)', '最小值(%)', '最大值(%)', 
                             '年化均值(%)', '年化波动(%)', '年化夏普']).T
print(stats.round(4))

# 相关性矩阵
print('\n' + '='*80)
print('因子相关性矩阵')
print('='*80)
corr = df_ch4[factors].corr()
corr.index = [factor_names[f] for f in factors]
corr.columns = [factor_names[f] for f in factors]
print(corr.round(4))

# 分位数统计
print('\n' + '='*80)
print('分位数统计 (月度收益 %)')
print('='*80)
quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
quantile_data = []
for f in factors:
    quantile_data.append([df_ch4[f].quantile(q) * 100 for q in quantiles])

quantile_df = pd.DataFrame(quantile_data,
                          index=[factor_names[f] for f in factors],
                          columns=[f'{int(q*100)}%' for q in quantiles]).T
print(quantile_df.round(4))

# 正收益比例
print('\n' + '='*80)
print('正收益月份比例')
print('='*80)
for f in factors:
    pos_ratio = (df_ch4[f] > 0).mean() * 100
    print(f'{factor_names[f]}: {pos_ratio:.2f}%')

# 极端收益统计
print('\n' + '='*80)
print('极端收益统计')
print('='*80)
for f in factors:
    data = df_ch4[f] * 100
    extreme_pos = (data > data.quantile(0.95)).sum()
    extreme_neg = (data < data.quantile(0.05)).sum()
    skew = df_ch4[f].skew()
    kurt = df_ch4[f].kurtosis()
    print(f'{factor_names[f]}:')
    print(f'  极端正收益月份(>95%分位数): {extreme_pos}')
    print(f'  极端负收益月份(<5%分位数): {extreme_neg}')
    print(f'  偏度(Skewness): {skew:.4f}')
    print(f'  峰度(Kurtosis): {kurt:.4f}')

# 保存统计结果
print('\n' + '='*80)
print('保存统计结果')
print('='*80)

# 保存到CSV
output_path = '/Users/xinyutan/Documents/量化投资/quant-project/results/tables/ch4_factor_statistics.csv'
with open(output_path, 'w', encoding='utf-8-sig') as f:
    f.write('CH4因子统计特征分析\n')
    f.write(f'数据期间: {df_ch4["mnthdt"].min().strftime("%Y-%m")} 至 {df_ch4["mnthdt"].max().strftime("%Y-%m")}\n')
    f.write(f'样本量: {len(df_ch4)} 个月\n\n')
    
    f.write('月度统计特征\n')
    stats.round(4).to_csv(f)
    f.write('\n')
    
    f.write('因子相关性矩阵\n')
    corr.round(4).to_csv(f)
    f.write('\n')
    
    f.write('分位数统计 (月度收益 %)\n')
    quantile_df.round(4).to_csv(f)

print(f'统计结果已保存至: {output_path}')

print('\n' + '='*80)
print('CH4因子统计特征分析完成')
print('='*80)
