#!/usr/bin/env python3
"""
绘制2025年纯日度情绪策略的月度收益率柱状图
实验7.12: 2025年纯日度情绪策略回测与CH3归因
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 2025年月度收益率数据（来自实验7.12）
monthly_data = {
    '2025-01': 1.90,
    '2025-02': 8.25,
    '2025-03': 4.28,
    '2025-04': 3.14,
    '2025-05': 4.39,
    '2025-06': 7.17,
    '2025-07': 5.23,
    '2025-08': 3.98,
    '2025-09': -0.97,
    '2025-10': 1.42,
    '2025-11': -0.49,
    '2025-12': 3.45
}

# 创建DataFrame
df = pd.DataFrame(list(monthly_data.items()), columns=['月份', '收益率'])
df['月份'] = pd.to_datetime(df['月份'])
df['月份标签'] = df['月份'].dt.strftime('%m月')

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

# 颜色设置：正收益为红色，负收益为绿色（A股习惯）
colors = ['#e74c3c' if r > 0 else '#27ae60' for r in df['收益率']]

# 绘制柱状图
bars = ax.bar(df['月份标签'], df['收益率'], color=colors, edgecolor='black', linewidth=0.5)

# 在柱子上添加数值标签
for bar, value in zip(bars, df['收益率']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height > 0 else -0.5),
            f'{value:+.2f}%', ha='center', va='bottom' if height > 0 else 'top',
            fontsize=10, fontweight='bold')

# 添加零线
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# 设置标题和标签
ax.set_title('2025年纯日度情绪策略月度收益率\n(实验7.12: 年化收益52.43%, 夏普比率2.572)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('月份', fontsize=12)
ax.set_ylabel('收益率 (%)', fontsize=12)

# 设置y轴范围
ax.set_ylim(-3, 10)

# 添加网格线
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加统计信息文本框
stats_text = '统计信息:\n'
stats_text += f'总收益率: +50.15%\n'
stats_text += f'正收益月份: 10/12 (83.3%)\n'
stats_text += f'最高月收益: +8.25% (2月)\n'
stats_text += f'最低月收益: -0.97% (9月)'

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# 保存图片
output_path = '/Users/xinyutan/Documents/量化投资/quant-project/results/figures/2025_monthly_returns_bar.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"图表已保存至: {output_path}")

plt.show()
