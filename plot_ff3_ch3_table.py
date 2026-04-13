#!/usr/bin/env python3
"""
绘制FF3 vs CH3因子归因对比表格图
数据来源: experiments/exp_log.md L2820-2826
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
metrics = ['Alpha\n(%/月)', 'MKT Beta', 'SMB Beta', '价值因子\nBeta', 'R² (%)']
ff3_values = [0.62, 1.021, 0.779, -0.645, 87.21]
ch3_values = [1.08, 1.037, 0.845, -0.447, 85.70]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# 表格数据
table_data = [
    ['指标', '标准FF3 (HML-BP)', 'CH3 (VMG-EP)', '差异'],
    ['Alpha', '0.62%/月 (t=2.45)', '1.08%/月 (t=3.85)', 'CH3更高'],
    ['MKT Beta', '1.021', '1.037', '基本一致'],
    ['SMB Beta', '0.779', '0.845', 'CH3略高'],
    ['价值因子', '-0.645 (HML)', '-0.447 (VMG)', 'FF3更负'],
    ['R²', '87.21%', '85.70%', 'FF3略高']
]

# 创建表格
table = ax.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    loc='center',
    cellLoc='center',
    colColours=['#4472C4'] * 4,
    colWidths=[0.2, 0.3, 0.3, 0.2]
)

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# 设置表头样式
for i in range(4):
    table[(0, i)].set_text_props(color='white', fontweight='bold')
    table[(0, i)].set_facecolor('#4472C4')

# 设置行颜色
colors = ['#E7E6E6', '#FFFFFF']
for i in range(1, 6):
    for j in range(4):
        table[(i, j)].set_facecolor(colors[i % 2])

# 高亮关键差异
# Alpha行 - CH3更高
for j in [1, 2]:
    table[(1, j)].set_text_props(fontweight='bold')
table[(1, 2)].set_facecolor('#C6E0B4')  # 浅绿色

# 价值因子行 - 显示负值
for j in [1, 2]:
    table[(4, j)].set_text_props(color='#C00000')  # 红色表示负值

# 设置标题
plt.title('FF3 vs CH3 因子归因对比分析\n(全市场月度组合策略)', 
          fontsize=14, fontweight='bold', pad=20)

# 添加注释
note_text = '''
关键发现:
• CH3的Alpha更高 (1.08% vs 0.62%)，说明基于EP的VMG因子能更好地解释策略收益
• 策略具有强烈的小盘倾向 (SMB Beta ≈ 0.8) 和成长倾向 (价值因子为负)
• 两个模型R² > 85%，表明大部分收益可被风格因子解释
'''

plt.figtext(0.5, 0.02, note_text, ha='center', fontsize=10, 
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# 保存图片
output_path = '/Users/xinyutan/Documents/量化投资/quant-project/results/figures/ff3_ch3_comparison_table.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"表格图已保存至: {output_path}")

plt.show()
