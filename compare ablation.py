import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({'font.size': 16})

# 数据准备
configurations = [
    'Baseline',
    'HFUC',
    'Fusion',
    'DHIFI',
    'HFUC+Fusion',
    'HFUC+Fusion+DHIFI',
    'DualConv',
    'HFUC+DualConv',
    'Fusion+DualConv',
    'DHIFI+DualConv',
    'HFUC+Fusion+DualConv',
    'HFUC+Fusion+DHIFI+DualConv(ours)'
]

mbfai = [0.914, 0.919, 0.924, 0.925, 0.928, 0.931, 0.917, 0.918, 0.925, 0.926, 0.931, 0.935]
bccd = [0.909, 0.912, 0.916, 0.919, 0.923, 0.929, 0.910, 0.912, 0.919, 0.921, 0.927, 0.933]
dsb2018 = [0.886, 0.890, 0.897, 0.895, 0.900, 0.903, 0.889, 0.891, 0.895, 0.897, 0.901, 0.907]

x = np.arange(len(configurations))
width = 0.25

fig, ax = plt.subplots(figsize=(15, 8))

# 绘制柱状图
rects1 = ax.bar(x - width, mbfai, width, label='MBEMF', color='#2878B5')
rects2 = ax.bar(x, bccd, width, label='BCCD', color='#9AC9DB')
rects3 = ax.bar(x + width, dsb2018, width, label='DSB2018', color='#C82423')

# 添加性能趋势线，调整线条样式使其更醒目
ax.plot(x - width, mbfai, 'o-', color='#2878B5', alpha=0.7, linewidth=1.5, markersize=4)
ax.plot(x, bccd, 'o-', color='#9AC9DB', alpha=0.7, linewidth=1.5, markersize=4)
ax.plot(x + width, dsb2018, 'o-', color='#C82423', alpha=0.7, linewidth=1.5, markersize=4)

# 添加净提升标注，保留百分号
def add_improvement_label(final, baseline, x_pos, y_pos, x_offset=10, y_offset=10):
    improvement = (final - baseline) * 100  # 转换为百分点
    ax.annotate(f'+{improvement:.1f}%',
                xy=(x_pos, y_pos),
                xytext=(x_offset, y_offset),
                textcoords='offset points',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                fontsize=12)

# 为最终模型添加提升标注，将MBFAI标注向左偏移
add_improvement_label(mbfai[-1], mbfai[0], len(configurations)-1-width, mbfai[-1], -50, 6)  # 向左偏移
add_improvement_label(bccd[-1], bccd[0], len(configurations)-1, bccd[-1])
add_improvement_label(dsb2018[-1], dsb2018[0], len(configurations)-1+width, dsb2018[-1], x_offset=-3, y_offset=10)


# 设置图表属性
ax.set_ylabel('mAP50', fontsize=16)
ax.set_title('Ablation Study Across Different Datasets', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(configurations, rotation=45, ha='right', fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.legend(fontsize=16)

# 添加网格线
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# 设置y轴范围
ax.set_ylim(0.88, 0.94)

# 优化布局
plt.tight_layout()

plt.show()