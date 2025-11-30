import matplotlib.pyplot as plt
import numpy as np


def avg_performance(mbemf, bccd, dsb):
    return (mbfai + bccd + dsb) / 3


baseline = avg_performance(0.914, 0.909, 0.886)
hfuc_imp = avg_performance(0.919, 0.912, 0.890) - baseline
ehff_imp = avg_performance(0.924, 0.916, 0.897) - avg_performance(0.919, 0.912, 0.890)
dhifi_imp = avg_performance(0.932, 0.929, 0.903) - avg_performance(0.924, 0.916, 0.897)


values = [baseline, hfuc_imp, ehff_imp, dhifi_imp]
labels = ['Baseline', 'HFUC', 'Fusion', 'DHIFI']
colors = ['#2878B5', '#32B897', '#9AC9DB', '#C82423']


fig, ax = plt.subplots(figsize=(10, 6))


cumsum = baseline
left = 0
for i in range(len(values)):
    if i == 0:  
        plt.bar(left, values[i], width=0.5, color=colors[i], label=labels[i])
    else:  
        plt.bar(left, values[i], width=0.5, bottom=cumsum, color=colors[i], label=labels[i])
    cumsum += values[i]
    left += 1


plt.plot([0.25, len(values)-0.25], [baseline, cumsum], 'k--', alpha=0.3)


plt.ylabel('Average mAP50')
plt.title('Performance Improvement Waterfall')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.3)


cumsum = baseline
for i in range(len(values)):
    if i == 0:
        plt.text(i, values[i]/2, f'{values[i]:.3f}', ha='center')
    else:
        plt.text(i, cumsum + values[i]/2, f'+{values[i]:.3f}', ha='center')
        cumsum += values[i]


plt.text(len(values)-1, cumsum+0.002, f'Final: {cumsum:.3f}', ha='right')

plt.tight_layout()
plt.show()
