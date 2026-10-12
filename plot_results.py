"""根据已保存的实验记录生成中文学习曲线和测试散点图。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = Path('C:/Windows/Fonts/msyh.ttc')
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
else:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
root = Path(__file__).parent
split_names = {'official': '官方兼容划分', 'grouped': '按微结构分组'}
model_names = {'baseline': '论文基线', 'pooled': '池化改进GNN', 'descriptor': '统计特征与树模型'}
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, split in zip(axes, ['official', 'grouped']):
    for model in ['baseline', 'pooled']:
        p = root / 'results' / f'{split}_{model}_seed124' / 'learning.csv'
        if not p.exists():
            continue
        d = np.genfromtxt(p, delimiter=',', names=True)
        ax.plot(d['epoch'], d['validation_rmse'] * 1e6, label=model_names[model])
    ax.set(title=split_names[split], xlabel='训练轮次',
           ylabel='验证集RMSE（微应变）')
    ax.legend()
    ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(root / 'results' / 'learning_curves.png', dpi=180)
plt.close(fig)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for row, split in enumerate(['official', 'grouped']):
    for col, model in enumerate(['baseline', 'pooled', 'descriptor']):
        p = root / 'results' / f'{split}_{model}_seed124' / 'predictions.csv'
        if not p.exists():
            continue
        d = np.loadtxt(p, delimiter=',', skiprows=1)
        actual = d[:, 2] * 1e6
        pred = d[:, 3] * 1e6
        ax = axes[row, col]
        ax.scatter(actual, pred, s=12, alpha=.6)
        low = min(actual.min(), pred.min())
        high = max(actual.max(), pred.max())
        ax.plot([low, high], [low, high], 'k--', linewidth=1)
        ax.set(
            title=split_names[split] + '：' + model_names[model],
            xlabel='真实磁致伸缩（微应变）',
            ylabel='预测磁致伸缩（微应变）')
fig.tight_layout()
fig.savefig(root / 'results' / 'test_predictions.png', dpi=180)
