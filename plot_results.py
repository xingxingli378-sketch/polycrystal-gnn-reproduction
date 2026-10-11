"""Optional publication-style plots; requires matplotlib and numpy."""
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
root = Path(__file__).parent
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, split in zip(axes, ['official', 'grouped']):
    for model in ['baseline', 'pooled']:
        p = root / 'results' / f'{split}_{model}_seed124' / 'learning.csv'
        if not p.exists():
            continue
        d = np.genfromtxt(p, delimiter=',', names=True)
        ax.plot(d['epoch'], d['validation_rmse'] * 1e6, label=model)
    ax.set(title=split + ' split', xlabel='Epoch',
           ylabel='Validation RMSE (microstrain)')
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
            title=split + ' / ' + model,
            xlabel='True magnetostriction (microstrain)',
            ylabel='Predicted (microstrain)')
fig.tight_layout()
fig.savefig(root / 'results' / 'test_predictions.png', dpi=180)
