"""Evaluate a validation-stagnated grouped baseline after training has stopped.

Exploratory early termination: the patience criterion was introduced after
observing validation stagnation, rather than preregistered before training.
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from experiment import FastBaseline, load

root = Path(__file__).parent
out = root / 'results' / 'grouped_baseline_seed124'
history = list(csv.DictReader((out / 'learning.csv').open()))
best = min(history, key=lambda r: float(r['validation_rmse']))
completed = int(history[-1]['epoch'])
assert completed - int(best['epoch']) >= 100, 'Patience criterion not met'
assert not (out / 'metrics.json').exists(), 'Result already finalized'
torch.set_num_threads(2)
a, x, mask, gi, h, y, groups = load(root)
s = np.load(out / 'split.npz')
train, val, test = s['train'], s['validation'], s['test']
mu, sd = float(y[train].mean()), float(y[train].std())
model = FastBaseline().eval()
model.load_state_dict(torch.load(out / 'best.pt', weights_only=True))
with torch.no_grad():
    yp = np.concatenate([
        model(a[gi[b]], x[gi[b]], h[b], mask[gi[b]]).numpy()
        for b in np.array_split(test, (len(test) + 31) // 32)
    ]) * sd + mu
report = {
    'epochs': completed, 'max_epochs': 300, 'seed': 124,
    'split': 'grouped', 'model': 'baseline', 'threads': 2, 'fast': True,
    'early_stopping_patience': 100,
    'early_stopping_note': 'Exploratory termination after observed validation stagnation; not preregistered',
    'samples': len(y), 'train_samples': len(train),
    'validation_samples': len(val), 'test_samples': len(test),
    'train_test_graph_overlap': len(set(groups[train]) & set(groups[test])),
    'parameters': sum(p.numel() for p in model.parameters()),
    'best_epoch': int(best['epoch']),
    'validation_rmse': float(best['validation_rmse']),
    'test_rmse': float(np.sqrt(np.mean((yp - y[test]) ** 2))),
    'test_mae': float(np.mean(np.abs(yp - y[test]))),
    'test_relative_absolute_error': float(np.abs(yp - y[test]).sum() / np.abs(y[test]).sum()),
    'test_r2': float(1 - ((yp - y[test]) ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum()),
    'seconds': float(history[-1]['elapsed_seconds']),
    'target_mean': mu, 'target_std': sd, 'torch': torch.__version__,
}
(out / 'metrics.json').write_text(json.dumps(report, indent=2))
np.savetxt(out / 'predictions.csv', np.column_stack([
    groups[test], h[test].numpy()[:, 0] * 10000, y[test], yp,
]), delimiter=',', header='structure,field,true,predicted', comments='')
(out / '.running').unlink(missing_ok=True)
print(json.dumps(report, indent=2), flush=True)
