"""Predict a benchmark microstructure at a supplied applied field using a GNN."""
import argparse
import json
from pathlib import Path
import torch
from experiment import Baseline, FastBaseline, Pooled, load

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True, help='GNN result directory relative to repository')
parser.add_argument('--structure', type=int, required=True, help='Benchmark structure ID, 1..492')
parser.add_argument('--field', type=float, required=True, help='Applied field in original dataset units')
args = parser.parse_args()
if not 1 <= args.structure <= 492:
    parser.error('structure must be between 1 and 492')
root = Path(__file__).parent
run = Path(args.run)
if not run.is_absolute():
    run = root / run
metrics = json.loads((run / 'metrics.json').read_text())
if metrics['model'] == 'pooled':
    model = Pooled()
elif metrics['model'] == 'baseline':
    model = FastBaseline() if metrics.get('fast') else Baseline()
else:
    parser.error('run must contain a GNN model; descriptor models use a separate pipeline')
torch.set_num_threads(2)
model.load_state_dict(torch.load(run / 'best.pt', weights_only=True))
model.eval()
a, x, mask, _, _, _, _ = load(root)
i = args.structure - 1
with torch.no_grad():
    standardized = model(a[i:i+1], x[i:i+1], torch.tensor([[args.field / 10000]], dtype=torch.float32), mask[i:i+1]).item()
prediction = standardized * metrics['target_std'] + metrics['target_mean']
print(json.dumps({'structure': args.structure, 'field': args.field,
                  'predicted_magnetostriction': prediction,
                  'predicted_microstrain': prediction * 1e6}, indent=2))
