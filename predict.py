"""根据保存的GNN模型预测指定微结构在给定磁场下的磁致伸缩。"""
import argparse
import json
from pathlib import Path
import torch
from experiment import Baseline, FastBaseline, Pooled, load
from course_model import CourseGNN

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True, help='GNN实验结果目录，相对于仓库根目录')
parser.add_argument('--structure', type=int, required=True, help='公共数据集微结构编号，范围1至492')
parser.add_argument('--field', type=float, required=True, help='外加磁场，采用原始数据集单位')
args = parser.parse_args()
if not 1 <= args.structure <= 492:
    parser.error('微结构编号必须在1至492之间')
root = Path(__file__).parent
run = Path(args.run)
if not run.is_absolute():
    run = root / run
metrics = json.loads((run / 'metrics.json').read_text())
if metrics['model'] == 'pooled':
    model = Pooled()
elif metrics['model'] == 'baseline':
    model = FastBaseline() if metrics.get('fast') else Baseline()
elif metrics['model'] == 'course':
    model = CourseGNN(include_field=True)
else:
    parser.error('请指定GNN结果目录；统计特征模型使用单独的预测流程')
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
