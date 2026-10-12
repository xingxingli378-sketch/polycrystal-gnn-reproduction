"""在Anaconda环境运行更正后的论文基线、池化改进和课程版GCN。"""
import argparse
import atexit
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from course_model import CourseGNN


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.g1 = nn.Linear(5, 80)
        self.g2 = nn.Linear(80, 3)
        self.head = nn.Sequential(
            nn.Linear(901, 1024), nn.ReLU(),
            nn.Linear(1024, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, a, x, h, mask):
        x = torch.sigmoid(self.g1(a @ x))
        x = torch.sigmoid(self.g2(a @ x))
        return self.head(torch.cat([x.flatten(1), h], 1)).squeeze(1)


class Pooled(nn.Module):
    def __init__(self):
        super().__init__()
        self.g1 = nn.Linear(8, 64)
        self.g2 = nn.Linear(64, 64)
        self.head = nn.Sequential(
            nn.Linear(130, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, a, x, h, mask):
        x = torch.cat([torch.sin(x[:, :, :3]), torch.cos(
            x[:, :, :3]), x[:, :, 3:]], 2) * mask
        x = torch.relu(self.g1(a @ x)) * mask
        x = torch.relu(self.g2(a @ x) + x) * mask
        mean = x.sum(1) / mask.sum(1).clamp_min(1)
        maximum = x.masked_fill(~mask.bool(), -1e9).max(1).values
        count = mask.sum(1) / 300
        return self.head(torch.cat([mean, maximum, h, count], 1)).squeeze(1)


class FastBaseline(Baseline):
    def forward(self, a, x, h, mask):
        x = torch.sigmoid(self.g1(a @ x))
        # A(XW)+b = (AX)W+b; the bias must be added AFTER aggregation.
        projected = torch.nn.functional.linear(x, self.g2.weight, None)
        x = torch.sigmoid(a @ projected + self.g2.bias)
        return self.head(torch.cat([x.flatten(1), h], 1)).squeeze(1)


def load(root):
    cache = root / 'data_cache.pt'
    if cache.exists():
        return torch.load(cache, weights_only=False)
    aa = []
    xx = []
    masks = []
    gi = []
    fields = []
    ys = []
    groups = []
    for sid in range(1, 493):
        p = root / 'data' / f'structure-{sid}'
        a = np.loadtxt(p / 'neighbor.txt', dtype=np.float32)
        np.fill_diagonal(a, 1)
        d = a.sum(0)**-.5
        a = d[:, None] * a * d[None, :]
        x = np.loadtxt(p / 'feature.txt', dtype=np.float32)[:, 1:]
        n = len(x)
        for j in [3, 4]:
            x[:, j] = (x[:, j] - x[:, j].mean()) / max(x[:, j].std(), 1e-8)
        aa.append(np.pad(a, ((0, 300 - n), (0, 300 - n))))
        xx.append(np.pad(x, ((0, 300 - n), (0, 0))))
        masks.append(((np.arange(300) > 0) & (np.arange(300) < n))[:, None])
        props = np.atleast_2d(np.loadtxt(p / 'property.txt', dtype=np.float32))
        for h, y in props[props.min(1) >= 0]:
            gi.append(sid - 1)
            fields.append([h / 10000])
            ys.append(y)
            groups.append(sid)
    result = (
        torch.tensor(
            np.array(aa)),
        torch.tensor(
            np.array(xx)),
        torch.tensor(
            np.array(masks),
            dtype=torch.float32),
        torch.tensor(gi),
        torch.tensor(fields),
        np.array(ys),
        np.array(groups))
    torch.save(result, cache)
    return result


def main():
    ap = argparse.ArgumentParser(description='公共数据集GNN训练与评测：论文基线、池化改进、课程版GCN')
    ap.add_argument('--epochs', type=int, default=1500, help='训练轮数上限')
    ap.add_argument('--seed', type=int, default=124, help='随机种子')
    ap.add_argument(
        '--split',
        choices=[
            'official',
            'grouped'],
        default='grouped', help='official：官方兼容划分；grouped：按微结构隔离')
    ap.add_argument(
        '--model',
        choices=[
            'baseline',
            'pooled',
            'course'],
        default='baseline', help='baseline：论文基线；pooled：池化改进；course：课程版GCN')
    ap.add_argument('--threads', type=int, default=0, help='CPU线程数，0表示自动选择')
    ap.add_argument('--fast', action='store_true', help='启用论文基线的矩阵计算换序加速')
    ap.add_argument('--overwrite', action='store_true', help='明确覆盖同目录已有结果，请先归档')
    args = ap.parse_args()
    if args.threads == 0:
        args.threads = 2 if args.model == 'baseline' else 4
    root = Path(__file__).resolve().parent
    out = root / 'results' / f'{args.split}_{args.model}_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists() and not args.overwrite:
        previous = json.loads((out / 'metrics.json').read_text())
        if previous.get('max_epochs', previous['epochs']) != args.epochs:
            raise RuntimeError(
                '已有结果的训练轮数不同。请先归档旧目录，再使用 --overwrite 重新运行。')
        print(
            '已有完整结果，默认跳过；重新运行需使用 --overwrite：',
            out,
            flush=True)
        return
    lock = out / '.running'
    try:
        with lock.open('x') as f:
            f.write('模型正在训练\n')
    except FileExistsError:
        print('同一配置已有运行中的实验：', out, flush=True)
        return
    atexit.register(lambda: lock.unlink(missing_ok=True))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    a, x, mask, gi, h, y, groups = load(root)
    # Match original random.Random(123) graph permutation and shuffled 10-fold
    # KFold.
    graphseq = list(range(1, 493))
    random.Random(123).shuffle(graphseq)
    order = np.concatenate([np.where(groups == g)[0] for g in graphseq])
    rng = np.random.RandomState(123)
    if args.split == 'official':
        perm = rng.permutation(len(y))
        folds = np.array_split(perm, 10)
        val = order[np.sort(folds[8])]
        test = order[np.sort(folds[9])]
    else:
        folds = np.array_split(rng.permutation(492) + 1, 10)
        val = np.where(np.isin(groups, folds[8]))[0]
        test = np.where(np.isin(groups, folds[9]))[0]
    train = np.setdiff1d(np.arange(len(y)), np.concatenate([val, test]))
    # Official compatibility includes global target normalization; grouped
    # uses train only.
    norm = y if args.split == 'official' else y[train]
    mu = float(norm.mean())
    sd = float(norm.std())
    target = torch.tensor((y - mu) / sd)
    if args.model == 'baseline':
        model = FastBaseline() if args.fast else Baseline()
    elif args.model == 'course':
        model = CourseGNN(include_field=True)
    else:
        model = Pooled()
    opt = torch.optim.Adam(model.parameters(),
                           lr=.0005 if args.model == 'baseline' else .001)
    out = root / 'results' / f'{args.split}_{args.model}_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / 'split.npz',
        train=train,
        validation=val,
        test=test,
        groups=groups)

    def predict(ids):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(a[gi[b]], x[gi[b]], h[b], mask[gi[b]]).numpy(
            ) for b in np.array_split(ids, max(1, (len(ids) + 31) // 32))]) * sd + mu
    best = float('inf')
    start = time.time()
    with (out / 'learning.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_normalized_mse',
                         'validation_rmse', 'elapsed_seconds'])
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for b in np.array_split(np.random.permutation(
                    train), max(1, (len(train) + 31) // 32)):
                opt.zero_grad()
                pred = model(a[gi[b]], x[gi[b]], h[b], mask[gi[b]])
                loss = ((pred - target[b])**2).mean()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            vp = predict(val)
            rmse = float(np.sqrt(np.mean((vp - y[val])**2)))
            writer.writerow([epoch + 1, np.mean(losses),
                             rmse, time.time() - start])
            f.flush()
            if rmse < best:
                best = rmse
                bestepoch = epoch + 1
                torch.save(model.state_dict(), out / 'best.pt')
            if (epoch + 1) % 25 == 0 or epoch == 0:
                print(
                    args.split,
                    args.model,
                    epoch + 1,
                    '验证RMSE',
                    rmse,
                    '耗时秒',
                    round(
                        time.time() - start),
                    flush=True)
        torch.save(model.state_dict(), out / 'last.pt')
    model.load_state_dict(torch.load(out / 'best.pt', weights_only=True))
    yp = predict(test)
    report = vars(args) | {'samples': len(y),
                           'train_samples': len(train),
                           'validation_samples': len(val),
                           'test_samples': len(test),
                           'train_test_graph_overlap': len(set(groups[train]) & set(groups[test])),
                           'parameters': sum(p.numel() for p in model.parameters()),
                           'best_epoch': bestepoch,
                           'validation_rmse': best,
                           'test_rmse': float(np.sqrt(np.mean((yp - y[test])**2))),
                           'test_mae': float(np.mean(np.abs(yp - y[test]))),
                           'test_relative_absolute_error': float(np.abs(yp - y[test]).sum() / np.abs(y[test]).sum()),
                           'test_r2': float(1 - ((yp - y[test])**2).sum() / ((y[test] - y[test].mean())**2).sum()),
                           'seconds': time.time() - start,
                           'target_mean': mu,
                           'target_std': sd,
                           'torch': torch.__version__}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    np.savetxt(out / 'predictions.csv',
               np.column_stack([groups[test],
                                h[test].numpy()[:,
                                                0] * 10000,
                                y[test],
                                yp]),
               delimiter=',',
               header='structure,field,true,predicted',
               comments='')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
