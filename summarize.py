"""Build a concise report from completed experiment outputs (standard library)."""
import json
from pathlib import Path
root = Path(__file__).parent
rows = []
for split in ['official', 'grouped']:
    for model in ['baseline', 'pooled']:
        p = root / 'results' / f'{split}_{model}_seed124' / 'metrics.json'
        if p.exists():
            rows.append(json.loads(p.read_text()))
budgets = ', '.join(str(e)
                    for e in sorted({r['epochs'] for r in rows})) or '尚未完成'
lines = [
    '# 实验报告',
    '',
    f'本报告由实际运行的 metrics.json 自动生成。使用 Anaconda Python 3.13.5 / PyTorch 2.14.0+cpu，单随机种子 124。已完成配置的训练预算：{budgets} epoch。作者推荐 1500 epoch；少于该预算的结果属于方法移植与探索实验，不能宣称完整复现论文指标。',
    '',
    '| 划分 | 模型 | Epoch | 参数量 | 测试相对绝对误差 | 测试 RMSE | 测试 R² | 训练/测试微结构重叠 | 最佳 Epoch |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['split']} | {r['model']} | {r['epochs']} | {r['parameters']:,} | {r['test_relative_absolute_error']:.2%} | {r['test_rmse']:.8f} | {r['test_r2']:.4f} | {r['train_test_graph_overlap']} | {r['best_epoch']} |")
lines += ['', '## 比较与限制', '']
for r in rows:
    if 'early_stopping_patience' in r:
        lines.append(f"- {r['split']} / {r['model']}：原定上限 {r['max_epochs']} epoch，实际完成 {r['epochs']} epoch；验证误差连续超过 {r['early_stopping_patience']} epoch 未改善后，提前结束并使用验证集最佳模型。此停止规则是在观察到停滞后加入的探索决策，不是预注册方案。")
for split in ['official', 'grouped']:
    pair = {r['model']: r for r in rows if r['split'] == split}
    if len(pair) == 2:
        b, p = pair['baseline'], pair['pooled']
        if b['epochs'] != p['epochs']:
            lines.append(f'- {split} 划分：两模型预算不同，不作同预算涨分比较。')
            continue
        change = 100 * (b['test_relative_absolute_error'] -
                        p['test_relative_absolute_error']) / b['test_relative_absolute_error']
        lines.append(
            f"- {split} 划分：池化候选相对绝对误差相对基线{'下降' if change>=0 else '上升'} {abs(change):.2f}%。该单次结果不能证明稳定涨分。")
lines += [
    '', '## 取向统计特征替代基线', '',
    '此方法没有 GNN 消息传递层。每种划分先比较三个预设树模型，按验证 RMSE 选择后再评估测试集。它与 GNN 的训练预算不同，不作相同计算预算的性能声明。', '',
    '| 划分 | 验证集选出的模型 | 测试相对绝对误差 | 测试 RMSE | 测试 R² | 微结构重叠 |',
    '|---|---|---:|---:|---:|---:|',
]
descriptor_comparisons = []
for split in ['official', 'grouped']:
    p = root / 'results' / f'{split}_descriptor_seed124' / 'metrics.json'
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    lines.append(f"| {split} | {d['model']} | {d['test_relative_absolute_error']:.2%} | {d['test_rmse']:.8f} | {d['test_r2']:.4f} | {d['train_test_graph_overlap']} |")
    baseline = next((r for r in rows if r['split'] == split and r['model'] == 'baseline'), None)
    if baseline:
        change = 100 * (baseline['test_relative_absolute_error'] - d['test_relative_absolute_error']) / baseline['test_relative_absolute_error']
        descriptor_comparisons.append(f"相对于本轮 {split} 的 GNN 基线，相对绝对误差{'下降' if change >= 0 else '上升'} {abs(change):.2f}%。这不是对完整预算论文结果的超越声明。")
lines.append('')
for comparison in descriptor_comparisons:
    lines.extend([comparison, ''])
lines += ['',
          '- 官方划分按观测随机拆分，同一微结构可出现在训练与测试集；其目标归一化也使用全部数据。',
          '- grouped 划分按完整微结构拆分，目标归一化仅使用训练数据，衡量未见微结构的泛化。两种任务难度不同。',
          '- 模型按验证集 RMSE 选择最佳 checkpoint；未用测试集调参。候选架构/学习率预先确定，但未做单因素消融。',
          '- 原始网络前向与官方源码完全一致性检查已通过；快速基线的矩阵乘法换序通过 float32 容差内输出/梯度对比；池化模型的晶粒编号置换不变性检查已通过。',
          '- 此处没有运行十折完整交叉验证、多随机种子、积分梯度或相场模拟。下一步应按相同设置运行 1500 epoch 与至少三个种子，再做 grouped 交叉验证。',
          '- 中断的训练与 2 epoch smoke 检查不计入正式比较。所有原始结果可从 results/ 检查。',
          '']
(root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
