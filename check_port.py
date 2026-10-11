"""Verify the port against the actual upstream model class and pooling invariance."""
import ast
import logging
from collections import OrderedDict
from pathlib import Path
import torch
from torch import nn
from experiment import Baseline, FastBaseline, Pooled, load

torch.set_num_threads(4)
root = Path(__file__).parent
tree = ast.parse((root / 'model.py').read_text())
classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
namespace = {
    'torch': torch,
    'nn': nn,
    'OrderedDict': OrderedDict,
    'logging': logging}
exec(
    compile(
        ast.Module(
            body=classes,
            type_ignores=[]),
        'upstream-model',
        'exec'),
    namespace)
original = namespace['GraphModel'](300, 5, 3, 80)
ported = Baseline()
ported.g1.load_state_dict(original.graph_modules.dense_0.state_dict())
ported.g2.load_state_dict(original.graph_modules.dense_1.state_dict())
ported.head.load_state_dict(original.fully_connected.state_dict())
a, x, m, gi, h, y, groups = load(root)
with torch.no_grad():
    expected = original(x[:4], a[:4], h[:4])[:, 0]
    actual = ported(a[:4], x[:4], h[:4], m[:4])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    pooled = Pooled().eval()
    perm = torch.randperm(300)
    before = pooled(a[:4], x[:4], h[:4], m[:4])
    after = pooled(a[:4][:, perm][:, :, perm], x[:4]
                   [:, perm], h[:4], m[:4][:, perm])
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
assert len(y) == 2287 and len(set(groups)) == 492
assert torch.isfinite(a).all() and torch.isfinite(x).all()
fast = FastBaseline()
fast.load_state_dict(ported.state_dict())
actual = ported(a[:4], x[:4], h[:4], m[:4])
optimized = fast(a[:4], x[:4], h[:4], m[:4])
torch.testing.assert_close(actual, optimized, rtol=1e-5, atol=1e-6)
actual.sum().backward()
optimized.sum().backward()
for (_, p), (_, q) in zip(ported.named_parameters(), fast.named_parameters()):
    torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-5)
print('PASS: exact upstream forward parity; pooling permutation invariance; 492 structures / 2287 samples; finite inputs')
print('PASS: reassociated baseline forward and gradient equivalence within float32 tolerance')
