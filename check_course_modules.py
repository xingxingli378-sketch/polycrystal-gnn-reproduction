"""验证课程参考模块的坐标变换、傅里叶模式和图池化性质。"""
import numpy as np
import torch
from course_model import CourseGNN
from physics_reference import bunge_rotation, demagnetizing_field

rng = np.random.default_rng(124)
angles = rng.uniform(0, 2*np.pi, size=(8, 3))
R = bunge_rotation(*angles.T)
np.testing.assert_allclose(R @ np.swapaxes(R, -1, -2),
                           np.broadcast_to(np.eye(3), R.shape), atol=1e-12)
np.testing.assert_allclose(np.linalg.det(R), 1, atol=1e-12)
vectors = rng.normal(size=(8, 3))
local = np.einsum('nij,nj->ni', R, vectors)
recovered = np.einsum('nji,nj->ni', R, local)
np.testing.assert_allclose(vectors, recovered, atol=1e-12)

shape = (8, 8, 8)
uniform = np.ones((3, *shape))
np.testing.assert_allclose(demagnetizing_field(uniform, [1, 1, 1]), 0, atol=1e-12)
wave = np.broadcast_to(np.sin(2*np.pi*np.arange(8)/8)[:, None, None], shape)
longitudinal = np.zeros((3, *shape)); longitudinal[0] = wave
transverse = np.zeros((3, *shape)); transverse[1] = wave
np.testing.assert_allclose(demagnetizing_field(longitudinal, [1, 1, 1]),
                           -longitudinal, atol=1e-12)
np.testing.assert_allclose(demagnetizing_field(transverse, [1, 1, 1]), 0, atol=1e-12)

torch.manual_seed(124)
torch.set_num_threads(2)
x = torch.randn(2, 7, 5)
chain = torch.eye(7) + torch.diag(torch.ones(6), 1) + torch.diag(torch.ones(6), -1)
inverse_degree = chain.sum(1).pow(-.5)
a = (inverse_degree[:, None] * chain * inverse_degree[None, :]).repeat(2, 1, 1)
h = torch.tensor([[.02], [.04]])
mask = torch.ones(2, 7, 1)
for include_field in [False, True]:
    model = CourseGNN(include_field).eval()
    perm = torch.randperm(7)
    with torch.no_grad():
        prediction = model(a, x, h, mask)
        reordered = model(a[:, perm][:, :, perm], x[:, perm], h, mask[:, perm])
        padded = model(torch.nn.functional.pad(a, (0, 3, 0, 3)),
                       torch.nn.functional.pad(x, (0, 0, 0, 3)), h,
                       torch.nn.functional.pad(mask, (0, 0, 0, 3)))
    torch.testing.assert_close(prediction, reordered, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(prediction, padded, atol=1e-6, rtol=1e-5)
print('通过：旋转矩阵正交性、坐标往返变换、周期性均匀场与单频退磁场解析检查')
print('通过：课程版GCN的节点置换与补零不变性，五特征与六特征入口均可运行')
