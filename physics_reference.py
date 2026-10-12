"""阶段汇报中的Bunge旋转与周期性FFT退磁场参考模块。

仅实现两个可独立验证的计算环节，不包含完整LLG求解、弹性平衡或标签生成。
"""
import numpy as np


def bunge_rotation(phi1, Phi, phi2):
    """返回全局坐标到晶体坐标的旋转矩阵，角度单位为弧度。

    支持标量或可广播的数组，输出形状为 (..., 3, 3)。
    公式对应汇报第21页截图，R的转置用于反向坐标变换。
    """
    phi1, Phi, phi2 = np.broadcast_arrays(phi1, Phi, phi2)
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    rows = [
        np.stack([c1*c2-s1*c*s2, s1*c2+c1*c*s2, s*s2], axis=-1),
        np.stack([-c1*s2-s1*c*c2, -s1*s2+c1*c*c2, s*c2], axis=-1),
        np.stack([s1*s, -c1*s, c], axis=-1),
    ]
    return np.stack(rows, axis=-2)


def demagnetizing_field(magnetization, lengths_m):
    """计算周期性盒子的退磁场，磁化强度和输出场均采用 A/m。

    magnetization 的形状为 (3, Nx, Ny, Nz)，lengths_m 为三个方向的物理长度。
    若输入是单位磁化方向m，应先乘饱和磁化强度Ms，再调用本函数。
    k=0模式约定为零。有限样品的形状退磁场需要另外处理。
    """
    M = np.asarray(magnetization, dtype=float)
    lengths = np.asarray(lengths_m, dtype=float)
    if M.ndim != 4 or M.shape[0] != 3:
        raise ValueError('磁化场形状应为 (3, Nx, Ny, Nz)')
    if lengths.shape != (3,) or np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError('请提供三个有限且大于零的物理长度，单位为米')
    if not np.isfinite(M).all():
        raise ValueError('磁化场含有非有限数值')
    k = np.array(np.meshgrid(*[
        2*np.pi*np.fft.fftfreq(n, d=length/n)
        for n, length in zip(M.shape[1:], lengths)
    ], indexing='ij'))
    k2 = (k*k).sum(axis=0)
    # 单独建立安全分母，保留原始k²，避免零频占位符影响后续交换场。
    denominator = np.where(k2 == 0, 1.0, k2)
    transformed = np.fft.fftn(M, axes=(1, 2, 3))
    longitudinal = (k*transformed).sum(axis=0) / denominator
    result_k = -k*longitudinal
    result_k[:, 0, 0, 0] = 0
    return np.fft.ifftn(result_k, axes=(1, 2, 3)).real
