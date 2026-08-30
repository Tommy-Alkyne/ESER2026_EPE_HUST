"""生成合成 CFD 数据集，用于端到端冒烟测试（不含真实物理，只保证形状/接口正确）。"""

import argparse
import os

import h5py
import numpy as np


def make_2d(path, n=24, c=3, h=137, w=137, seed=0):
    rng = np.random.RandomState(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
    out = np.zeros((n, c, h, w), dtype=np.float32)
    inp = np.zeros((n, 1, h, w), dtype=np.float32)
    for i in range(n):
        a = 0.5 + rng.rand()
        # 通道量级刻意拉开，用来验证"逐通道归一化 + 反归一化"确实生效
        out[i, 0] = 100.0 * a * np.sin(2 * np.pi * xx) * np.cos(np.pi * yy)   # P
        if c > 1:
            out[i, 1] = 2.0 * a * np.cos(2 * np.pi * yy)                       # Vx
        if c > 2:
            out[i, 2] = 0.3 * a * np.sin(3 * np.pi * xx)                       # Vy
        inp[i, 0] = a
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('input', data=inp)
        f.create_dataset('output', data=out)
    return path


def make_3d(path, n=12, c=3, d=2, h=65, w=65, seed=1):
    rng = np.random.RandomState(seed)
    zz, yy, xx = np.meshgrid(np.linspace(0, 1, d), np.linspace(0, 1, h),
                             np.linspace(0, 1, w), indexing='ij')
    out = np.zeros((n, c, d, h, w), dtype=np.float32)
    inp = np.zeros((n, 1, d, h, w), dtype=np.float32)
    for i in range(n):
        a = 0.5 + rng.rand()
        out[i, 0] = 50.0 * a * np.sin(2 * np.pi * xx) * (1 + zz)
        if c > 1:
            out[i, 1] = 2.0 * a * np.cos(2 * np.pi * yy)
        if c > 2:
            out[i, 2] = 0.3 * a * np.sin(3 * np.pi * xx)
        inp[i, 0] = a
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('input', data=inp)
        f.create_dataset('output', data=out)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='dataset')
    a = p.parse_args()
    print(make_2d(os.path.join(a.out_dir, 'fake2d.hdf5')))
    print(make_2d(os.path.join(a.out_dir, 'fake2d_even.hdf5'), h=128, w=128, seed=7))
    print(make_2d(os.path.join(a.out_dir, 'fake2d_1ch.hdf5'), c=1, h=65, w=65, seed=3))
    print(make_3d(os.path.join(a.out_dir, 'fake3d.hdf5')))
