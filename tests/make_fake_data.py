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


def make_flow_stage1(path, n=24, h=96, w=96, seed=11):
    """地铁站方案阶段一：工况 -> (P, Vx, Vy, T)，对应 --channel-order pvvt。"""
    rng = np.random.RandomState(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
    out = np.zeros((n, 4, h, w), dtype=np.float32)
    inp = np.zeros((n, 3, h, w), dtype=np.float32)
    for i in range(n):
        vent = 0.5 + rng.rand()             # 通风量
        door = float(rng.randint(2))        # 屏蔽门开闭
        train = float(rng.randint(4))       # 列车工况
        out[i, 0] = 120.0 * vent * np.sin(2 * np.pi * xx) * np.cos(np.pi * yy)   # P
        out[i, 1] = 2.0 * vent * np.cos(2 * np.pi * yy) + 0.3 * door            # Vx
        out[i, 2] = 0.4 * vent * np.sin(3 * np.pi * xx)                          # Vy
        out[i, 3] = 293.0 + 6.0 * vent * np.exp(-((xx - 0.5) ** 2) * 8)          # T
        inp[i, 0], inp[i, 1], inp[i, 2] = vent, door, train
    return _write(path, inp, out)


def make_dispersion_stage2(path, n=32, h=96, w=96, seed=13):
    """地铁站方案阶段二：(Vx, Vy, 源位置图, t) -> C，对应 --channel-order c。

    浓度用移动高斯烟团构造，量级和形态接近真实泄漏扩散，冒烟测试才有意义。
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
    out = np.zeros((n, 1, h, w), dtype=np.float32)
    inp = np.zeros((n, 4, h, w), dtype=np.float32)
    sites = [(0.25, 0.2), (0.5, 0.5), (0.75, 0.8)]      # 固定关心点位
    diffusivity = 2e-3
    for i in range(n):
        u, v = 0.05 + 0.05 * rng.rand(), 0.02 * rng.randn()
        sy, sx = sites[i % len(sites)]
        t = float([5, 10, 20, 40][i % 4])
        denom = 4.0 * diffusivity * t
        r2 = (xx - sx - u * t * 0.01) ** 2 + (yy - sy - v * t * 0.01) ** 2
        out[i, 0] = np.exp(-r2 / denom) / (np.pi * denom)
        inp[i, 0] = u                                    # Vx
        inp[i, 1] = v                                    # Vy
        inp[i, 2] = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / 2e-3)   # 源位置图
        inp[i, 3] = t                                    # 时刻
    return _write(path, inp, out)


def make_station_mask(path, h=96, w=96):
    """非矩形的车站平面掩码，用来验证 .npy 栅格掩码通路。"""
    mask = np.zeros((h, w), dtype=np.float32)
    mask[int(h * 0.40):int(h * 0.60), :] = 1                    # 站台层，横贯整幅
    mask[int(h * 0.15):int(h * 0.40), int(w * 0.20):int(w * 0.30)] = 1   # 楼梯 1
    mask[int(h * 0.15):int(h * 0.40), int(w * 0.70):int(w * 0.80)] = 1   # 楼梯 2
    for i in range(int(h * 0.2)):                               # 斜向通道
        r = int(h * 0.60) + i // 2
        c0 = int(w * 0.45) + i
        if r < h and c0 < w:
            mask[r, c0:min(c0 + 4, w)] = 1
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.save(path, mask)
    return path


def _write(path, inp, out):
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
    print(make_flow_stage1(os.path.join(a.out_dir, 'fake_stage1.hdf5')))
    print(make_dispersion_stage2(os.path.join(a.out_dir, 'fake_stage2.hdf5')))
    print(make_station_mask(os.path.join(a.out_dir, 'station_mask.npy')))
