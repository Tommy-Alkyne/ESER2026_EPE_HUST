"""数据统计与归一化的公共实现，2D / 3D 加载器共用。

修复的核心问题
--------------
1. 旧版 min/max **只用前 1000 个样本**估计（还不是随机抽样），然后
   ``torch.clamp(y, 0, 1)`` 把之后所有超出范围的真实值永久削平 —— 回流死角、
   压力峰这些最难也最重要的区域被系统性抹掉。现在全量分块扫描，并把结果缓存
   到 ``<file>.stats.npz``，只在首次或数据变更时重算。
2. 旧版训练集和测试集**各自**算归一化极值，两边被映射到不同的标度上，R² 因此
   不可比。现在统计量可以显式传入，测试集强制复用训练集的统计量。
"""

import os

import h5py
import numpy as np

STATS_VERSION = 2


def _stats_cache_path(data_dir, key):
    return '{}.{}.stats.npz'.format(data_dir, key)


def _cache_is_fresh(cache_path, data_dir):
    if not os.path.exists(cache_path):
        return False
    try:
        return os.path.getmtime(cache_path) >= os.path.getmtime(data_dir)
    except OSError:
        return False


def compute_output_stats(data_dir, key='output', chunk=256, use_cache=True,
                         verbose=True):
    """全量扫描数据集，返回逐通道极值与均值/方差。

    Returns:
        dict(y_min=(C,), y_max=(C,), y_mean=(C,), y_std=(C,), n_samples=int,
             shape=tuple, channel_axis=1)
    """
    cache_path = _stats_cache_path(data_dir, key)
    if use_cache and _cache_is_fresh(cache_path, data_dir):
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if int(cached['version']) == STATS_VERSION:
                if verbose:
                    print('[stats] 复用缓存 {}'.format(cache_path))
                return {
                    'y_min': cached['y_min'], 'y_max': cached['y_max'],
                    'y_mean': cached['y_mean'], 'y_std': cached['y_std'],
                    'n_samples': int(cached['n_samples']),
                    'shape': tuple(cached['shape'].tolist()),
                }
        except Exception as exc:                       # 缓存损坏就重算
            print('[stats] 缓存不可用（{}），重新计算'.format(exc))

    with h5py.File(data_dir, 'r') as f:
        dset = f[key]
        shape = tuple(dset.shape)
        n_samples, n_channels = shape[0], shape[1]

        y_min = np.full(n_channels, np.inf, dtype=np.float64)
        y_max = np.full(n_channels, -np.inf, dtype=np.float64)
        total = np.zeros(n_channels, dtype=np.float64)
        total_sq = np.zeros(n_channels, dtype=np.float64)
        count = 0

        for start in range(0, n_samples, chunk):
            block = np.asarray(dset[start:start + chunk], dtype=np.float64)
            # 除通道轴以外全部归约
            axes = tuple(i for i in range(block.ndim) if i != 1)
            y_min = np.minimum(y_min, block.min(axis=axes))
            y_max = np.maximum(y_max, block.max(axis=axes))
            total += block.sum(axis=axes)
            total_sq += (block ** 2).sum(axis=axes)
            count += block.size // n_channels

    mean = total / max(count, 1)
    var = np.maximum(total_sq / max(count, 1) - mean ** 2, 0.0)
    stats = {
        'y_min': y_min.astype(np.float32),
        'y_max': y_max.astype(np.float32),
        'y_mean': mean.astype(np.float32),
        'y_std': np.sqrt(var).astype(np.float32),
        'n_samples': int(n_samples),
        'shape': shape,
    }

    if verbose:
        print('[stats] 全量扫描 {} 个样本 · 通道极值:'.format(n_samples))
        for c in range(n_channels):
            print('        ch{}: min={:+.6g}  max={:+.6g}  mean={:+.6g}  std={:.6g}'
                  .format(c, stats['y_min'][c], stats['y_max'][c],
                          stats['y_mean'][c], stats['y_std'][c]))

    degenerate = np.isclose(stats['y_max'], stats['y_min'])
    if degenerate.any():
        print('[stats] 警告：通道 {} 的 max==min（常数场），归一化后将恒为 0'
              .format(np.nonzero(degenerate)[0].tolist()))

    if use_cache:
        try:
            np.savez(cache_path, version=STATS_VERSION,
                     y_min=stats['y_min'], y_max=stats['y_max'],
                     y_mean=stats['y_mean'], y_std=stats['y_std'],
                     n_samples=stats['n_samples'],
                     shape=np.array(shape, dtype=np.int64))
        except OSError as exc:
            print('[stats] 缓存写入失败（{}），忽略'.format(exc))

    return stats


def normalize_array(y, y_min, y_max, eps=1e-8):
    """逐通道 min-max 归一化到 [0, 1]。**不做 clamp**：超范围保持原样，
    由调用方决定是否告警，避免静默削平真实极值。

    y 的通道轴：ndim>=4 时为 axis=1（带 batch），ndim==3 时为 axis=0（单样本）。
    """
    channel_axis = 1 if y.ndim >= 4 else 0
    shape = [1] * y.ndim
    shape[channel_axis] = -1
    lo = np.asarray(y_min, dtype=np.float32).reshape(shape)
    hi = np.asarray(y_max, dtype=np.float32).reshape(shape)
    return (y - lo) / (hi - lo + eps)


def denormalize_array(y, y_min, y_max, eps=1e-8):
    """normalize_array 的逆操作。"""
    channel_axis = 1 if y.ndim >= 4 else 0
    shape = [1] * y.ndim
    shape[channel_axis] = -1
    lo = np.asarray(y_min, dtype=np.float32).reshape(shape)
    hi = np.asarray(y_max, dtype=np.float32).reshape(shape)
    return y * (hi - lo + eps) + lo


def resolve_input_key(h5file, input_key=None, prefer_sensor_input=False):
    """确定使用哪一个数据集作为网络输入。"""
    if input_key is not None:
        if input_key not in h5file:
            raise KeyError('HDF5 中不存在数据集 {!r}'.format(input_key))
        return input_key
    candidates = ['sensor_input', 'input'] if prefer_sensor_input \
        else ['input', 'sensor_input']
    for key in candidates:
        if key in h5file:
            return key
    raise KeyError('HDF5 中找不到 input / sensor_input 数据集')


def split_indices(n_samples, val_split, seed=1):
    """固定 seed 的 train/val 下标划分。val_split<=0 时验证集为空。"""
    idx = np.arange(n_samples)
    if val_split is None or val_split <= 0:
        return idx, np.array([], dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_val = int(round(n_samples * float(val_split)))
    n_val = max(1, min(n_val, n_samples - 1)) if n_samples > 1 else 0
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])
