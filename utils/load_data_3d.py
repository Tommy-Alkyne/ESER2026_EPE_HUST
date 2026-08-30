"""3D 体素流场数据加载器。

与 2D 版保持**完全一致的函数签名**（旧版两者签名不同，导致
`surrogate_predict.py` 在 2D 路径下必然 TypeError）。

3D 版特有的输入组装逻辑（spatial_input + 标量向量平铺成体素）保留，但归一化
统计量改为走 `utils.data_common`，与 2D 共用同一套逻辑与缓存。
"""

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data_common import (compute_output_stats, normalize_array,
                          resolve_input_key, split_indices)


def _reshape_to_5d(arr):
    if arr.ndim == 3:
        return arr[:, None, None, :, :]
    if arr.ndim == 4:
        return arr[:, :, None, :, :]
    if arr.ndim == 5:
        return arr
    raise ValueError('不支持的输入维度 {}'.format(arr.ndim))


def _vector_to_5d(arr, target_shape):
    """把每个样本的标量特征向量平铺成体素通道。"""
    n_samples = arr.shape[0]
    vec = arr.reshape(n_samples, -1)[:, :, None, None, None]
    d, h, w = target_shape
    return np.tile(vec, (1, 1, d, h, w))


def _read_inputs(f, input_key, prefer_sensor_input, combine_inputs):
    if input_key is not None:
        return np.asarray(f[input_key][()], dtype=np.float32)
    if combine_inputs and 'spatial_input' in f:
        spatial = _reshape_to_5d(np.asarray(f['spatial_input'][()], dtype=np.float32))
        _, _, d_s, h_s, w_s = spatial.shape
        parts = [spatial]
        for extra in ('input', 'sensor_input'):
            if extra in f:
                parts.append(_vector_to_5d(
                    np.asarray(f[extra][()], dtype=np.float32), (d_s, h_s, w_s)))
        return np.concatenate(parts, axis=1)
    key = resolve_input_key(f, None, prefer_sensor_input)
    return np.asarray(f[key][()], dtype=np.float32)


def load_data(data_dir, batch_size, stats=None, input_key=None,
              prefer_sensor_input=False, align_input_to_output=True,
              shuff=True, indices=None, num_workers=None, drop_last=False,
              combine_inputs=True):
    """构造 3D DataLoader。签名与 `utils.load_data.load_data` 一致。"""
    with h5py.File(data_dir, 'r') as f:
        y_data = np.asarray(f['output'][()], dtype=np.float32)
        x_data = _read_inputs(f, input_key, prefer_sensor_input, combine_inputs)

    print('[load_data_3d] input shape {} · output shape {}'
          .format(x_data.shape, y_data.shape))

    # 统一成 5D (N, C, D, H, W)
    if x_data.ndim == 2:
        n_samples, n_feats = x_data.shape
        side = int(np.sqrt(n_feats))
        if side * side != n_feats:
            raise ValueError('二维输入无法重排成方形网格，请提供 spatial_input '
                             '或用 --input-key 指定带空间结构的数据集')
        x_data = x_data.reshape(n_samples, 1, 1, side, side)
    else:
        x_data = _reshape_to_5d(x_data)

    if y_data.ndim == 4:
        y_data = y_data[:, :, None, :, :]
    elif y_data.ndim != 5:
        raise ValueError('不支持的输出维度 {}'.format(y_data.ndim))

    if stats is None:
        stats = compute_output_stats(data_dir)
    y_min = np.asarray(stats['y_min'], dtype=np.float32)
    y_max = np.asarray(stats['y_max'], dtype=np.float32)

    # 若文件里 output 本身是 4D，统计量的通道轴与这里补维后一致，可直接用
    y_norm = normalize_array(y_data, y_min, y_max)
    if y_norm.min() < -1e-3 or y_norm.max() > 1 + 1e-3:
        print('[load_data_3d] 警告：归一化后越界 [{:.4f}, {:.4f}]，'
              '统计量可能来自另一个文件'.format(float(y_norm.min()), float(y_norm.max())))

    x_tensor = torch.from_numpy(np.ascontiguousarray(x_data))
    y_tensor = torch.from_numpy(np.ascontiguousarray(y_norm))

    if align_input_to_output:
        if x_tensor.shape[-3:] != y_tensor.shape[-3:]:
            x_tensor = F.interpolate(x_tensor, size=tuple(y_tensor.shape[-3:]),
                                     mode='trilinear', align_corners=False)

    if indices is not None:
        sel = torch.as_tensor(np.asarray(indices, dtype=np.int64))
        x_tensor, y_tensor = x_tensor[sel], y_tensor[sel]

    kwargs = {}
    if num_workers:
        kwargs['num_workers'] = num_workers
    if torch.cuda.is_available():
        kwargs['pin_memory'] = True

    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size,
                        shuffle=shuff, drop_last=drop_last, **kwargs)
    return loader, stats


def load_train_val(data_dir, batch_size, val_split=0.2, seed=1,
                   test_batch_size=None, **kwargs):
    """从同一个文件切出互不重叠的训练集 / 验证集。"""
    with h5py.File(data_dir, 'r') as f:
        n_samples = f['output'].shape[0]

    stats = kwargs.pop('stats', None) or compute_output_stats(data_dir)
    train_idx, val_idx = split_indices(n_samples, val_split, seed)

    train_loader, stats = load_data(data_dir, batch_size, stats=stats,
                                    shuff=True, indices=train_idx, **kwargs)
    val_loader = None
    if len(val_idx) > 0:
        val_loader, _ = load_data(data_dir, test_batch_size or batch_size,
                                  stats=stats, shuff=False, indices=val_idx,
                                  **kwargs)
    print('[load_data_3d] 数据划分: 训练 {} / 验证 {} (val_split={}, seed={})'
          .format(len(train_idx), len(val_idx), val_split, seed))
    return train_loader, val_loader, stats
