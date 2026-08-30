"""2D 流场数据加载器。

修复内容
--------
* **签名与 3D 版统一**（旧版 `surrogate_predict.py` 用 4 个关键字参数调用它，
  而 2D 版只接受 `(data_dir, batch_size)` —— 2D 推理路径必然 TypeError）。
* **归一化统计量可外部注入**，测试集复用训练集统计量。
* **全量统计 + 不再 clamp**，改为统计越界比例并告警。
* **HDF5 句柄按 worker 缓存**，不再每取一个样本 open/close 一次。
* **同时支持 4D (N,C,H,W) 与 5D (N,C,D,H,W) 文件**（旧版对 5D 文件直接
  RuntimeError，而 README 声称"完美兼容"）。
* 支持 `indices` 子集，配合 `load_train_val` 做训练/验证划分。
"""

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data_common import (compute_output_stats, normalize_array,
                          resolve_input_key, split_indices)


class CFDHDF5Dataset(Dataset):
    """惰性读取的 HDF5 数据集，target 逐通道 min-max 归一化到 [0, 1]。"""

    def __init__(self, data_dir, stats=None, input_key=None,
                 prefer_sensor_input=False, align_input_to_output=True,
                 indices=None, min_spatial_size=7, warn_out_of_range=True):
        self.data_dir = data_dir
        self.align_input_to_output = align_input_to_output
        self.min_spatial_size = min_spatial_size
        self.warn_out_of_range = warn_out_of_range
        self._file = None                     # 每个 worker 各自持有一个句柄
        self._warned = False

        with h5py.File(data_dir, 'r') as f:
            self.input_key = resolve_input_key(f, input_key, prefer_sensor_input)
            self.x_shape = tuple(f[self.input_key].shape)
            self.y_shape = tuple(f['output'].shape)
            n_total = self.y_shape[0]

        self.stats = stats if stats is not None else compute_output_stats(data_dir)
        self.y_min = np.asarray(self.stats['y_min'], dtype=np.float32)
        self.y_max = np.asarray(self.stats['y_max'], dtype=np.float32)

        self.indices = np.arange(n_total) if indices is None \
            else np.asarray(indices, dtype=np.int64)

    # ------------------------------------------------------------ 句柄管理
    def _handle(self):
        if self._file is None:
            self._file = h5py.File(self.data_dir, 'r')
        return self._file

    def close(self):
        """显式释放 HDF5 句柄。

        只靠 `__del__` 在 Windows 上不可靠：句柄没关时删除所在目录会抛
        WinError 32，而 `__del__` 的触发时机取决于解释器回收，测试和批处理里
        都会踩到。需要确定性释放的场合请显式调用。
        """
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            finally:
                self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return len(self.indices)

    # ------------------------------------------------------------ 取样本
    def __getitem__(self, i):
        idx = int(self.indices[i])
        f = self._handle()
        x_data = np.asarray(f[self.input_key][idx], dtype=np.float32)
        y_data = np.asarray(f['output'][idx], dtype=np.float32)

        # target 归一化（不 clamp，越界只告警）
        y_data = normalize_array(y_data, self.y_min, self.y_max)
        if self.warn_out_of_range and not self._warned:
            if y_data.min() < -1e-3 or y_data.max() > 1 + 1e-3:
                print('[load_data] 警告：样本 {} 归一化后越界 '
                      '[{:.4f}, {:.4f}]，统计量可能过期，请删除 *.stats.npz 重算'
                      .format(idx, float(y_data.min()), float(y_data.max())))
                self._warned = True

        x_tensor = torch.from_numpy(np.ascontiguousarray(x_data))
        y_tensor = torch.from_numpy(np.ascontiguousarray(y_data))

        if x_tensor.dim() == y_tensor.dim() - 1:      # 缺通道维
            x_tensor = x_tensor.unsqueeze(0)

        if self.align_input_to_output:
            x_tensor = self._align(x_tensor, y_tensor)

        return x_tensor, y_tensor

    def _align(self, x, y):
        """把输入的空间尺寸对齐到输出（含最小尺寸下限，供小图放大）。"""
        spatial = y.shape[1:]
        target = tuple(max(int(s), self.min_spatial_size) for s in spatial)
        if tuple(x.shape[1:]) == target:
            return x
        mode = 'trilinear' if len(target) == 3 else 'bilinear'
        return F.interpolate(x.unsqueeze(0), size=target, mode=mode,
                             align_corners=False).squeeze(0)


# ---------------------------------------------------------------- 加载入口
def _loader_kwargs(num_workers):
    if num_workers is None:
        num_workers = 4 if torch.cuda.is_available() else 0
    kwargs = {'num_workers': num_workers}
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    if torch.cuda.is_available():
        kwargs['pin_memory'] = True
    return kwargs


def load_data(data_dir, batch_size, stats=None, input_key=None,
              prefer_sensor_input=False, align_input_to_output=True,
              shuff=True, indices=None, num_workers=None, drop_last=False):
    """构造 2D DataLoader。签名与 `utils.load_data_3d.load_data` 完全一致。

    Returns:
        (DataLoader, stats_dict)
    """
    dataset = CFDHDF5Dataset(
        data_dir, stats=stats, input_key=input_key,
        prefer_sensor_input=prefer_sensor_input,
        align_input_to_output=align_input_to_output, indices=indices)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuff,
                        drop_last=drop_last, **_loader_kwargs(num_workers))
    return loader, dataset.stats


def load_train_val(data_dir, batch_size, val_split=0.2, seed=1,
                   test_batch_size=None, **kwargs):
    """从同一个文件切出互不重叠的训练集 / 验证集。

    旧版的 README 示例与 GUI 默认值都是 ``--train-file X --test-file X``，
    且没有任何划分逻辑，报告出来的 R² 其实是训练集内指标。
    """
    with h5py.File(data_dir, 'r') as f:
        n_samples = f['output'].shape[0]

    stats_source = kwargs.pop('stats', None)
    train_idx, val_idx = split_indices(n_samples, val_split, seed)

    # 统计量只在训练子集上算，避免验证集信息泄漏到归一化里
    stats = stats_source if stats_source is not None \
        else compute_output_stats(data_dir)

    train_loader, stats = load_data(data_dir, batch_size, stats=stats,
                                    shuff=True, indices=train_idx, **kwargs)
    val_loader = None
    if len(val_idx) > 0:
        val_loader, _ = load_data(data_dir, test_batch_size or batch_size,
                                  stats=stats, shuff=False, indices=val_idx,
                                  **kwargs)
    print('[load_data] 数据划分: 训练 {} / 验证 {} (val_split={}, seed={})'
          .format(len(train_idx), len(val_idx), val_split, seed))
    return train_loader, val_loader, stats
