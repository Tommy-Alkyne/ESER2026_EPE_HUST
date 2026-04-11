import torch
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def load_data(data_dir, batch_size, input_key=None, prefer_sensor_input=False,
              align_input_to_output=True, shuff=True, combine_inputs=True):
    """Return data loader (3D).

    Args:
        data_dir: directory to hdf5 file, e.g. `dir/to/10243d.hdf5`
        batch_size (int): mini-batch size for loading data
        input_key (str|None): explicit input dataset key in hdf5
        prefer_sensor_input (bool): use `sensor_input` if present
        align_input_to_output (bool): resize input spatial size to match output

    Returns:
        (data_loader (torch.utils.data.DataLoader), stats)
    """

    def _reshape_to_5d(arr):
        if arr.ndim == 3:
            return arr[:, None, None, :, :]
        if arr.ndim == 4:
            return arr[:, :, None, :, :]
        if arr.ndim == 5:
            return arr
        raise ValueError("Unsupported input ndim {}".format(arr.ndim))

    def _vector_to_5d(arr, target_shape):
        n_samples = arr.shape[0]
        vec = arr.reshape(n_samples, -1)
        vec = vec[:, :, None, None, None]
        d, h, w = target_shape
        return np.tile(vec, (1, 1, d, h, w))

    with h5py.File(data_dir, 'r') as f:
        y_data = f['output'][()]

        if input_key is not None and input_key in f:
            x_data = f[input_key][()]
        elif combine_inputs and 'spatial_input' in f:
            spatial_data = _reshape_to_5d(f['spatial_input'][()])
            _, _, d_s, h_s, w_s = spatial_data.shape
            parts = [spatial_data]
            if 'input' in f:
                parts.append(_vector_to_5d(f['input'][()], (d_s, h_s, w_s)))
            if 'sensor_input' in f:
                parts.append(_vector_to_5d(f['sensor_input'][()], (d_s, h_s, w_s)))
            x_data = np.concatenate(parts, axis=1)
        elif 'input' in f and (not prefer_sensor_input or 'sensor_input' not in f):
            x_data = f['input'][()]
        elif 'sensor_input' in f:
            x_data = f['sensor_input'][()]
        else:
            raise KeyError('No valid input dataset found in HDF5 file')

    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(y_data.shape))

    # reshape input to 5D (N, C, D, H, W)
    if x_data.ndim == 2:
        n_samples, n_feats = x_data.shape
        side = int(np.sqrt(n_feats))
        if side * side != n_feats:
            raise ValueError(
                "2D input cannot be reshaped into a square map. "
                "Provide `spatial_input` or `input_key` with spatial data."
            )
        x_data = x_data.reshape(n_samples, 1, 1, side, side)
    elif x_data.ndim in (3, 4, 5):
        x_data = _reshape_to_5d(x_data)
    else:
        raise ValueError("Unsupported input ndim {}".format(x_data.ndim))

    # reshape output to 5D (N, C, D, H, W)
    if y_data.ndim == 4:
        y_data = y_data[:, :, None, :, :]
    elif y_data.ndim != 5:
        raise ValueError("Unsupported output ndim {}".format(y_data.ndim))

    # ==========================================
    # [新增] 计算各物理通道的全局最大最小值 (3D)
    # y_data 此时 shape 固定为 (N, C, D, H, W)
    # ==========================================
    y_min = np.min(y_data, axis=(0, 2, 3, 4)) # Shape: (C,)
    y_max = np.max(y_data, axis=(0, 2, 3, 4)) # Shape: (C,)
    
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)

    # ==========================================
    # [新增] 对 Target 进行严格的 Min-Max 归一化 -> [0, 1]
    # ==========================================
    # 扩展维度以匹配 5D Tensor广播机制：(1, C, 1, 1, 1)
    y_min_t = torch.tensor(y_min, dtype=torch.float32).view(1, -1, 1, 1, 1)
    y_max_t = torch.tensor(y_max, dtype=torch.float32).view(1, -1, 1, 1, 1)
    
    # 归一化公式: (y - min) / (max - min)
    y_tensor = (y_tensor - y_min_t) / (y_max_t - y_min_t + 1e-8)
    # 防止极微小越界，硬截断到 0-1
    y_tensor = torch.clamp(y_tensor, 0.0, 1.0)

    if align_input_to_output:
        _, _, d_out, h_out, w_out = y_tensor.shape
        _, _, d_in, h_in, w_in = x_tensor.shape
        if (d_in, h_in, w_in) != (d_out, h_out, w_out):
            x_tensor = F.interpolate(
                x_tensor,
                size=(d_out, h_out, w_out),
                mode='trilinear',
                align_corners=False
            )

    kwargs = {'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

    dataset = TensorDataset(x_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuff, **kwargs)

    # simple statistics of output data (基于原数据计算，或者可调)
    y_data_mean = np.mean(y_data, 0)
    y_data_var = np.sum((y_data - y_data_mean) ** 2)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var
    
    # ==========================================
    # [新增] 导出极值供其它流程（如物理损失、后处理还原）使用
    # ==========================================
    stats['y_min'] = y_min
    stats['y_max'] = y_max

    return data_loader, stats

if __name__ == "__main__":
    dir = '/home/tommy/code/ESER2026_EPE_HUST/dataset/p_c10e6_128.hdf5'
    size = 16
    data_loader, stats = load_data(dir, size)
    print("\n统计信息:")
    print(f"y_mean 形状: {stats['y_mean'].shape}")
    print(f"y_var 值: {stats['y_var']}")
    print(f"y_min 值: {stats['y_min']}")
    print(f"y_max 值: {stats['y_max']}")