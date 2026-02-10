import torch
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def load_data(data_dir, batch_size, input_key=None, prefer_sensor_input=True,
              align_input_to_output=True,shuff=True):
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

    with h5py.File(data_dir, 'r') as f:
        if input_key is not None and input_key in f:
            x_data = f[input_key][()]
        elif prefer_sensor_input and 'sensor_input' in f:
            x_data = f['sensor_input'][()]
        else:
            x_data = f['input'][()]
        y_data = f['output'][()]

    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(y_data.shape))

    # reshape input to 5D (N, C, D, H, W)
    if x_data.ndim == 2:
        n_samples, n_feats = x_data.shape
        side = int(np.sqrt(n_feats))
        if side * side != n_feats:
            raise ValueError(
                "2D input cannot be reshaped into a square map. "
                "Provide `sensor_input` or `input_key` with spatial data."
            )
        x_data = x_data.reshape(n_samples, 1, 1, side, side)
    elif x_data.ndim == 3:
        x_data = x_data[:, None, None, :, :]
    elif x_data.ndim == 4:
        x_data = x_data[:, :, None, :, :]
    elif x_data.ndim != 5:
        raise ValueError("Unsupported input ndim {}".format(x_data.ndim))

    # reshape output to 5D (N, C, D, H, W)
    if y_data.ndim == 4:
        y_data = y_data[:, :, None, :, :]
    elif y_data.ndim != 5:
        raise ValueError("Unsupported output ndim {}".format(y_data.ndim))

    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)

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

    # simple statistics of output data
    y_data_mean = np.mean(y_data, 0)
    y_data_var = np.sum((y_data - y_data_mean) ** 2)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

    return data_loader, stats

if __name__ == "__main__":
    dir = '/home/tommy/code/ESER2026_EPE_HUST/dataset/p_c10e6_128.hdf5'
    size = 16
    data_loader,stats = load_data(dir,size)
    print("\n统计信息:")
    print(f"y_mean 形状: {stats['y_mean'].shape}")
    #print(f"y_mean 前5个值: {stats['y_mean'][:5]}")
    print(f"y_var 值: {stats['y_var']}")
    #print(stats)