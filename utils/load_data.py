import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset


def load_data(data_dir, batch_size):
    """Return data loader

    Args:
        data_dir: directory to hdf5 file, e.g. `dir/to/kle4225_lhs256.hdf5`
        batch_size (int): mini-batch size for loading data

    Returns:
        (data_loader (torch.utils.data.DataLoader), stats)
    """

    with h5py.File(data_dir, 'r') as f:
        x_data = f['input'][()]
        y_data = f['output'][()]

    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(y_data.shape))

    kwargs = {'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

    dataset = TensorDataset(torch.tensor(x_data), torch.tensor(y_data))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # simple statistics of output data
    y_data_mean = np.mean(y_data, 0)
    y_data_var = np.sum((y_data - y_data_mean) ** 2)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

    return data_loader, stats

if __name__ == "__main__":
    dir = '/home/tommy/code/ESER2026_EPE_HUST/dataset/kle4225_lhs128.hdf5'
    size = 16
    data_loader,stats = load_data(dir,size)
    print("\n统计信息:")
    # print(f"y_mean 形状: {stats['y_mean'].shape}")
    # print(f"y_mean 前5个值: {stats['y_mean'][:5]}")
    # print(f"y_var 值: {stats['y_var']}")
    print(stats)