import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_data(data_dir, batch_size, test_batch_size=None, split_ratio=None, split_seed=None):
    """Return data loader

    Args:
        data_dir: directory to hdf5 file, e.g. `dir/to/kle4225_lhs256.hdf5`
        batch_size (int): mini-batch size for loading data
        test_batch_size (int|None): batch size for the split-out test set
        split_ratio (float|None): if not None, split the file into train/test
            with this fraction used for training (the rest for testing) and
            return (train_loader, test_loader, stats)
        split_seed (int|None): seed for a reproducible random split

    Returns:
        If split_ratio is None:
            (data_loader (torch.utils.data.DataLoader), stats)
        Else:
            (train_loader, test_loader, stats)
    """

    with h5py.File(data_dir, 'r') as f:
        x_data = f['input'][()]
        y_data = f['output'][()]

    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(y_data.shape))

    kwargs = {'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

    dataset = TensorDataset(torch.tensor(x_data), torch.tensor(y_data))

    # simple statistics of output data
    y_data_mean = np.mean(y_data, 0)
    y_data_var = np.sum((y_data - y_data_mean) ** 2)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

    if split_ratio is not None:
        if not 0.0 < split_ratio < 1.0:
            raise ValueError('split_ratio must be in (0, 1), got {}'.format(split_ratio))
        n_total = len(dataset)
        n_train = int(n_total * split_ratio)
        n_test = n_total - n_train
        if n_train < 1 or n_test < 1:
            raise ValueError('split_ratio {} leaves an empty train/test set ({} samples)'.format(split_ratio, n_total))
        if test_batch_size is None:
            test_batch_size = batch_size
        gen = torch.Generator().manual_seed(split_seed) if split_seed is not None else None
        train_set, test_set = random_split(dataset, [n_train, n_test], generator=gen)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)
        print("split into train({}) / test({}) samples at ratio {}".format(n_train, n_test, split_ratio))
        return train_loader, test_loader, stats

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return data_loader, stats

if __name__ == "__main__":
    dir = '/home/tommy/code/ESER2026_EPE_HUST/dataset/kle4225_lhs1024.hdf5'
    size = 16
    data_loader,stats = load_data(dir,size)
    print("\n统计信息:")
    print(f"y_mean 形状: {stats['y_mean'].shape}")
    #print(f"y_mean 前5个值: {stats['y_mean'][:5]}")
    print(f"y_var 值: {stats['y_var']}")
    #print(stats)