import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils.hybrid_targets import build_hybrid_dataset


def _prepare_mask(mask_data, n_samples, target_shape):
    """Return a bool mask shaped (N, 1, H, W)."""
    if mask_data is None:
        return np.ones((n_samples, 1) + tuple(target_shape), dtype=bool)

    mask = np.asarray(mask_data)
    target_shape = tuple(target_shape)
    if mask.ndim == 2 and tuple(mask.shape) == target_shape:
        mask = mask[None, None, :, :]
    elif mask.ndim == 3 and tuple(mask.shape[1:]) == target_shape:
        mask = mask[:, None, :, :]
    elif (
        mask.ndim == 4
        and mask.shape[1] == 1
        and tuple(mask.shape[2:]) == target_shape
    ):
        pass
    else:
        raise ValueError(
            "Mask shape {} cannot be aligned with output spatial shape {}".format(
                mask.shape, target_shape
            )
        )

    if mask.shape[0] == 1 and n_samples > 1:
        mask = np.broadcast_to(mask, (n_samples,) + mask.shape[1:])
    elif mask.shape[0] != n_samples:
        raise ValueError(
            "Mask batch dimension {} does not match sample count {}".format(
                mask.shape[0], n_samples
            )
        )
    return mask.astype(bool, copy=False)


def load_data(data_dir, batch_size, test_batch_size=None, split_ratio=None,
              split_seed=None, shuff=True, prepare_hybrid_targets=False,
              cls_thresholds=None, num_classes=None, target_channel=0,
              ignore_index=-100, center_max_source="target",
              center_max_value=1.0, bin_center_ratios=None):
    """Return data loader

    Args:
        data_dir: directory to hdf5 file, e.g. `dir/to/kle4225_lhs256.hdf5`
        batch_size (int): mini-batch size for loading data
        test_batch_size (int|None): batch size for the split-out test set
        split_ratio (float|None): if not None, split the file into train/test
            with this fraction used for training (the rest for testing) and
            return (train_loader, test_loader, stats)
        split_seed (int|None): seed for a reproducible random split
        prepare_hybrid_targets (bool): return named, loss-ready target fields

    Returns:
        If split_ratio is None:
            (data_loader (torch.utils.data.DataLoader), stats)
        Else:
            (train_loader, test_loader, stats)
    """

    with h5py.File(data_dir, 'r') as f:
        x_data = f['input'][()]
        y_data = f['output'][()]
        mask_data = f['mask'][()] if 'mask' in f else None

    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(y_data.shape))

    kwargs = {'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    mask_tensor = torch.tensor(
        _prepare_mask(mask_data, y_data.shape[0], y_data.shape[-2:]),
        dtype=torch.bool,
    )
    if prepare_hybrid_targets:
        preprocess_kwargs = {
            'num_classes': num_classes,
            'target_channel': target_channel,
            'ignore_index': ignore_index,
            'center_max_source': center_max_source,
            'center_max_value': center_max_value,
            'bin_center_ratios': bin_center_ratios,
        }
        if cls_thresholds is not None:
            preprocess_kwargs['cls_thresholds'] = cls_thresholds
        dataset = build_hybrid_dataset(
            x_tensor,
            y_tensor,
            mask=mask_tensor,
            **preprocess_kwargs
        )
    else:
        dataset = TensorDataset(x_tensor, y_tensor)

    # simple statistics of output data
    y_data_mean = np.mean(y_data, 0)
    y_data_var = np.sum((y_data - y_data_mean) ** 2)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var
    stats['hybrid_targets_prepared'] = bool(prepare_hybrid_targets)
    stats['mask_shape'] = tuple(mask_tensor.shape[2:])

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
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuff, **kwargs)
        test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)
        print("split into train({}) / test({}) samples at ratio {}".format(n_train, n_test, split_ratio))
        return train_loader, test_loader, stats

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuff, **kwargs)

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
