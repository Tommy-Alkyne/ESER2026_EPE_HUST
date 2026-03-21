import os
import numpy as np
import h5py


def resolve_data_paths(args):
    """Resolve train/test HDF5 paths from args for 3D workflow."""
    if args.train_file:
        if os.path.isabs(args.train_file) or os.path.dirname(args.train_file):
            train_data_dir = args.train_file
        else:
            train_data_dir = os.path.join(args.data_dir, args.train_file)
    else:
        train_data_dir = os.path.join(args.data_dir, '10243d.hdf5')

    if args.test_file:
        if os.path.isabs(args.test_file) or os.path.dirname(args.test_file):
            test_data_dir = args.test_file
        else:
            test_data_dir = os.path.join(args.data_dir, args.test_file)
    else:
        test_data_dir = train_data_dir

    return train_data_dir, test_data_dir


def infer_io_shapes_from_hdf5(data_path, args):
    """Infer args.nic, args.noc and args.image_size from HDF5 datasets."""
    try:
        with h5py.File(data_path, 'r') as f:
            y_shape = f['output'].shape
            if len(y_shape) >= 5:
                args.noc = y_shape[1]
                args.image_size = list(y_shape[-3:])
            elif len(y_shape) == 4:
                args.noc = y_shape[1]
                args.image_size = [1] + list(y_shape[-2:])

            if 'spatial_input' in f:
                spatial_shape = f['spatial_input'].shape
                args.nic = spatial_shape[1]
                if 'input' in f:
                    args.nic += f['input'].shape[1]
                if 'sensor_input' in f:
                    args.nic += f['sensor_input'].shape[1]
            elif 'input' in f:
                x_shape = f['input'].shape
                if len(x_shape) in (4, 5):
                    args.nic = x_shape[1]
                else:
                    args.nic = int(np.prod(x_shape[1:]))
                if 'sensor_input' in f:
                    args.nic += int(np.prod(f['sensor_input'].shape[1:]))
            elif 'sensor_input' in f:
                x_shape = f['sensor_input'].shape
                if len(x_shape) in (4, 5):
                    args.nic = x_shape[1]
                else:
                    args.nic = int(np.prod(x_shape[1:]))
            else:
                args.nic = 1
    except Exception as error:
        print('Warning: failed to infer 3D data shapes from {}: {}'.format(data_path, error))

    return args
