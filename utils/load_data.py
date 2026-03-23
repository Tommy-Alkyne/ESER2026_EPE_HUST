import torch
import numpy as np
import h5py
import torch.nn.functional as F
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
        # 优先使用sensor_input（通常是更好的输入），如果没有则用input
        if 'sensor_input' in f:
            x_data = f['sensor_input'][()]
            print("[INFO] Using 'sensor_input' as input (shape: {})".format(f['sensor_input'].shape))
        else:
            x_data = f['input'][()]
            print("[INFO] Using 'input' as input (shape: {})".format(f['input'].shape))
        
        y_data = f['output'][()]

    print("output data shape (原始): {}".format(y_data.shape))
    print("input data shape (原始): {}".format(x_data.shape))
    
    # 处理输入数据的形状
    # 如果输入是2D (N, features)，需要重塑为4D (N, C, H, W)
    if x_data.ndim == 2:
        # 对于(N, features)，尝试重塑为合理的(N, C, H, W)
        n_samples = x_data.shape[0]
        features = x_data.shape[1]
        # 将features分解为H和W（优先找接近的平方根）
        h = int(np.sqrt(features))
        w = features // h
        if h * w == features:
            x_data = x_data.reshape(n_samples, 1, h, w)
            print(f"[INFO] Reshaped input from (N, {features}) to {x_data.shape}")
        else:
            # 无法完美分解，尝试(N, 1, 1, features)
            x_data = x_data.reshape(n_samples, 1, 1, features)
            print(f"[INFO] Reshaped input to (N, 1, 1, features): {x_data.shape}")
    elif x_data.ndim == 3:
        # (N, H, W) -> (N, 1, H, W)
        # sensor_input 通常是(N, H, W)，需要补通道维
        n_samples, h, w = x_data.shape
        x_data = x_data.reshape(n_samples, 1, h, w)
        print(f"[INFO] Expanded input from (N, H, W) to (N, C, H, W): {x_data.shape}")
    
    # 处理输出数据的形状
    # 如果输出是5D (N, C, 1, H, W) 且深度维为1，挤压掉深度维变成4D
    if y_data.ndim == 5 and y_data.shape[2] == 1:
        y_data = np.squeeze(y_data, axis=2)
        print(f"[INFO] Squeezed output from 5D to 4D (removed depth dim): {y_data.shape}")
    
    output_h, output_w = y_data.shape[-2:]
    print(f"[INFO] Target output spatial size: {output_h}x{output_w}")
    
    # 如果输入空间大小太小，通过双线性插值放大到匹配输出大小或至少7x7
    min_spatial_size = 7  # 模型第一层需要至少7x7
    if x_data.ndim == 4:
        _, c, h, w = x_data.shape
        target_h = max(output_h, min_spatial_size)
        target_w = max(output_w, min_spatial_size)
        
        if h < target_h or w < target_w:
            x_tensor = torch.tensor(x_data, dtype=torch.float32)
            x_upsampled = F.interpolate(
                x_tensor, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
            x_data = x_upsampled.numpy()
            print(f"[INFO] Upsampled input from ({h}x{w}) to ({target_h}x{target_w})")

    print(f"input data shape (处理后): {x_data.shape}")
    print(f"output data shape (处理后): {y_data.shape}")

    kwargs = {'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

    # 转换为torch张量并转换数据类型为float32
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    
    dataset = TensorDataset(x_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # simple statistics of output data
    y_data_mean = np.mean(y_data, 0)
    y_data_var = np.sum((y_data - y_data_mean) ** 2)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

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