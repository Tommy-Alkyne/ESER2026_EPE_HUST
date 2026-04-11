import torch
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CFDHDF5Dataset(Dataset):
    def __init__(self, data_dir, min_spatial_size=7):
        self.data_dir = data_dir
        self.min_spatial_size = min_spatial_size
        
        # 初始化时仅读取元数据，不加载张量
        with h5py.File(data_dir, 'r') as f:
            self.input_key = 'sensor_input' if 'sensor_input' in f else 'input'
            self.n_samples = f[self.input_key].shape[0]
            
            # 预计算统计信息 (为了性能，仅使用前1000个样本估算方差)
            sample_size = min(1000, self.n_samples)
            y_subset = f['output'][:sample_size]
            self.y_mean = np.mean(y_subset, axis=0)
            self.y_var = np.sum((y_subset - self.y_mean) ** 2) * (self.n_samples / sample_size)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 运行时按需动态读取
        with h5py.File(self.data_dir, 'r') as f:
            x_data = f[self.input_key][idx]
            y_data = f['output'][idx]
            
        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.float32)

        # 输入维度处理：补齐通道维 (H, W) -> (1, H, W)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)
            
        # 输出维度处理：如果是 (C, 1, H, W) 则挤压掉深度维
        if y_tensor.ndim == 4 and y_tensor.shape[1] == 1:
            y_tensor = y_tensor.squeeze(1)

        # 空间插值放大 (针对输入特征图过小的情况)
        _, h, w = x_tensor.shape
        target_h = max(y_tensor.shape[-2], self.min_spatial_size)
        target_w = max(y_tensor.shape[-1], self.min_spatial_size)
        
        if h < target_h or w < target_w:
            x_tensor = F.interpolate(
                x_tensor.unsqueeze(0), 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)

        return x_tensor, y_tensor

def load_data(data_dir, batch_size):
    """优化后的数据加载器：Lazy Loading + GPU预取支持"""
    dataset = CFDHDF5Dataset(data_dir)
    
    kwargs = {
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 2  # 加速数据向GPU的输送
    } if torch.cuda.is_available() else {}

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
    
    stats = {
        'y_mean': dataset.y_mean,
        'y_var': dataset.y_var
    }
    return data_loader, stats