import torch
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CFDHDF5Dataset(Dataset):
    def __init__(self, data_dir, min_spatial_size=7):
        self.data_dir = data_dir
        self.min_spatial_size = min_spatial_size
        
        with h5py.File(data_dir, 'r') as f:
            self.input_key = 'sensor_input' if 'sensor_input' in f else 'input'
            self.n_samples = f[self.input_key].shape[0]
            
            # 预计算统计信息 (使用子集估算)
            sample_size = min(1000, self.n_samples)
            y_subset = f['output'][:sample_size]
            
            self.y_mean = np.mean(y_subset, axis=0)
            self.y_var = np.sum((y_subset - self.y_mean) ** 2) * (self.n_samples / sample_size)
            
            # ==========================================
            # [新增] 计算各物理通道(P, Vx, Vy)的全局最大最小值
            # y_subset 的 shape 通常为 (N, C, H, W)
            # ==========================================
            self.y_min = np.min(y_subset, axis=(0, 2, 3)) # Shape: (C,)
            self.y_max = np.max(y_subset, axis=(0, 2, 3)) # Shape: (C,)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with h5py.File(self.data_dir, 'r') as f:
            x_data = f[self.input_key][idx]
            y_data = f['output'][idx]
            
        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.float32)

        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)
            
        if y_tensor.ndim == 4 and y_tensor.shape[1] == 1:
            y_tensor = y_tensor.squeeze(1)
            
        # ==========================================
        # [新增] 对 Target 进行严格的 Min-Max 归一化 -> [0, 1]
        # ==========================================
        y_min_t = torch.tensor(self.y_min, dtype=torch.float32).view(-1, 1, 1)
        y_max_t = torch.tensor(self.y_max, dtype=torch.float32).view(-1, 1, 1)
        
        # 归一化公式: (y - min) / (max - min)
        y_tensor = (y_tensor - y_min_t) / (y_max_t - y_min_t + 1e-8)
        # 防止子集采样导致的极微小越界，硬截断到 0-1
        y_tensor = torch.clamp(y_tensor, 0.0, 1.0)

        # 空间插值放大
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
    dataset = CFDHDF5Dataset(data_dir)
    
    kwargs = {
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 2 
    } if torch.cuda.is_available() else {}

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
    
    # 导出极值字典供物理损失层使用
    stats = {
        'y_mean': dataset.y_mean,
        'y_var': dataset.y_var,
        'y_min': dataset.y_min,  # [新增]
        'y_max': dataset.y_max   # [新增]
    }
    return data_loader, stats