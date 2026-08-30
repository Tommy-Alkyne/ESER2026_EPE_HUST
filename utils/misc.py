"""杂项工具。

旧版这里堆着上游 SVGD / 贝叶斯版本遗留的死代码：一个全局可变 `logger` 字典、
`log_sum_exp`、`parameters_to_vector` / `vector_to_parameters`（后者还硬依赖
已废弃的 `torch.cuda.FloatTensor`）。全项目没有任何地方引用它们，已删除。
"""

import os

import numpy as np
import torch


def mkdir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def mkdirs(paths):
    if isinstance(paths, (list, tuple)):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    raise TypeError('期望 torch.Tensor 或 np.ndarray，得到 {}'.format(type(x)))


def new_logger():
    """训练日志容器。改成函数返回，避免旧版模块级可变全局被跨实验复用。"""
    return {
        'rmse_train': [],
        'rmse_test': [],
        'r2_train': [],
        'r2_test': [],
        'r2_test_per_channel': [],
        'div_rms': [],
        'ns_rms': [],
        'advdiff_rms': [],      # dispersion 任务的对流扩散残差
        'time': [],
    }


class RunningMetrics:
    """按掩码累计 SSE / SST，支持整体与分通道 R²。

    旧版把三个通道池化成一个 R²，压力场好、速度场差会被平均掩盖。
    """

    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.reset()

    def reset(self):
        z = np.zeros(self.n_channels, dtype=np.float64)
        self.sse = z.copy()
        self.sum_y = z.copy()
        self.sum_y2 = z.copy()
        self.count = z.copy()

    @torch.no_grad()
    def update(self, output, target, mask=None):
        """output / target: (B, C, ...)；mask 可广播。"""
        err2 = (output - target) ** 2
        if mask is None:
            weight = torch.ones_like(target)
        else:
            weight = mask.to(dtype=target.dtype)
            if weight.shape != target.shape:
                weight = weight.expand_as(target)

        dims = [0] + list(range(2, target.dim()))
        self.sse += (err2 * weight).sum(dim=dims).double().cpu().numpy()
        self.sum_y += (target * weight).sum(dim=dims).double().cpu().numpy()
        self.sum_y2 += ((target ** 2) * weight).sum(dim=dims).double().cpu().numpy()
        self.count += weight.sum(dim=dims).double().cpu().numpy()

    def _sst(self):
        n = np.maximum(self.count, 1.0)
        return np.maximum(self.sum_y2 - self.sum_y ** 2 / n, 0.0)

    def r2_per_channel(self):
        sst = self._sst()
        return np.where(sst > 1e-12, 1.0 - self.sse / np.maximum(sst, 1e-12), np.nan)

    def r2(self):
        """全局 R²（各通道 SSE / SST 汇总）。"""
        sst = self._sst().sum()
        return float(1.0 - self.sse.sum() / max(sst, 1e-12))

    def rmse(self):
        return float(np.sqrt(self.sse.sum() / max(self.count.sum(), 1.0)))

    def rmse_per_channel(self):
        return np.sqrt(self.sse / np.maximum(self.count, 1.0))
