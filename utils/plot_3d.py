"""3D 绘图入口。

旧版 `plot.py` 与 `plot_3d.py` 是两份高度重复、且只修了一份 bug 的代码
（matplotlib 3.8+ 的 `.collections` 兼容只在 3D 版修过）。现在 3D 版直接复用
`utils/plot.py` 的实现 —— 那边已经通过 `_select_middle_depth` 支持
(C, D, H, W)，行为完全一致。
"""

from .plot import (CHANNEL_LABELS, plot_prediction, save_stats)  # noqa: F401

__all__ = ['plot_prediction', 'save_stats', 'CHANNEL_LABELS']
