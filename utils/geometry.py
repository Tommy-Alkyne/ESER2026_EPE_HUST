"""T 型管计算域几何定义（全项目唯一真源）。

原实现把掩码写死成 137x137 网格上的像素坐标（136/35/51/87/137），换分辨率
时用 min() 静默夹取，导致掩码几何完全错位却不报错：

    65x65   -> 覆盖 63.8%
    137x137 -> 覆盖 44.9%  (设计值)
    257x257 -> 覆盖 12.8%

这里改成用**归一化比例**描述几何，任意分辨率都得到相同的物理区域。
默认比例由原 137x137 硬编码坐标反推得到，因此在 137x137 上与旧版逐像素等价。
"""

import json

import numpy as np

# 每段管道用 (y0, y1, x0, x1) 的归一化比例描述，取值范围 [0, 1]
# y 沿 H（行）方向，x 沿 W（列）方向，右开区间。
DEFAULT_T_GEOMETRY = {
    # 竖管：原 rows 0:136, cols 0:35   @137
    'vertical': (0.0, 136.0 / 137.0, 0.0, 35.0 / 137.0),
    # 横管：原 rows 51:87, cols 35:137 @137
    'horizontal': (51.0 / 137.0, 87.0 / 137.0, 35.0 / 137.0, 1.0),
}


def parse_geometry(spec):
    """把 --t-geometry 的取值解析成几何字典。

    Args:
        spec: None / dict / JSON 字符串 / 指向 JSON 文件的路径
    """
    if spec is None or spec == '':
        return dict(DEFAULT_T_GEOMETRY)
    if isinstance(spec, dict):
        return {k: tuple(float(v) for v in vals) for k, vals in spec.items()}
    if isinstance(spec, str):
        text = spec.strip()
        if not text.startswith('{'):
            with open(text, 'r', encoding='utf-8') as fh:
                text = fh.read()
        raw = json.loads(text)
        return {k: tuple(float(v) for v in vals) for k, vals in raw.items()}
    raise TypeError('无法解析的几何定义: {!r}'.format(spec))


def validate_geometry(geometry):
    for name, box in geometry.items():
        if len(box) != 4:
            raise ValueError('几何段 {} 必须是 (y0, y1, x0, x1) 四元组'.format(name))
        y0, y1, x0, x1 = box
        for v in box:
            if not (0.0 <= v <= 1.0):
                raise ValueError('几何段 {} 的比例 {} 超出 [0, 1]'.format(name, v))
        if y1 <= y0 or x1 <= x0:
            raise ValueError('几何段 {} 的区间为空: {}'.format(name, box))
    return geometry


def t_mask_numpy(h, w, geometry=None, dtype=np.float32):
    """生成 (H, W) 的 0/1 掩码。任意分辨率下覆盖同一物理区域。"""
    geometry = validate_geometry(parse_geometry(geometry))
    mask = np.zeros((h, w), dtype=dtype)
    for y0f, y1f, x0f, x1f in geometry.values():
        y0, y1 = int(round(y0f * h)), int(round(y1f * h))
        x0, x1 = int(round(x0f * w)), int(round(x1f * w))
        y0, x0 = max(y0, 0), max(x0, 0)
        y1, x1 = min(y1, h), min(x1, w)
        if y1 > y0 and x1 > x0:
            mask[y0:y1, x0:x1] = 1
    if mask.sum() == 0:
        raise ValueError(
            '在 {}x{} 网格上生成的 T 型掩码为空，请检查 --t-geometry'.format(h, w))
    return mask


def t_mask_like(tensor, geometry=None):
    """按 `tensor` 的形状/设备/精度生成可广播的掩码。

    支持 4D (B, C, H, W) 与 5D (B, C, D, H, W)，返回形状分别为
    (1, 1, H, W) / (1, 1, 1, H, W)，交给广播去对齐 batch 与通道。
    """
    import torch  # 局部导入，保持本模块可被无 torch 环境使用

    h, w = tensor.shape[-2:]
    mask = torch.as_tensor(
        t_mask_numpy(h, w, geometry), device=tensor.device, dtype=tensor.dtype)
    if tensor.dim() == 4:
        return mask.view(1, 1, h, w)
    if tensor.dim() == 5:
        return mask.view(1, 1, 1, h, w)
    raise ValueError('不支持的张量维度 {}'.format(tensor.dim()))


def mask_coverage(h, w, geometry=None):
    """返回掩码覆盖率，用于启动时打印自检信息。"""
    mask = t_mask_numpy(h, w, geometry)
    return float(mask.mean())
