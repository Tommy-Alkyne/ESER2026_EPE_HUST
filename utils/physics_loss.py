"""物理约束损失（PINN）。

相对旧版的关键修正
------------------
1. **通道顺序显式化**：不再靠注释约定，由 `channel_order` 参数决定哪一个通道
   是 P / Vx / Vy。旧版 `physics_loss` 假设 [Vx, Vy, P]，而 `load_data` 的注释
   与 README 写的是 (P, Vx, Vy) —— 二者只要不一致，散度约束实际在算
   ``dP/dx + dVx/dy`` 这种无意义的量，且不会报任何错。
2. **在有量纲的物理场上求残差**：旧版直接对逐通道 min-max 归一化后的场求导，
   ``div(v_norm)=0`` 只有在 Vx / Vy 动态范围恰好相等时才等价于 ``div(v)=0``。
   现在损失内部先用 (y_min, y_max) 反归一化，再用真实网格间距 dx/dy 求残差。
3. **5D 张量重排修正**：旧版 ``x.reshape(B*D, C, H, W)`` 把通道维和深度维搅乱了
   （实测 C=3/D=2 时得到 [1,1,2] 而非 [1,2,3]）。现在先 permute 再 reshape。
4. **全部损失项都受掩码约束，且分母是掩码内像素数**：旧版散度/平滑损失完全没有
   掩码，乘了掩码的项分母仍是全图像素数（137 网格下约 55% 是零），损失值被稀释
   近一半，打印出来的数值也互不可比。
5. **边界用 replicate padding**：旧版 zero padding 等价于把管外当成 0 值流场，
   在边界上引入虚假的巨大梯度。
"""

import torch
import torch.nn.functional as F

# 通道顺序代号 -> {物理量: 通道下标}
CHANNEL_ORDERS = {
    'pvv': {'p': 0, 'vx': 1, 'vy': 2},   # (P, Vx, Vy)  —— README / load_data 注释
    'vvp': {'vx': 0, 'vy': 1, 'p': 2},   # (Vx, Vy, P)  —— 旧版 physics_loss 的隐式假设
}


def resolve_channels(channel_order, n_channels):
    """把通道顺序代号解析成下标字典，并剔除超出实际通道数的项。"""
    if isinstance(channel_order, dict):
        idx = dict(channel_order)
    else:
        key = str(channel_order).lower()
        if key not in CHANNEL_ORDERS:
            raise ValueError('未知的 channel_order={!r}，可选 {}'.format(
                channel_order, sorted(CHANNEL_ORDERS)))
        idx = dict(CHANNEL_ORDERS[key])
    return {k: v for k, v in idx.items() if v < n_channels}


# ---------------------------------------------------------------- 张量重排
def flatten_depth(x):
    """(B, C, D, H, W) -> (B*D, C, H, W)，保持通道语义不变。"""
    b, c, d, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w), (b, d)


def unflatten_depth(x, b, d):
    """(B*D, C, H, W) -> (B, C, D, H, W)"""
    _, c, h, w = x.shape
    return x.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4)


def as_4d(x):
    """把 4D/5D 统一成 4D，返回 (tensor4d, restore_info)。"""
    if x.dim() == 5:
        return flatten_depth(x)
    if x.dim() == 4:
        return x, None
    raise ValueError('不支持的张量维度 {}'.format(x.dim()))


# ---------------------------------------------------------------- 差分算子
def _diff_kernels(device, dtype):
    kx = torch.tensor([[0., 0., 0.], [-0.5, 0., 0.5], [0., 0., 0.]],
                      dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[0., -0.5, 0.], [0., 0., 0.], [0., 0.5, 0.]],
                      dtype=dtype, device=device).view(1, 1, 3, 3)
    return kx, ky


def _conv_replicate(x, kernel):
    """replicate padding + 无 padding 卷积，避免管外零值造成的虚假边界梯度。"""
    channels = x.shape[1]
    padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
    if channels > 1:
        return F.conv2d(padded, kernel.repeat(channels, 1, 1, 1), groups=channels)
    return F.conv2d(padded, kernel)


def grad_xy(field, dx=1.0, dy=1.0):
    """一阶中心差分，返回 (d/dx, d/dy)。"""
    kx, ky = _diff_kernels(field.device, field.dtype)
    return _conv_replicate(field, kx) / dx, _conv_replicate(field, ky) / dy


def laplacian(field, dx=1.0, dy=1.0):
    """五点 Laplacian。各向异性网格下 dx^2 / dy^2 分别缩放。"""
    kxx = torch.tensor([[0., 0., 0.], [1., -2., 1.], [0., 0., 0.]],
                       dtype=field.dtype, device=field.device).view(1, 1, 3, 3)
    kyy = torch.tensor([[0., 1., 0.], [0., -2., 0.], [0., 1., 0.]],
                       dtype=field.dtype, device=field.device).view(1, 1, 3, 3)
    return (_conv_replicate(field, kxx) / (dx * dx)
            + _conv_replicate(field, kyy) / (dy * dy))


# ---------------------------------------------------------------- 掩码统计
def masked_mean(x, mask=None):
    """掩码内均值。mask 为 None 时退化成普通 mean。"""
    if mask is None:
        return x.mean()
    mask = mask.to(dtype=x.dtype)
    if mask.shape != x.shape:
        mask = mask.expand_as(x)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def _single_channel_mask(mask):
    """把 (B, C, ...) 的掩码压成 (B, 1, ...)，逐物理场计算时用。"""
    if mask is None:
        return None
    if mask.shape[1] == 1:
        return mask
    return mask[:, :1]


def erode_mask(mask, radius=1):
    """把掩码向内收缩 radius 个格点。

    3x3 差分模板在贴壁的格点上会取到墙外的值，如果直接在原掩码上评估物理残差，
    管外的数值就会从边界"漏"进损失里（实测能造成 50% 以上的偏差）。收缩一格后，
    每个评估点的整个模板都落在计算域内部，残差才是纯粹的域内物理量。
    """
    if mask is None or radius <= 0:
        return mask
    kernel = 2 * radius + 1
    inv = 1.0 - mask.to(dtype=torch.float32)
    inv = F.max_pool2d(inv, kernel_size=kernel, stride=1, padding=radius)
    return (1.0 - inv).to(dtype=mask.dtype)


# ---------------------------------------------------------------- 反归一化
def denormalize(x, y_min, y_max):
    """把 [0, 1] 归一化场还原成有量纲物理场。y_min/y_max 形状 (C,)。"""
    if y_min is None or y_max is None:
        return x
    shape = [1, -1] + [1] * (x.dim() - 2)
    lo = torch.as_tensor(y_min, dtype=x.dtype, device=x.device).view(*shape)
    hi = torch.as_tensor(y_max, dtype=x.dtype, device=x.device).view(*shape)
    return x * (hi - lo) + lo


# ---------------------------------------------------------------- 各项损失
def divergence(vx, vy, dx=1.0, dy=1.0):
    """连续性方程左端 div(v) = dvx/dx + dvy/dy"""
    dvx_dx, _ = grad_xy(vx, dx, dy)
    _, dvy_dy = grad_xy(vy, dx, dy)
    return dvx_dx + dvy_dy


def divergence_loss(fields, mask=None, dx=1.0, dy=1.0):
    """不可压缩约束：min ||div(v)||^2（仅在计算域内）"""
    vx, vy = fields.get('vx'), fields.get('vy')
    if vx is None or vy is None:
        return None
    return masked_mean(divergence(vx, vy, dx, dy) ** 2, mask)


def navier_stokes_loss(fields, mask=None, rho=1.0, nu=1e-2, dx=1.0, dy=1.0):
    """稳态不可压缩 N-S 动量方程残差。

        x: u du/dx + v du/dy + (1/rho) dp/dx - nu * lap(u) = 0
        y: u dv/dx + v dv/dy + (1/rho) dp/dy - nu * lap(v) = 0
    """
    vx, vy, p = fields.get('vx'), fields.get('vy'), fields.get('p')
    if vx is None or vy is None or p is None:
        return None

    dvx_dx, dvx_dy = grad_xy(vx, dx, dy)
    dvy_dx, dvy_dy = grad_xy(vy, dx, dy)
    dp_dx, dp_dy = grad_xy(p, dx, dy)
    lap_vx = laplacian(vx, dx, dy)
    lap_vy = laplacian(vy, dx, dy)

    mom_x = vx * dvx_dx + vy * dvx_dy + dp_dx / rho - nu * lap_vx
    mom_y = vx * dvy_dx + vy * dvy_dy + dp_dy / rho - nu * lap_vy
    return masked_mean(mom_x ** 2, mask) + masked_mean(mom_y ** 2, mask)


def smoothness_loss(tensor, mask=None, dx=1.0, dy=1.0):
    """全场一阶导 L1，抑制棋盘伪影。只统计计算域内部。"""
    kx, ky = _diff_kernels(tensor.device, tensor.dtype)
    gx = _conv_replicate(tensor, kx) / dx
    gy = _conv_replicate(tensor, ky) / dy
    return masked_mean(gx.abs(), mask) + masked_mean(gy.abs(), mask)


def reconstruction_loss(output, target, mask=None, use_weighted=True):
    """数据项。分母是掩码内像素数，不再被管外的零值稀释。"""
    err = output - target
    mse = masked_mean(err ** 2, mask)
    l1 = masked_mean(err.abs(), mask)
    if use_weighted:
        return 0.6 * mse + 0.4 * l1
    return mse + l1


# ---------------------------------------------------------------- 组合损失
def combined_physics_loss(output, target, mask=None, epoch=1, total_epochs=200,
                          div_weight=0.05, ns_weight=0.005, smooth_weight=0.005,
                          channel_order='pvv', y_min=None, y_max=None,
                          rho=1.0, nu=1e-2, dx=1.0, dy=1.0,
                          physics_warmup=0.0, scales=None):
    """数据项 + 物理项的组合损失。

    Args:
        output / target: (B, C, H, W) 或 (B, C, D, H, W) 的**归一化**场
        mask: 与 output 可广播的 0/1 计算域掩码
        channel_order: 'pvv' 或 'vvp'，决定哪个通道是 P/Vx/Vy
        y_min / y_max: 形状 (C,) 的逐通道极值；给了就在物理项里反归一化
        dx / dy: 真实网格间距（domain_size / (N-1)）
        rho / nu: 流体密度与运动粘度，需与数据集工况一致
        physics_warmup: 前多少比例的 epoch 完全不加物理约束（0~1）
        scales: {'div':float,'ns':float,'smooth':float} 归一化因子。物理残差的
            绝对量级随 dx/dy 和工况变化好几个数量级（dx=0.007 时散度残差轻松上
            千），固定权重根本没法跨数据集复用。传入各项自身量级的估计（见
            `PhysicsScaler`）后，权重就变成与量纲无关的相对强度。

    Returns:
        (total_loss, dict) —— dict 内 base/div/ns/smooth 为**未缩放**的原始值
    """
    ch = resolve_channels(channel_order, output.shape[1])

    # 数据项在归一化空间算：各通道等权，压力量级不会压过速度
    base_loss = reconstruction_loss(output, target, mask=mask, use_weighted=True)

    # 物理项在有量纲空间算
    phys = denormalize(output, y_min, y_max)
    phys4d, _ = as_4d(phys)
    mask4d = None
    if mask is not None:
        m = mask if mask.shape == output.shape else mask.expand_as(output)
        m4d, _ = as_4d(m)
        mask4d = _single_channel_mask(m4d)

    fields = {name: phys4d[:, i:i + 1] for name, i in ch.items()}

    # 物理残差只在"模板完全落在域内"的格点上评估
    interior = erode_mask(mask4d, radius=1)

    div_loss = divergence_loss(fields, interior, dx, dy)
    ns_loss = navier_stokes_loss(fields, interior, rho, nu, dx, dy)
    smooth = smoothness_loss(phys4d, interior, dx, dy)

    zero = output.sum() * 0.0            # 保持计算图连通的零张量
    div_loss = zero if div_loss is None else div_loss
    ns_loss = zero if ns_loss is None else ns_loss

    # 动态权重：早期先把数据拟合住，中后期逐步压紧物理约束
    progress = min(max(epoch / max(total_epochs, 1), 0.0), 1.0)
    if progress < physics_warmup:
        ramp = 0.0
    elif physics_warmup >= 1.0:
        ramp = 1.0
    else:
        ramp = (progress - physics_warmup) / max(1.0 - physics_warmup, 1e-8)

    w_div = div_weight * (1.0 + ramp)
    w_ns = ns_weight * (1.0 + ramp)
    w_smooth = smooth_weight * (1.0 - 0.5 * ramp)

    scales = scales or {}
    s_div = max(float(scales.get('div', 1.0)), 1e-12)
    s_ns = max(float(scales.get('ns', 1.0)), 1e-12)
    s_smooth = max(float(scales.get('smooth', 1.0)), 1e-12)

    total = (base_loss
             + w_div * div_loss / s_div
             + w_ns * ns_loss / s_ns
             + w_smooth * smooth / s_smooth)

    return total, {
        'base': float(base_loss.detach()),
        'div': float(div_loss.detach()),
        'ns': float(ns_loss.detach()),
        'smooth': float(smooth.detach()),
        'total': float(total.detach()),
        'w_div': w_div,
        'w_ns': w_ns,
        'w_smooth': w_smooth,
        'scale_div': s_div,
        'scale_ns': s_ns,
        'scale_smooth': s_smooth,
    }


class PhysicsScaler:
    """各物理项量级的指数滑动估计，让 --div-weight 之类的权重与量纲解耦。

    不加这层的话，同一组权重换个 domain_size 或分辨率就会从"几乎不起作用"变成
    "压垮数据项"，因为残差量级随 1/dx、1/dx^2 变化。
    """

    def __init__(self, momentum=0.99, enabled=True, keys=('div', 'ns', 'smooth')):
        self.momentum = momentum
        self.enabled = enabled
        self.keys = tuple(keys)
        self.state = {}

    def current(self):
        if not self.enabled:
            return {}
        return {k: v for k, v in self.state.items()}

    def update(self, parts):
        if not self.enabled:
            return
        for key in self.keys:
            value = float(parts.get(key, 0.0))
            if not np_isfinite(value) or value <= 0:
                continue
            if key not in self.state:
                self.state[key] = value
            else:
                m = self.momentum
                self.state[key] = m * self.state[key] + (1.0 - m) * value


def np_isfinite(value):
    return value == value and value not in (float('inf'), float('-inf'))


def physics_report(output, mask=None, channel_order='pvv', y_min=None, y_max=None,
                   rho=1.0, nu=1e-2, dx=1.0, dy=1.0):
    """诊断用：返回有量纲的散度 / 动量残差 RMS，判断物理约束是否真的生效。"""
    with torch.no_grad():
        ch = resolve_channels(channel_order, output.shape[1])
        phys4d, _ = as_4d(denormalize(output, y_min, y_max))
        mask4d = None
        if mask is not None:
            m = mask if mask.shape == output.shape else mask.expand_as(output)
            m4d, _ = as_4d(m)
            mask4d = _single_channel_mask(m4d)
        fields = {name: phys4d[:, i:i + 1] for name, i in ch.items()}
        interior = erode_mask(mask4d, radius=1)
        d = divergence_loss(fields, interior, dx, dy)
        n = navier_stokes_loss(fields, interior, rho, nu, dx, dy)
        return {
            'div_rms': float(d.sqrt()) if d is not None else float('nan'),
            'ns_rms': float(n.sqrt()) if n is not None else float('nan'),
        }
