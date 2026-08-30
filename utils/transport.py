"""对流扩散（污染物输运）残差。

方案文档 §3.2 / §5。求解的方程是

    ∂C/∂t + u·∂C/∂x + v·∂C/∂y = D·∇²C + S

各项说明
--------
* **速度场 (u, v) 来自网络输入，不是网络输出**。两阶段架构里阶段二的输出只有
  浓度 C 一个通道，流场是阶段一的产物、作为输入通道喂进来。所以本模块的
  `fields` 字典要由调用方组装，而不是像流场那样从 output 里按通道切。
* **`dCdt=None` 时走稳态**。这是方案文档里瞬态支持的唯一接口：将来做预报时，
  由调用方在 t 和 t+Δt 各前向一次做中心差分，把结果传进来，方程本身一行不用改。
  ∂C/∂t 必须用**网络输出**算——用真值差分那是监督信号，不是物理约束。
* **源项 S 默认为 0**，源点格点应当由调用方从掩码里排除，而不是在这里建模。

数值精度
--------
空间导数用 `physics_loss` 里现成的二阶中心差分算子，因此残差有 O(h²) 截断误差，
在解析解上不会精确为零。`tests/test_transport.py` 验证的是**收敛阶**：网格加密
一倍，残差应下降约 4 倍。这比"残差很小"是强得多的证据。
"""

from .physics_loss import grad_xy, laplacian, masked_mean


def advection_diffusion_residual(c, vx, vy, diffusivity=1e-5, dx=1.0, dy=1.0,
                                 dcdt=None, source=None):
    """返回方程残差 ∂C/∂t + u·∇C - D·∇²C - S（理想情况下处处为 0）。

    Args:
        c: (B, 1, H, W) 有量纲浓度场
        vx / vy: (B, 1, H, W) 有量纲速度场，形状需与 c 可广播
        diffusivity: 扩散系数 D
        dx / dy: 真实网格间距
        dcdt: (B, 1, H, W) 时间导数；None 表示稳态
        source: (B, 1, H, W) 源项 S；None 表示无源
    """
    dc_dx, dc_dy = grad_xy(c, dx, dy)
    residual = vx * dc_dx + vy * dc_dy - diffusivity * laplacian(c, dx, dy)
    if dcdt is not None:
        residual = residual + dcdt
    if source is not None:
        residual = residual - source
    return residual


def advection_diffusion_loss(fields, mask=None, diffusivity=1e-5, dx=1.0, dy=1.0,
                             dcdt=None, source=None):
    """对流扩散残差的掩码内均方。缺少任一必需场时返回 None。

    Args:
        fields: 至少含 'c' / 'vx' / 'vy' 三个键的字典，值为 (B, 1, H, W)
    """
    c, vx, vy = fields.get('c'), fields.get('vx'), fields.get('vy')
    if c is None or vx is None or vy is None:
        return None
    residual = advection_diffusion_residual(
        c, vx, vy, diffusivity=diffusivity, dx=dx, dy=dy,
        dcdt=dcdt, source=source)
    return masked_mean(residual ** 2, mask)


def central_time_derivative(c_prev, c_next, dt):
    """中心差分求 ∂C/∂t。

    两个入参都应当是**网络输出**（在 t-Δt 与 t+Δt 各前向一次），dt 是 Δt 而非
    2Δt —— 中心差分的分母是 2Δt，这里显式写出来避免调用方传错。
    """
    if dt <= 0:
        raise ValueError('dt 必须为正，当前为 {}'.format(dt))
    return (c_next - c_prev) / (2.0 * dt)
