//! 对流扩散（污染物输运）残差。移植自 `utils/transport.py`。
//!
//! 求解的方程是
//!
//! ```text
//! ∂C/∂t + u·∂C/∂x + v·∂C/∂y = D·∇²C + S
//! ```
//!
//! * **速度场 (u, v) 来自网络输入，不是网络输出**。两阶段架构里阶段二的输出只有
//!   浓度 C 一个通道，流场是阶段一的产物、作为输入通道喂进来。所以 [`Fields`]
//!   要由调用方组装，而不是像流场那样从 output 里按通道切。
//! * **`dcdt = None` 时走稳态**。这是瞬态支持的唯一接口：将来做预报时，由调用方
//!   在 t-Δt 和 t+Δt 各前向一次做中心差分，把结果传进来，方程本身一行不用改。
//!   ∂C/∂t 必须用**网络输出**算——用真值差分那是监督信号，不是物理约束。
//! * **源项 S 默认为 0**，源点格点应当由调用方从掩码里排除，而不是在这里建模。
//!
//! # 数值精度
//!
//! 空间导数用 [`crate::ops`] 里的二阶中心差分，因此残差有 O(h²) 截断误差，在解析
//! 解上不会精确为零。`tests/transport_analytic.rs` 验证的是**收敛阶**：网格加密
//! 一倍，残差应下降约 4 倍。这比「残差很小」是强得多的证据——把 D 写成 0、把速度
//! 乘 0.001，残差也会很小。

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::ops::{grad_xy, laplacian, masked_mean};
use crate::physics::Fields;

/// 方程残差 `∂C/∂t + u·∇C - D·∇²C - S`（理想情况下处处为 0）。
///
/// 参数与 Python 版一一对应，硬拆成结构体只会让两边的对照变难。
#[allow(clippy::too_many_arguments)]
pub fn advection_diffusion_residual<B: Backend>(
    c: Tensor<B, 4>,
    vx: Tensor<B, 4>,
    vy: Tensor<B, 4>,
    diffusivity: f64,
    dx: f64,
    dy: f64,
    dcdt: Option<Tensor<B, 4>>,
    source: Option<Tensor<B, 4>>,
) -> Tensor<B, 4> {
    let (dc_dx, dc_dy) = grad_xy(c.clone(), dx, dy);
    let mut residual = vx * dc_dx + vy * dc_dy - laplacian(c, dx, dy).mul_scalar(diffusivity);
    if let Some(dcdt) = dcdt {
        residual = residual + dcdt;
    }
    if let Some(source) = source {
        residual = residual - source;
    }
    residual
}

/// 对流扩散残差的掩码内均方。缺 `c` / `vx` / `vy` 任一场时返回 `None`——**宁可
/// 不算，也不能拿别的场顶上去静默算错**。
#[allow(clippy::too_many_arguments)]
pub fn advection_diffusion_loss<B: Backend>(
    fields: &Fields<B>,
    mask: Option<Tensor<B, 4>>,
    diffusivity: f64,
    dx: f64,
    dy: f64,
    dcdt: Option<Tensor<B, 4>>,
    source: Option<Tensor<B, 4>>,
    z: Option<crate::physics::ZAxis>,
) -> Option<Tensor<B, 1>> {
    let (c, vx, vy) = (fields.c.clone()?, fields.vx.clone()?, fields.vy.clone()?);
    let mut r = advection_diffusion_residual(c.clone(), vx, vy, diffusivity, dx, dy, dcdt, source);
    // 三维补上竖向对流 w·∂C/∂z 与竖向扩散 D·∂²C/∂z²。
    // 地铁站的烟气就是靠楼梯竖向串层的，漏了这两项等于假设各层互不相通
    if let (Some(z), Some(vz)) = (z, fields.vz.clone()) {
        r = r + vz * crate::physics::dz_of(c.clone(), z, false)
            - crate::physics::dz_of(c, z, true).mul_scalar(diffusivity);
    }
    Some(masked_mean(r.clone() * r, mask))
}

/// 中心差分求 `∂C/∂t`。
///
/// 两个入参都应当是**网络输出**（在 t-Δt 与 t+Δt 各前向一次）。`dt` 是 Δt 而非
/// 2Δt——中心差分的分母是 2Δt，这里显式写出来避免调用方传错。
pub fn central_time_derivative<B: Backend>(
    c_prev: Tensor<B, 4>,
    c_next: Tensor<B, 4>,
    dt: f64,
) -> Result<Tensor<B, 4>, String> {
    if dt <= 0.0 {
        return Err(format!("dt 必须为正，当前为 {}", dt));
    }
    Ok((c_next - c_prev).div_scalar(2.0 * dt))
}
