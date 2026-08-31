//! 物理约束损失（PINN）：散度 / 动量 / 平滑 / 重建 / 组合损失 / 量级归一化。
//!
//! 移植自 `utils/physics_loss.py`。这一层的每个标量都必须对上
//! `tests/golden/golden.npz` 的 `phys/*` 与 `trans/*`——物理残差写错不会报错，
//! 只会让 R² 差一点点，然后被当成超参没调好。
//!
//! Python 版相对旧版修过的几个坑，这里一并继承（改错了黄金对照会立刻抓到）：
//!
//! 1. **通道顺序显式化**：由 [`ChannelOrder`] 决定哪个通道是 P / Vx / Vy，
//!    不靠注释约定。顺序错了，散度约束实际在算 `dP/dx + dVx/dy` 这种无意义的量。
//! 2. **物理项在有量纲场上算**：先用 (y_min, y_max) 反归一化再求残差。逐通道
//!    归一化后 `div(v_norm)=0` 只有在 Vx / Vy 动态范围恰好相等时才等价于
//!    `div(v)=0`。
//! 3. **所有损失项都受掩码约束，分母是掩码内像素数**，不被域外的零值稀释。
//! 4. **残差只在腐蚀后的内部域上评估**，否则墙外的值会顺着差分模板漏进损失。

use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor, TensorData};

use crate::ops::{erode_mask, grad_xy, laplacian, masked_mean};
use crate::transport::advection_diffusion_loss;

// ---------------------------------------------------------------- 通道顺序

/// 输出通道的物理含义。对应 Python 的 `CHANNEL_ORDERS`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChannelOrder {
    /// (P, Vx, Vy) —— README / `load_data` 注释的口径
    #[default]
    Pvv,
    /// (Vx, Vy, P) —— 旧版 `physics_loss` 的隐式假设
    Vvp,
    /// (P, Vx, Vy, T)。T 只作为输出学习，**不参与任何残差**
    Pvvt,
    /// 只有浓度一个通道，速度场从输入通道喂进来（地铁站方案阶段二）
    C,
}

impl ChannelOrder {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "pvv" => Ok(Self::Pvv),
            "vvp" => Ok(Self::Vvp),
            "pvvt" => Ok(Self::Pvvt),
            "c" => Ok(Self::C),
            other => Err(format!(
                "未知的 channel_order={:?}，可选 c / pvv / pvvt / vvp",
                other
            )),
        }
    }

    /// `(p, vx, vy, c)` 各自的通道下标；`None` 表示这个顺序里没有该物理量。
    pub fn indices(self) -> [Option<usize>; 4] {
        match self {
            Self::Pvv | Self::Pvvt => [Some(0), Some(1), Some(2), None],
            Self::Vvp => [Some(2), Some(0), Some(1), None],
            Self::C => [None, None, None, Some(0)],
        }
    }
}

/// 按物理量拆开的场。缺哪个量对应的损失就返回 `None`——**宁可不算，也不能拿
/// 别的通道顶上去静默算错**。
#[derive(Debug, Clone, Default)]
pub struct Fields<B: Backend> {
    pub p: Option<Tensor<B, 4>>,
    pub vx: Option<Tensor<B, 4>>,
    pub vy: Option<Tensor<B, 4>>,
    pub c: Option<Tensor<B, 4>>,
}

/// 把 (B, C, H, W) 按通道顺序拆成各物理场。
///
/// 通道数不足时自动降级（例如 `pvv` 只喂了 2 个通道就只有 P 和 Vx），与 Python
/// 的 `resolve_channels` 一致。
pub fn split_fields<B: Backend>(x: &Tensor<B, 4>, order: ChannelOrder) -> Fields<B> {
    let [b, nc, h, w] = x.dims();
    let take = |i: Option<usize>| -> Option<Tensor<B, 4>> {
        let i = i?;
        (i < nc).then(|| x.clone().slice([0..b, i..i + 1, 0..h, 0..w]))
    };
    let [p, vx, vy, c] = order.indices();
    Fields {
        p: take(p),
        vx: take(vx),
        vy: take(vy),
        c: take(c),
    }
}

// ---------------------------------------------------------------- 张量重排

/// (B, C, D, H, W) -> (B*D, C, H, W)，返回 `(x4d, b, d)`。
///
/// **必须先换轴再 reshape**：旧版直接 `reshape(B*D, C, H, W)` 把通道维和深度维
/// 搅乱了（实测 C=3/D=2 时得到 [1,1,2] 而不是 [1,2,3]），物理量对不上号却不报错。
pub fn flatten_depth<B: Backend>(x: Tensor<B, 5>) -> (Tensor<B, 4>, usize, usize) {
    let [b, c, d, h, w] = x.dims();
    (x.swap_dims(1, 2).reshape([b * d, c, h, w]), b, d)
}

/// [`flatten_depth`] 的逆变换。
pub fn unflatten_depth<B: Backend>(x: Tensor<B, 4>, b: usize, d: usize) -> Tensor<B, 5> {
    let [_, c, h, w] = x.dims();
    x.reshape([b, d, c, h, w]).swap_dims(1, 2)
}

// ---------------------------------------------------------------- 反归一化

/// 把 [0, 1] 归一化场还原成有量纲物理场。`y_min` / `y_max` 长度须等于通道数。
///
/// 任一为 `None` 就原样返回——物理项此时在归一化空间算，只在两个速度分量动态
/// 范围恰好相等时才等价，调用方要清楚自己在做什么。
pub fn denormalize<B: Backend>(
    x: Tensor<B, 4>,
    y_min: Option<&[f32]>,
    y_max: Option<&[f32]>,
) -> Tensor<B, 4> {
    let (Some(lo), Some(hi)) = (y_min, y_max) else {
        return x;
    };
    let c = x.dims()[1];
    assert_eq!(lo.len(), c, "y_min 长度 {} 与通道数 {} 不符", lo.len(), c);
    assert_eq!(hi.len(), c, "y_max 长度 {} 与通道数 {} 不符", hi.len(), c);
    let device = x.device();
    let to_t =
        |v: &[f32]| Tensor::<B, 1>::from_data(TensorData::from(v), &device).reshape([1, c, 1, 1]);
    let lo_t = to_t(lo);
    x * (to_t(hi) - lo_t.clone()) + lo_t
}

/// 把 (B, C, H, W) 的掩码压成单通道——逐物理场求残差时只需要一份几何。
fn single_channel<B: Backend>(m: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, c, h, w] = m.dims();
    if c == 1 {
        m
    } else {
        m.slice([0..b, 0..1, 0..h, 0..w])
    }
}

// ---------------------------------------------------------------- 各项损失

/// 连续性方程左端 `div(v) = dvx/dx + dvy/dy`。
pub fn divergence<B: Backend>(
    vx: Tensor<B, 4>,
    vy: Tensor<B, 4>,
    dx: f64,
    dy: f64,
) -> Tensor<B, 4> {
    let (dvx_dx, _) = grad_xy(vx, dx, dy);
    let (_, dvy_dy) = grad_xy(vy, dx, dy);
    dvx_dx + dvy_dy
}

/// 不可压缩约束 `min ||div(v)||²`（仅计算域内）。缺速度场时返回 `None`。
pub fn divergence_loss<B: Backend>(
    fields: &Fields<B>,
    mask: Option<Tensor<B, 4>>,
    dx: f64,
    dy: f64,
) -> Option<Tensor<B, 1>> {
    let (vx, vy) = (fields.vx.clone()?, fields.vy.clone()?);
    let d = divergence(vx, vy, dx, dy);
    Some(masked_mean(d.clone() * d, mask))
}

/// 稳态不可压缩 N-S 动量方程残差。
///
/// ```text
/// x: u du/dx + v du/dy + (1/rho) dp/dx - nu * lap(u) = 0
/// y: u dv/dx + v dv/dy + (1/rho) dp/dy - nu * lap(v) = 0
/// ```
pub fn navier_stokes_loss<B: Backend>(
    fields: &Fields<B>,
    mask: Option<Tensor<B, 4>>,
    rho: f64,
    nu: f64,
    dx: f64,
    dy: f64,
) -> Option<Tensor<B, 1>> {
    let (vx, vy, p) = (fields.vx.clone()?, fields.vy.clone()?, fields.p.clone()?);

    let (dvx_dx, dvx_dy) = grad_xy(vx.clone(), dx, dy);
    let (dvy_dx, dvy_dy) = grad_xy(vy.clone(), dx, dy);
    let (dp_dx, dp_dy) = grad_xy(p, dx, dy);
    let lap_vx = laplacian(vx.clone(), dx, dy);
    let lap_vy = laplacian(vy.clone(), dx, dy);

    let mom_x =
        vx.clone() * dvx_dx + vy.clone() * dvx_dy + dp_dx.div_scalar(rho) - lap_vx.mul_scalar(nu);
    let mom_y = vx * dvy_dx + vy * dvy_dy + dp_dy.div_scalar(rho) - lap_vy.mul_scalar(nu);

    Some(
        masked_mean(mom_x.clone() * mom_x, mask.clone()) + masked_mean(mom_y.clone() * mom_y, mask),
    )
}

/// 全场一阶导 L1，抑制棋盘伪影。只统计计算域内部。
pub fn smoothness_loss<B: Backend>(
    tensor: Tensor<B, 4>,
    mask: Option<Tensor<B, 4>>,
    dx: f64,
    dy: f64,
) -> Tensor<B, 1> {
    let (gx, gy) = grad_xy(tensor, dx, dy);
    masked_mean(gx.abs(), mask.clone()) + masked_mean(gy.abs(), mask)
}

/// 数据项。分母是掩码内像素数，不再被域外的零值稀释。
pub fn reconstruction_loss<B: Backend>(
    output: Tensor<B, 4>,
    target: Tensor<B, 4>,
    mask: Option<Tensor<B, 4>>,
    use_weighted: bool,
) -> Tensor<B, 1> {
    let err = output - target;
    let mse = masked_mean(err.clone() * err.clone(), mask.clone());
    let l1 = masked_mean(err.abs(), mask);
    if use_weighted {
        mse.mul_scalar(0.6) + l1.mul_scalar(0.4)
    } else {
        mse + l1
    }
}

// ---------------------------------------------------------------- 组合损失

/// 求解的方程组。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task {
    /// 连续性方程 + N-S 动量
    #[default]
    Flow,
    /// 对流扩散（污染物输运）
    Dispersion,
}

/// 各物理项的量级归一化因子。`None` 表示不缩放（等价于 1.0）。
///
/// 物理残差的绝对量级随 dx/dy 和工况变化好几个数量级（dx=0.007 时散度残差轻松
/// 上千），固定权重根本没法跨数据集复用。除掉各项自身量级后，权重才变成与量纲
/// 无关的相对强度。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Scales {
    pub div: Option<f64>,
    pub ns: Option<f64>,
    pub advdiff: Option<f64>,
    pub smooth: Option<f64>,
}

impl Scales {
    /// 缺省 1.0，并夹到 >= 1e-12 防除零。
    fn or_one(v: Option<f64>) -> f64 {
        v.unwrap_or(1.0).max(1e-12)
    }
}

/// 组合损失的分项明细。`base` / `div` / `ns` / `advdiff` / `smooth` 是**未缩放**
/// 的原始值，缩放因子与权重另行给出，方便日志里判断到底是哪一项在起作用。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossParts {
    pub base: f64,
    pub div: f64,
    pub ns: f64,
    pub advdiff: f64,
    pub smooth: f64,
    pub total: f64,
    pub w_div: f64,
    pub w_ns: f64,
    pub w_advdiff: f64,
    pub w_smooth: f64,
    pub scale_div: f64,
    pub scale_ns: f64,
    pub scale_advdiff: f64,
    pub scale_smooth: f64,
}

/// 诊断用的有量纲物理残差 RMS。没算的项是 `None`（Python 侧是 nan）。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PhysicsReport {
    pub div_rms: Option<f64>,
    pub ns_rms: Option<f64>,
    pub advdiff_rms: Option<f64>,
}

/// 数据项 + 物理项的组合损失。
///
/// 字段与 Python `combined_physics_loss` 的关键字参数一一对应，[`Default`] 给的
/// 也是同一组默认值。Rust 没有关键字参数，用结构体更新语法代替：
///
/// ```
/// use burn::backend::NdArray;
/// use burn::tensor::Tensor;
/// use eser::physics::PhysicsLoss;
///
/// type B = NdArray<f32>;
/// let d = Default::default();
/// let output: Tensor<B, 4> = Tensor::zeros([2, 3, 8, 8], &d);
/// let target: Tensor<B, 4> = Tensor::ones([2, 3, 8, 8], &d);
///
/// let (total, parts) = PhysicsLoss::<B> { dx: 0.1, dy: 0.1, ..Default::default() }
///     .compute(output, target);
/// assert!((parts.base - 1.0).abs() < 1e-6); // 全 0 对全 1 的 MSE
/// # let _ = total;
/// ```
///
/// 5D 场 (B, C, D, H, W) 先用 [`flatten_depth`] 压成 4D 再进来——参与平均的元素
/// 集合没变，损失值与在 5D 上算完全相同。
#[derive(Clone)]
pub struct PhysicsLoss<B: Backend> {
    pub epoch: usize,
    pub total_epochs: usize,
    pub div_weight: f64,
    pub ns_weight: f64,
    pub smooth_weight: f64,
    pub advdiff_weight: f64,
    pub channel_order: ChannelOrder,
    pub task: Task,
    /// 流体密度
    pub rho: f64,
    /// 运动粘度
    pub nu: f64,
    /// 扩散系数 D（dispersion 任务）
    pub diffusivity: f64,
    /// 真实网格间距 domain_size / (N - 1)
    pub dx: f64,
    pub dy: f64,
    /// 前多少比例的 epoch 完全不加物理约束（0~1）
    pub physics_warmup: f64,
    pub scales: Scales,
    /// 与 output 可广播的 0/1 计算域掩码
    pub mask: Option<Tensor<B, 4>>,
    /// 逐通道极值；给了就在物理项里反归一化
    pub y_min: Option<Vec<f32>>,
    pub y_max: Option<Vec<f32>>,
    /// 仅 dispersion 用：**有量纲**速度场，来自网络输入通道而非输出
    pub velocity: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    /// ∂C/∂t；`None` 表示稳态
    pub dcdt: Option<Tensor<B, 4>>,
    /// 源项 S
    pub source: Option<Tensor<B, 4>>,
}

impl<B: Backend> Default for PhysicsLoss<B> {
    fn default() -> Self {
        Self {
            epoch: 1,
            total_epochs: 200,
            div_weight: 0.05,
            ns_weight: 0.005,
            smooth_weight: 0.005,
            advdiff_weight: 0.05,
            channel_order: ChannelOrder::Pvv,
            task: Task::Flow,
            rho: 1.0,
            nu: 1e-2,
            diffusivity: 1e-5,
            dx: 1.0,
            dy: 1.0,
            physics_warmup: 0.0,
            scales: Scales::default(),
            mask: None,
            y_min: None,
            y_max: None,
            velocity: None,
            dcdt: None,
            source: None,
        }
    }
}

impl<B: Backend> PhysicsLoss<B> {
    /// 腐蚀后的物理残差评估域。掩码为 `None` 时返回 `None`（退化成全场平均）。
    fn interior(&self) -> Option<Tensor<B, 4>> {
        self.mask.clone().map(|m| erode_mask(single_channel(m), 1))
    }

    /// 把有量纲场按通道拆开；dispersion 任务下再把输入通道来的速度场接上。
    fn fields(&self, phys: &Tensor<B, 4>) -> Fields<B> {
        let mut fields = split_fields(phys, self.channel_order);
        if self.task == Task::Dispersion {
            if let Some((vx, vy)) = self.velocity.clone() {
                fields.vx = Some(vx);
                fields.vy = Some(vy);
            }
        }
        fields
    }

    /// 动态权重斜坡：早期先把数据拟合住，中后期逐步压紧物理约束。
    fn ramp(&self) -> f64 {
        let progress = (self.epoch as f64 / self.total_epochs.max(1) as f64).clamp(0.0, 1.0);
        if progress < self.physics_warmup {
            0.0
        } else if self.physics_warmup >= 1.0 {
            1.0
        } else {
            (progress - self.physics_warmup) / (1.0 - self.physics_warmup).max(1e-8)
        }
    }

    /// 返回 `(总损失, 分项明细)`。
    ///
    /// 权重为 0 的项**直接跳过计算**，不是算完再乘 0——地铁站方案 §3.5 去掉动量
    /// 残差就是靠 `ns_weight = 0` 实现的，跳过才能真正省下算力。
    pub fn compute(&self, output: Tensor<B, 4>, target: Tensor<B, 4>) -> (Tensor<B, 1>, LossParts) {
        // 数据项在归一化空间算：各通道等权，压力量级不会压过速度
        let base = reconstruction_loss(output.clone(), target, self.mask.clone(), true);

        // 物理项在有量纲空间算
        let phys = denormalize(output.clone(), self.y_min.as_deref(), self.y_max.as_deref());
        let fields = self.fields(&phys);
        let interior = self.interior();

        // 乘 0 的和：跳过某项时用它占位，保持计算图连通
        let zero = output.sum().mul_scalar(0.0);
        let mut div_loss = zero.clone();
        let mut ns_loss = zero.clone();
        let mut advdiff_loss = zero;

        match self.task {
            Task::Dispersion => {
                if self.advdiff_weight > 0.0 {
                    if let Some(got) = advection_diffusion_loss(
                        &fields,
                        interior.clone(),
                        self.diffusivity,
                        self.dx,
                        self.dy,
                        self.dcdt.clone(),
                        self.source.clone(),
                    ) {
                        advdiff_loss = got;
                    }
                }
            }
            Task::Flow => {
                if self.div_weight > 0.0 {
                    if let Some(got) = divergence_loss(&fields, interior.clone(), self.dx, self.dy)
                    {
                        div_loss = got;
                    }
                }
                if self.ns_weight > 0.0 {
                    if let Some(got) = navier_stokes_loss(
                        &fields,
                        interior.clone(),
                        self.rho,
                        self.nu,
                        self.dx,
                        self.dy,
                    ) {
                        ns_loss = got;
                    }
                }
            }
        }

        let smooth = smoothness_loss(phys, interior, self.dx, self.dy);

        let ramp = self.ramp();
        let w_div = self.div_weight * (1.0 + ramp);
        let w_ns = self.ns_weight * (1.0 + ramp);
        let w_advdiff = self.advdiff_weight * (1.0 + ramp);
        let w_smooth = self.smooth_weight * (1.0 - 0.5 * ramp);

        let s_div = Scales::or_one(self.scales.div);
        let s_ns = Scales::or_one(self.scales.ns);
        let s_advdiff = Scales::or_one(self.scales.advdiff);
        let s_smooth = Scales::or_one(self.scales.smooth);

        let total = base.clone()
            + div_loss.clone().mul_scalar(w_div).div_scalar(s_div)
            + ns_loss.clone().mul_scalar(w_ns).div_scalar(s_ns)
            + advdiff_loss
                .clone()
                .mul_scalar(w_advdiff)
                .div_scalar(s_advdiff)
            + smooth.clone().mul_scalar(w_smooth).div_scalar(s_smooth);

        let parts = LossParts {
            base: scalar(base),
            div: scalar(div_loss),
            ns: scalar(ns_loss),
            advdiff: scalar(advdiff_loss),
            smooth: scalar(smooth),
            total: scalar(total.clone()),
            w_div,
            w_ns,
            w_advdiff,
            w_smooth,
            scale_div: s_div,
            scale_ns: s_ns,
            scale_advdiff: s_advdiff,
            scale_smooth: s_smooth,
        };
        (total, parts)
    }

    /// 诊断用：有量纲物理残差 RMS，判断物理约束是否真的生效。
    ///
    /// 与 [`Self::compute`] 不同，这里**不看权重**——`ns_weight = 0` 时照样报动量
    /// 残差，否则关掉约束后就没法监控它有没有恶化。
    pub fn report(&self, output: Tensor<B, 4>) -> PhysicsReport {
        let phys = denormalize(
            output.detach(),
            self.y_min.as_deref(),
            self.y_max.as_deref(),
        );
        let fields = self.fields(&phys);
        let interior = self.interior();

        match self.task {
            Task::Dispersion => PhysicsReport {
                advdiff_rms: advection_diffusion_loss(
                    &fields,
                    interior,
                    self.diffusivity,
                    self.dx,
                    self.dy,
                    self.dcdt.clone(),
                    self.source.clone(),
                )
                .map(|t| scalar(t.sqrt())),
                ..Default::default()
            },
            Task::Flow => PhysicsReport {
                div_rms: divergence_loss(&fields, interior.clone(), self.dx, self.dy)
                    .map(|t| scalar(t.sqrt())),
                ns_rms: navier_stokes_loss(&fields, interior, self.rho, self.nu, self.dx, self.dy)
                    .map(|t| scalar(t.sqrt())),
                advdiff_rms: None,
            },
        }
    }
}

/// 取标量值（等价于 Python 的 `float(t.detach())`）。
pub fn scalar<B: Backend>(t: Tensor<B, 1>) -> f64 {
    t.into_scalar().elem::<f64>()
}

// ------------------------------------------------------------ 量级归一化

/// 各物理项量级的指数滑动估计，让 `div_weight` 之类的权重与量纲解耦。
///
/// 不加这层的话，同一组权重换个 domain_size 或分辨率就会从「几乎不起作用」变成
/// 「压垮数据项」，因为残差量级随 1/dx、1/dx² 变化。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsScaler {
    pub momentum: f64,
    pub enabled: bool,
    state: Scales,
}

impl Default for PhysicsScaler {
    fn default() -> Self {
        Self {
            momentum: 0.99,
            enabled: true,
            state: Scales::default(),
        }
    }
}

impl PhysicsScaler {
    pub fn new(momentum: f64, enabled: bool) -> Self {
        Self {
            momentum,
            enabled,
            state: Scales::default(),
        }
    }

    /// 当前估计。关掉时返回全 `None`，[`PhysicsLoss`] 会当成不缩放。
    pub fn current(&self) -> Scales {
        if self.enabled {
            self.state
        } else {
            Scales::default()
        }
    }

    /// 用一步的分项值更新估计。非有限值与非正值直接跳过——训练初期某项恰好为 0
    /// 时若把 0 记进去，后面所有残差都会被除爆。
    pub fn update(&mut self, parts: &LossParts) {
        if !self.enabled {
            return;
        }
        let m = self.momentum;
        ema(&mut self.state.div, parts.div, m);
        ema(&mut self.state.ns, parts.ns, m);
        ema(&mut self.state.advdiff, parts.advdiff, m);
        ema(&mut self.state.smooth, parts.smooth, m);
    }
}

fn ema(slot: &mut Option<f64>, value: f64, momentum: f64) {
    if !value.is_finite() || value <= 0.0 {
        return;
    }
    *slot = Some(match *slot {
        None => value,
        Some(prev) => momentum * prev + (1.0 - momentum) * value,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn channel_order_resolves_and_degrades() {
        assert_eq!(ChannelOrder::parse("pvv").unwrap(), ChannelOrder::Pvv);
        assert_eq!(ChannelOrder::parse("VVP").unwrap(), ChannelOrder::Vvp);
        assert!(ChannelOrder::parse("nonsense").is_err(), "非法顺序必须报错");

        let x = Tensor::<B, 4>::zeros([1, 3, 4, 4], &Default::default());
        let f = split_fields(&x, ChannelOrder::Vvp);
        assert!(f.vx.is_some() && f.vy.is_some() && f.p.is_some());

        // 通道数不足时自动降级：pvv 只喂 2 个通道就只剩 P 和 Vx
        let x2 = Tensor::<B, 4>::zeros([1, 2, 4, 4], &Default::default());
        let f2 = split_fields(&x2, ChannelOrder::Pvv);
        assert!(f2.p.is_some() && f2.vx.is_some());
        assert!(f2.vy.is_none(), "第 3 个通道不存在时不能凭空造出 Vy");
    }

    #[test]
    fn channel_order_actually_changes_which_field_is_which() {
        // 通道顺序搞反是「不报错的错」的典型：拿 P 当 Vx 去算散度，数值照样出得来
        let device = Default::default();
        let data: Vec<f32> = (0..3 * 4).map(|i| (i / 4) as f32).collect(); // 通道 i 全等于 i
        let x = Tensor::<B, 1>::from_data(TensorData::from(data.as_slice()), &device)
            .reshape([1, 3, 2, 2]);
        let pvv = split_fields(&x, ChannelOrder::Pvv);
        let vvp = split_fields(&x, ChannelOrder::Vvp);
        assert_eq!(scalar(pvv.p.unwrap().mean()), 0.0);
        assert_eq!(
            scalar(vvp.p.unwrap().mean()),
            2.0,
            "vvp 的 P 应当是第 3 个通道"
        );
    }

    #[test]
    fn every_channel_order_maps_each_field_to_the_right_index() {
        // **vx / vy 互换是「不报错的错」的典型**：散度约束会变成算 dVy/dx + dVx/dy，
        // 数值照样出得来。原先的测试只断言「存在」和「p 落在哪」，vx/vy 一个都没验，
        // 于是把 Vvp 分支的 vx 和 vy 写反，全套测试仍然全绿。
        // 这里让每个通道恒等于它自己的下标，逐个物理量核对取到的是第几个通道。
        let device = Default::default();
        let make = |nc: usize| {
            let data: Vec<f32> = (0..nc * 4).map(|i| (i / 4) as f32).collect();
            Tensor::<B, 1>::from_data(TensorData::from(data.as_slice()), &device)
                .reshape([1, nc, 2, 2])
        };
        let idx = |t: Option<Tensor<B, 4>>| t.map(|v| scalar(v.mean()) as usize);

        type Case = (
            ChannelOrder,
            usize,
            Option<usize>,
            Option<usize>,
            Option<usize>,
            Option<usize>,
        );
        // (顺序, 通道数, p, vx, vy, c) —— 与 utils/physics_loss.py 的 CHANNEL_ORDERS 对齐
        let cases: [Case; 4] = [
            (ChannelOrder::Pvv, 3, Some(0), Some(1), Some(2), None),
            (ChannelOrder::Vvp, 3, Some(2), Some(0), Some(1), None),
            (ChannelOrder::Pvvt, 4, Some(0), Some(1), Some(2), None),
            (ChannelOrder::C, 1, None, None, None, Some(0)),
        ];
        for (order, nc, p, vx, vy, c) in cases {
            let f = split_fields(&make(nc), order);
            assert_eq!(idx(f.p), p, "{:?} 的 P 取错了通道", order);
            assert_eq!(idx(f.vx), vx, "{:?} 的 Vx 取错了通道", order);
            assert_eq!(idx(f.vy), vy, "{:?} 的 Vy 取错了通道", order);
            assert_eq!(idx(f.c), c, "{:?} 的 C 取错了通道", order);
        }
    }

    #[test]
    fn flatten_depth_keeps_channel_semantics() {
        // 旧版 reshape 不换轴，会把通道维和深度维搅乱，且不报错
        let (b, c, d, h, w) = (2usize, 3usize, 4usize, 5usize, 6usize);
        let device = Default::default();
        let mut data = vec![0.0f32; b * c * d * h * w];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i / (d * h * w)) % c) as f32 + 1.0; // 第 ch 个通道全等于 ch+1
        }
        let x = Tensor::<B, 1>::from_data(TensorData::from(data.as_slice()), &device)
            .reshape([b, c, d, h, w]);

        let (flat, bb, dd) = flatten_depth(x.clone());
        assert_eq!(flat.dims(), [b * d, c, h, w]);
        for slice in 0..b * d {
            for ch in 0..c {
                let v = flat
                    .clone()
                    .slice([slice..slice + 1, ch..ch + 1, 0..h, 0..w])
                    .mean();
                assert_eq!(
                    scalar(v),
                    ch as f64 + 1.0,
                    "切片 {} 的通道 {} 被搅乱了",
                    slice,
                    ch
                );
            }
        }

        let back = unflatten_depth(flat, bb, dd);
        assert_eq!(scalar((back - x).abs().max()), 0.0, "往返变换必须可逆");
    }

    #[test]
    fn analytic_divergence_free_field_has_near_zero_divergence() {
        // v = (sin(2πx)cos(2πy), -cos(2πx)sin(2πy)) 解析无散
        let n = 33usize;
        let device = Default::default();
        let h = 1.0f64 / (n - 1) as f64;
        let mut vx = vec![0.0f32; n * n];
        let mut vy = vec![0.0f32; n * n];
        for iy in 0..n {
            for ix in 0..n {
                let tx = 2.0 * std::f64::consts::PI * (ix as f64 * h);
                let ty = 2.0 * std::f64::consts::PI * (iy as f64 * h);
                vx[iy * n + ix] = (tx.sin() * ty.cos()) as f32;
                vy[iy * n + ix] = (-tx.cos() * ty.sin()) as f32;
            }
        }
        let mk = |v: &[f32]| {
            Tensor::<B, 1>::from_data(TensorData::from(v), &device).reshape([1, 1, n, n])
        };

        let inner = divergence(mk(&vx), mk(&vy), h, h).slice([0..1, 0..1, 2..n - 2, 2..n - 2]);
        let worst = scalar(inner.abs().max());
        assert!(worst < 0.6, "解析无散场的离散散度应接近 0，实得 {}", worst);

        // 逐通道归一化会把两个分量按不同因子缩放，归一化空间的散度不再为零
        let vyn: Vec<f32> = vy.iter().map(|v| v / 10.0).collect();
        let bad = scalar(
            divergence(mk(&vx), mk(&vyn), h, h)
                .slice([0..1, 0..1, 2..n - 2, 2..n - 2])
                .abs()
                .max(),
        );
        assert!(
            bad > 1.0,
            "在归一化空间求散度本来就是错的（旧版正是这么干的），实得 {}",
            bad
        );
    }

    /// 造一份只有残差分项有意义的 [`LossParts`]，权重与缩放不参与量级估计。
    fn parts_of(div: f64, ns: f64, advdiff: f64, smooth: f64) -> LossParts {
        LossParts {
            base: 0.0,
            div,
            ns,
            advdiff,
            smooth,
            total: 0.0,
            w_div: 0.0,
            w_ns: 0.0,
            w_advdiff: 0.0,
            w_smooth: 0.0,
            scale_div: 1.0,
            scale_ns: 1.0,
            scale_advdiff: 1.0,
            scale_smooth: 1.0,
        }
    }

    #[test]
    fn scaler_converges_and_can_be_disabled() {
        let mut s = PhysicsScaler::new(0.5, true);
        // advdiff 为 0：不参与的项应被跳过，而不是把 0 记进去
        let parts = parts_of(1000.0, 2e6, 0.0, 300.0);
        for _ in 0..50 {
            s.update(&parts);
        }
        let cur = s.current();
        assert!((cur.div.unwrap() - 1000.0).abs() < 1.0);
        assert!((cur.ns.unwrap() - 2e6).abs() < 1e4);
        assert_eq!(cur.advdiff, None, "值为 0 的项不能进 EMA，否则后面会被除爆");

        let mut off = PhysicsScaler::new(0.5, false);
        off.update(&parts);
        assert_eq!(off.current(), Scales::default(), "关掉后应当不缩放");
    }

    #[test]
    fn scaler_smooths_spikes_instead_of_tracking_them() {
        // 只喂恒定值的话，「EMA」和「直接覆盖」给出的结果一样，测不出区别。
        // 量级估计的全部意义就是抗突变：单个 batch 的异常值不能把缩放因子带跑。
        let mut s = PhysicsScaler::new(0.9, true);
        s.update(&parts_of(100.0, 0.0, 0.0, 0.0));
        assert_eq!(s.current().div, Some(100.0), "首个有效值应直接作为初值");

        s.update(&parts_of(1.0, 0.0, 0.0, 0.0));
        let got = s.current().div.unwrap();
        assert!(
            (got - 90.1).abs() < 1e-9,
            "0.9*100 + 0.1*1 = 90.1，实得 {}（若等于 1.0 说明退化成了直接覆盖）",
            got
        );
    }

    #[test]
    fn warmup_holds_physics_terms_at_base_weight() {
        // 斜坡逻辑只在 progress 与 physics_warmup 的相对位置上体现，黄金数据只覆盖
        // 了 warmup 之后的一个 epoch，单靠它抓不到 warmup 失效
        let device = Default::default();
        let out = Tensor::<B, 4>::zeros([1, 3, 6, 6], &device);
        let tgt = Tensor::<B, 4>::zeros([1, 3, 6, 6], &device);
        let weights = |epoch: usize| {
            PhysicsLoss::<B> {
                epoch,
                total_epochs: 100,
                physics_warmup: 0.5,
                ..Default::default()
            }
            .compute(out.clone(), tgt.clone())
            .1
        };

        // warmup 期内（progress = 0.1 < 0.5）：斜坡为 0，权重就是基准值
        let early = weights(10);
        assert_eq!(early.w_div, 0.05, "warmup 期内物理权重不应被抬高");
        assert_eq!(early.w_smooth, 0.005, "warmup 期内平滑权重不应被压低");

        // 刚过 warmup（progress = 0.6）：斜坡 0.2
        let mid = weights(60);
        assert!(
            (mid.w_div - 0.05 * 1.2).abs() < 1e-12,
            "warmup 之后物理权重应随进度线性抬高，实得 {}",
            mid.w_div
        );

        // 训练末尾：斜坡满格，物理项翻倍、平滑项减半
        let late = weights(100);
        assert!((late.w_div - 0.10).abs() < 1e-12, "实得 {}", late.w_div);
        assert!((late.w_ns - 0.010).abs() < 1e-12, "实得 {}", late.w_ns);
        assert!(
            (late.w_smooth - 0.0025).abs() < 1e-12,
            "实得 {}",
            late.w_smooth
        );
    }
}
