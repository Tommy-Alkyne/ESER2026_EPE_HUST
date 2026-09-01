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
    /// (P, Vx, Vy, Vz) —— 三维流场
    Pvvv,
    /// (P, Vx, Vy, Vz, T)。T 只作为输出学习，**不参与任何残差**
    Pvvvt,
}

impl ChannelOrder {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "pvv" => Ok(Self::Pvv),
            "vvp" => Ok(Self::Vvp),
            "pvvt" => Ok(Self::Pvvt),
            "c" => Ok(Self::C),
            "pvvv" => Ok(Self::Pvvv),
            "pvvvt" => Ok(Self::Pvvvt),
            other => Err(format!(
                "未知的 channel_order={:?}，可选 c / pvv / pvvt / pvvv / pvvvt / vvp",
                other
            )),
        }
    }

    /// `(p, vx, vy, c)` 各自的通道下标；`None` 表示这个顺序里没有该物理量。
    /// 各物理量在通道维的下标：`[p, vx, vy, vz, c]`。
    pub fn indices(self) -> [Option<usize>; 5] {
        match self {
            Self::Pvv | Self::Pvvt => [Some(0), Some(1), Some(2), None, None],
            Self::Vvp => [Some(2), Some(0), Some(1), None, None],
            Self::C => [None, None, None, None, Some(0)],
            Self::Pvvv | Self::Pvvvt => [Some(0), Some(1), Some(2), Some(3), None],
        }
    }

    /// 这个通道口径带不带第三个速度分量。
    pub fn has_vz(self) -> bool {
        matches!(self, Self::Pvvv | Self::Pvvvt)
    }
}

/// 按物理量拆开的场。缺哪个量对应的损失就返回 `None`——**宁可不算，也不能拿
/// 别的通道顶上去静默算错**。
#[derive(Debug, Clone, Default)]
pub struct Fields<B: Backend> {
    pub p: Option<Tensor<B, 4>>,
    pub vx: Option<Tensor<B, 4>>,
    pub vy: Option<Tensor<B, 4>>,
    /// 竖向速度。只有三维口径才有；二维路径恒为 `None`，所有 z 项因此自动不算
    pub vz: Option<Tensor<B, 4>>,
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
    let [p, vx, vy, vz, c] = order.indices();
    Fields {
        p: take(p),
        vx: take(vx),
        vy: take(vy),
        vz: take(vz),
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

// ---------------------------------------------------------------- 深度轴

/// 三维时的深度轴上下文。
///
/// 三维残差 = **现有二维代码逐层算出来的 x/y 项** + 这里补的 z 项。这样安排是为了
/// 让 `tests/golden/golden.npz` 里 502 个冻结张量覆盖的那条路径一个字都不改——
/// 另写一份三维公式的话，二维和三维会各自漂移，而且没有任何参考数据能发现。
///
/// `None` 表示二维：所有 z 项一概不算，不是「按 0 算」。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZAxis {
    /// 展平前的 batch 数
    pub b: usize,
    /// 深度层数
    pub d: usize,
    /// 层间距（真实物理长度）
    pub dz: f64,
}

impl ZAxis {
    /// 中心差分要求上下各有一层邻居，`d < 3` 时**任何**格点都算不出 ∂/∂z。
    pub fn check(&self) -> Result<(), String> {
        if self.d < 3 {
            return Err(format!(
                "深度只有 {} 层：z 方向中心差分在任何格点上都不成立，\n\
                 三维物理项无法计算。要么给至少 3 层，要么按二维逐层训练",
                self.d
            ));
        }
        if self.dz <= 0.0 || self.dz.is_nan() {
            return Err(format!("层间距 dz 必须为正，实得 {}", self.dz));
        }
        Ok(())
    }
}

/// 对**已展平成 4D** 的场求 z 方向导数：还原成 5D、求导、再展平回去。
///
/// `second = true` 取二阶导（补全三维 Laplacian），否则一阶。
pub fn dz_of<B: Backend>(x: Tensor<B, 4>, z: ZAxis, second: bool) -> Tensor<B, 4> {
    let x5 = unflatten_depth(x, z.b, z.d);
    let g = if second {
        crate::ops::second_diff_z(x5, z.dz)
    } else {
        crate::ops::grad_z(x5, z.dz)
    };
    flatten_depth(g).0
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
    z: Option<ZAxis>,
) -> Option<Tensor<B, 1>> {
    let (vx, vy) = (fields.vx.clone()?, fields.vy.clone()?);
    let mut d = divergence(vx, vy, dx, dy);
    // 三维的连续性方程是 ∂u/∂x + ∂v/∂y + ∂w/∂z = 0。少了第三项就不是不可压条件，
    // 而且**数值上照样收敛**——网络会学出一个二维意义上无散、三维意义上错的场
    if let (Some(z), Some(vz)) = (z, fields.vz.clone()) {
        d = d + dz_of(vz, z, false);
    }
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
    z: Option<ZAxis>,
) -> Option<Tensor<B, 1>> {
    let moms = navier_stokes_residual(fields, rho, nu, dx, dy, z)?;
    Some(
        moms.into_iter()
            .map(|m| masked_mean(m.clone() * m, mask.clone()))
            .reduce(|a, b| a + b)
            .expect("动量残差至少有 x / y 两个分量"),
    )
}

/// N-S 动量残差**的各个分量场**（二维两个、三维三个）。
///
/// 单独拆出来而不是塞在 `navier_stokes_loss` 里：三维要在这些场上继续加 z 项，
/// 拆开才能保证损失和诊断用的是同一份公式。
pub fn navier_stokes_residual<B: Backend>(
    fields: &Fields<B>,
    rho: f64,
    nu: f64,
    dx: f64,
    dy: f64,
    z: Option<ZAxis>,
) -> Option<Vec<Tensor<B, 4>>> {
    let (vx, vy, p) = (fields.vx.clone()?, fields.vy.clone()?, fields.p.clone()?);

    let (dvx_dx, dvx_dy) = grad_xy(vx.clone(), dx, dy);
    let (dvy_dx, dvy_dy) = grad_xy(vy.clone(), dx, dy);
    let (dp_dx, dp_dy) = grad_xy(p.clone(), dx, dy);
    let lap_vx = laplacian(vx.clone(), dx, dy);
    let lap_vy = laplacian(vy.clone(), dx, dy);

    let mut mom_x = vx.clone() * dvx_dx + vy.clone() * dvx_dy + dp_dx.div_scalar(rho)
        - lap_vx.mul_scalar(nu);
    let mut mom_y =
        vx.clone() * dvy_dx + vy.clone() * dvy_dy + dp_dy.div_scalar(rho) - lap_vy.mul_scalar(nu);

    let (Some(z), Some(vz)) = (z, fields.vz.clone()) else {
        return Some(vec![mom_x, mom_y]);
    };

    // x / y 分量补上竖向对流 w·∂/∂z 与竖向粘性 ν·∂²/∂z²
    mom_x = mom_x + vz.clone() * dz_of(vx.clone(), z, false)
        - dz_of(vx.clone(), z, true).mul_scalar(nu);
    mom_y = mom_y + vz.clone() * dz_of(vy.clone(), z, false)
        - dz_of(vy.clone(), z, true).mul_scalar(nu);

    // 第三个动量方程。没有它，竖向速度只受连续性方程约束——楼梯口的竖向流会
    // 学成一个满足质量守恒但完全不符合动量守恒的场
    let (dvz_dx, dvz_dy) = grad_xy(vz.clone(), dx, dy);
    let dp_dz = dz_of(p, z, false);
    let mom_z = vx * dvz_dx + vy * dvz_dy + vz.clone() * dz_of(vz.clone(), z, false)
        + dp_dz.div_scalar(rho)
        - (laplacian(vz.clone(), dx, dy) + dz_of(vz, z, true)).mul_scalar(nu);

    Some(vec![mom_x, mom_y, mom_z])
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
    /// 边界条件残差。量级是 O(v²)，比 div 的 O((v/dx)²) 小好几个数量级，
    /// 不单独归一化的话边界项在总损失里根本看不见。
    pub bc: Option<f64>,
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
    /// 边界条件残差合计（固壁不可穿透 / 无通量、风口法向速度）
    pub bc: f64,
    pub total: f64,
    pub w_div: f64,
    pub w_ns: f64,
    pub w_advdiff: f64,
    pub w_smooth: f64,
    pub w_bc: f64,
    pub scale_div: f64,
    pub scale_ns: f64,
    pub scale_advdiff: f64,
    pub scale_smooth: f64,
    pub scale_bc: f64,
}

/// 诊断用的有量纲物理残差 RMS。没算的项是 `None`（Python 侧是 nan）。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PhysicsReport {
    pub div_rms: Option<f64>,
    pub ns_rms: Option<f64>,
    pub advdiff_rms: Option<f64>,
    /// 边界条件残差 RMS。`None` 表示这次几何里根本没有边界标签
    pub bc_rms: Option<f64>,
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
    /// 层间距。只有三维路径用得到；二维时取什么值都不影响结果
    pub dz: f64,
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
    /// 边界分区几何。`None` = 这次几何没有边界标签，所有边界项自动不算
    pub boundary: Option<crate::boundary::Boundary<B>>,
    /// 深度轴。`None` = 二维；三维路径由 `compute_3d` 自动填
    pub z_axis: Option<ZAxis>,
    /// 仅 dispersion 三维用：**有量纲**竖向速度场，来自网络输入通道
    pub velocity_z: Option<Tensor<B, 4>>,
    /// 边界条件残差的权重
    pub bc_weight: f64,
    /// 送风口风速（m/s，正值 = 向域内送风）
    pub inlet_speed: f64,
    /// 排风口风速（m/s，正值 = 从域内抽出）
    pub outlet_speed: f64,
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
            dz: 1.0,
            physics_warmup: 0.0,
            scales: Scales::default(),
            mask: None,
            y_min: None,
            y_max: None,
            velocity: None,
            dcdt: None,
            source: None,
            boundary: None,
            z_axis: None,
            velocity_z: None,
            bc_weight: 0.05,
            inlet_speed: 0.0,
            outlet_speed: 0.0,
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
            fields.vz = self.velocity_z.clone();
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
        let mut advdiff_loss = zero.clone();

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
                        self.z_axis,
                    ) {
                        advdiff_loss = got;
                    }
                }
            }
            Task::Flow => {
                if self.div_weight > 0.0 {
                    if let Some(got) =
                        divergence_loss(&fields, interior.clone(), self.dx, self.dy, self.z_axis)
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
                        self.z_axis,
                    ) {
                        ns_loss = got;
                    }
                }
            }
        }

        // 没有边界标签时用 0 占位，保持计算图连通（与上面几项同一套路）
        let bc_loss = self.boundary_loss(&fields).unwrap_or(zero);

        let smooth = smoothness_loss(phys, interior, self.dx, self.dy);

        let ramp = self.ramp();
        let w_div = self.div_weight * (1.0 + ramp);
        let w_ns = self.ns_weight * (1.0 + ramp);
        let w_advdiff = self.advdiff_weight * (1.0 + ramp);
        let w_smooth = self.smooth_weight * (1.0 - 0.5 * ramp);
        let w_bc = self.bc_weight * (1.0 + ramp);

        let s_div = Scales::or_one(self.scales.div);
        let s_ns = Scales::or_one(self.scales.ns);
        let s_advdiff = Scales::or_one(self.scales.advdiff);
        let s_smooth = Scales::or_one(self.scales.smooth);
        let s_bc = Scales::or_one(self.scales.bc);

        let total = base.clone()
            + div_loss.clone().mul_scalar(w_div).div_scalar(s_div)
            + ns_loss.clone().mul_scalar(w_ns).div_scalar(s_ns)
            + advdiff_loss
                .clone()
                .mul_scalar(w_advdiff)
                .div_scalar(s_advdiff)
            + smooth.clone().mul_scalar(w_smooth).div_scalar(s_smooth)
            + bc_loss.clone().mul_scalar(w_bc).div_scalar(s_bc);

        let parts = LossParts {
            base: scalar(base),
            div: scalar(div_loss),
            ns: scalar(ns_loss),
            advdiff: scalar(advdiff_loss),
            smooth: scalar(smooth),
            bc: scalar(bc_loss),
            total: scalar(total.clone()),
            w_div,
            w_ns,
            w_advdiff,
            w_smooth,
            w_bc,
            scale_div: s_div,
            scale_ns: s_ns,
            scale_advdiff: s_advdiff,
            scale_smooth: s_smooth,
            scale_bc: s_bc,
        };
        (total, parts)
    }

    /// 这次前向到底用不用得上 z 项，以及层数够不够。
    ///
    /// **没有第三个速度分量就不算「三维物理」**：`--ndim 3` 只是说网络是三维卷积，
    /// 通道口径仍可能是二维的（P, Vx, Vy）。那种情况下所有 z 项本来就不参与，
    /// 层数只有 2 层也没关系——退回逐层二维，与本次改动之前的行为完全一致。
    ///
    /// 反过来，真要算 z 项却只给 2 层，就必须报错：中心差分在任何格点上都不成立，
    /// 算出来的是个没有意义的数，而且不会报错。
    fn resolve_z(&self, b: usize, d: usize) -> Result<Option<ZAxis>, String> {
        if !(self.channel_order.has_vz() || self.velocity_z.is_some()) {
            return Ok(None);
        }
        let z = ZAxis { b, d, dz: self.dz };
        z.check()?;
        Ok(Some(z))
    }

    /// 三维版本：吃 (B, C, D, H, W)，把深度维折进 batch 维后走**同一条**损失路径，
    /// 并把 z 方向的项接上。
    ///
    /// 掩码按深度存在 `mask` 的第 0 维（二维时该维恒为 1），这里复制 B 份对齐展平
    /// 后的 (B*D, C, H, W)。顺序必须是「b 外 d 内」，与 [`flatten_depth`] 一致。
    pub fn compute_3d(
        &self,
        output: Tensor<B, 5>,
        target: Tensor<B, 5>,
    ) -> Result<(Tensor<B, 1>, LossParts), String> {
        let (out4, b, d) = flatten_depth(output);
        let (y4, _, _) = flatten_depth(target);
        let cfg = Self {
            z_axis: self.resolve_z(b, d)?,
            mask: self.mask.clone().map(|m| tile_along_batch(m, b)),
            ..self.clone()
        };
        Ok(cfg.compute(out4, y4))
    }

    /// 三维版本的残差诊断。参数含义同 [`Self::compute_3d`]。
    pub fn report_3d(&self, output: Tensor<B, 5>) -> Result<PhysicsReport, String> {
        let (out4, b, d) = flatten_depth(output);
        let cfg = Self {
            z_axis: self.resolve_z(b, d)?,
            mask: self.mask.clone().map(|m| tile_along_batch(m, b)),
            ..self.clone()
        };
        Ok(cfg.report(out4))
    }

    /// 边界条件残差合计。没有边界标签、或者该任务下没有可约束的场时返回 `None`。
    ///
    /// **按任务分派，不是两个都算**：
    /// * `Flow` 约束速度——速度是网络输出，约束它才有梯度
    /// * `Dispersion` 约束浓度的法向导数——这个任务里速度来自**输入通道**，
    ///   是固定的数据，对它加约束一点梯度都没有，纯属白算
    fn boundary_loss(&self, fields: &Fields<B>) -> Option<Tensor<B, 1>> {
        let b = self.boundary.as_ref()?;
        let mut acc: Option<Tensor<B, 1>> = None;
        let mut add = |t: Tensor<B, 1>| {
            acc = Some(match acc.take() {
                None => t,
                Some(prev) => prev + t,
            });
        };
        match self.task {
            Task::Flow => {
                let (vx, vy) = (fields.vx.clone()?, fields.vy.clone()?);
                for (zone, target) in [
                    (b.wall.as_ref(), 0.0),
                    // 外法向朝域外，所以「送进来」是负、「抽出去」是正
                    (b.inlet.as_ref(), -self.inlet_speed),
                    (b.outlet.as_ref(), self.outlet_speed),
                ] {
                    if let Some(z) = zone {
                        add(crate::boundary::normal_velocity_loss(
                            vx.clone(),
                            vy.clone(),
                            z,
                            target,
                        ));
                    }
                }
            }
            Task::Dispersion => {
                let c = fields.c.clone()?;
                if let Some(z) = b.wall.as_ref() {
                    add(crate::boundary::normal_gradient_loss(c, z, self.dx, self.dy));
                }
            }
        }
        acc
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

        // 边界残差与权重无关地报出来——bc_weight = 0 时也要能监控它有没有恶化
        let bc_rms = self.boundary_loss(&fields).map(|t| scalar(t.sqrt()));

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
                    self.z_axis,
                )
                .map(|t| scalar(t.sqrt())),
                bc_rms,
                ..Default::default()
            },
            Task::Flow => PhysicsReport {
                div_rms: divergence_loss(&fields, interior.clone(), self.dx, self.dy, self.z_axis)
                    .map(|t| scalar(t.sqrt())),
                ns_rms: navier_stokes_loss(
                    &fields,
                    interior,
                    self.rho,
                    self.nu,
                    self.dx,
                    self.dy,
                    self.z_axis,
                )
                .map(|t| scalar(t.sqrt())),
                advdiff_rms: None,
                bc_rms,
            },
        }
    }
}

/// 把 (D, 1, H, W) 的逐层掩码复制 `b` 份，得到与展平后的 (B*D, C, H, W) 对齐的形状。
///
/// 顺序是 `[层0..层D-1, 层0..层D-1, ...]`，正好对上 [`flatten_depth`] 给出的
/// `b * D + d`。搞反成「层内复制」不会报错，只会让每个样本配到错误的那一层掩码。
fn tile_along_batch<B: Backend>(m: Tensor<B, 4>, b: usize) -> Tensor<B, 4> {
    if b <= 1 {
        return m;
    }
    Tensor::cat((0..b).map(|_| m.clone()).collect::<Vec<_>>(), 0)
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
        ema(&mut self.state.bc, parts.bc, m);
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

    /// 边界项必须**真的进总损失**，不是只算出来打个日志。
    ///
    /// 这条补的是集成测试的一个洞：那边只验了进度行里有 `bc=`，而 `parts.bc` 是
    /// 无条件算的——就算 `total` 里根本没加这一项，日志照样好看、训练照样跑完，
    /// 边界条件却一条都没起作用。
    #[test]
    fn boundary_term_is_actually_added_to_the_total() {
        use crate::boundary::Boundary;
        use crate::geometry::Zone;

        let device = Default::default();
        let (h, w) = (6usize, 8usize);

        // 左边流体、右边墙
        let mut z = vec![Zone::Fluid.label(); h * w];
        for y in 0..h {
            for x in 5..w {
                z[y * w + x] = Zone::Wall.label();
            }
        }
        let boundary = Boundary::<B>::build(&z, h, w, 1.0, 1.0, &device).expect("应当认出墙");

        // 输出三通道 (P, Vx, Vy)，Vx 恒为 1 => 处处垂直穿墙，边界残差必然非零
        let mut vals = vec![0.0f32; 3 * h * w];
        for v in vals.iter_mut().skip(h * w).take(h * w) {
            *v = 1.0;
        }
        let output = Tensor::<B, 1>::from_data(TensorData::from(vals.as_slice()), &device)
            .reshape([1, 3, h, w]);
        let target = Tensor::<B, 4>::zeros([1, 3, h, w], &device);

        let base = PhysicsLoss::<B> {
            task: Task::Flow,
            channel_order: ChannelOrder::Pvv,
            div_weight: 0.0,
            ns_weight: 0.0,
            smooth_weight: 0.0,
            dx: 1.0,
            dy: 1.0,
            physics_warmup: 0.0,
            boundary: Some(boundary),
            ..Default::default()
        };

        let (t_off, p_off) = PhysicsLoss::<B> {
            bc_weight: 0.0,
            ..base.clone()
        }
        .compute(output.clone(), target.clone());
        let (t_on, p_on) = PhysicsLoss::<B> {
            bc_weight: 1.0,
            ..base
        }
        .compute(output, target);

        assert!(p_off.bc > 0.0, "残差本身应当非零，实得 {}", p_off.bc);
        assert!(
            (p_on.bc - p_off.bc).abs() < 1e-6,
            "权重不该改变残差本身：{} vs {}",
            p_off.bc,
            p_on.bc
        );
        let delta = scalar(t_on) - scalar(t_off);
        assert!(
            delta > 1e-6,
            "把 --bc-weight 从 0 提到 1，总损失必须变大；实得差 {}（说明边界项没进 total）",
            delta
        );
        let _ = (p_off.total, p_on.total);
    }

    /// 造一份只有残差分项有意义的 [`LossParts`]，权重与缩放不参与量级估计。
    fn parts_of(div: f64, ns: f64, advdiff: f64, smooth: f64) -> LossParts {
        LossParts {
            base: 0.0,
            div,
            ns,
            advdiff,
            smooth,
            bc: 0.0,
            total: 0.0,
            w_div: 0.0,
            w_ns: 0.0,
            w_advdiff: 0.0,
            w_smooth: 0.0,
            w_bc: 0.0,
            scale_div: 1.0,
            scale_ns: 1.0,
            scale_advdiff: 1.0,
            scale_smooth: 1.0,
            scale_bc: 1.0,
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
