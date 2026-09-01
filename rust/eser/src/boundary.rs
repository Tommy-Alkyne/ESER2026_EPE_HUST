//! 边界条件残差：把「墙」「送风口」「排风口」从掩码里区分出来，并作为**真正的
//! 约束项**进损失，而不只是换个权重。
//!
//! ## 为什么需要它
//!
//! 旧的 0/1 掩码里，固壁和送风口同为 `0`——两者在损失看来完全一样。而物理残差
//! 又要先把掩码腐蚀一圈（`erode_mask`，否则差分模板会取到墙外的值），结果是
//! **边界上一条约束都没有**。网络对「风机开多大」没有任何物理依据，只能靠数据
//! 硬记，一换工况就外推失效——这正是方案 §7.4 的一个直接根源。
//!
//! ## 法向怎么算
//!
//! 某类边界 Z 在贴边流体格点上的**外法向**取 `∇(ind_Z) / |∇(ind_Z)|`，其中
//! `ind_Z` 是该类边界的 0/1 指示场。`∇` 从流体（0）指向边界（1），方向天然朝外，
//! 不用再取负号。
//!
//! **逐类分别算，不是共用一个法向**：一个流体格点可能一侧贴墙、另一侧贴风口，
//! 共用法向会把两个约束都加在错误的方向上，而且不会报错。
//!
//! ## 贴边格点怎么定
//!
//! 就用 `|∇(ind_Z)| > 0`：梯度非零当且仅当该点的**三点差分模板**碰到了 Z 类边界。
//! 这与微分算子用的模板严格一致——另写一套膨胀（dilate）判据反而会引入
//! 「判定说贴边、但算子其实够不着」的错位。斜对角相邻因此不算贴边，那种位置本来
//! 也定义不出法向。

use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Tensor, TensorData};

use crate::geometry::{zone_indicator, Zone};
use crate::ops::{grad_xy, masked_mean};

/// 一类边界在贴边流体格点上的几何信息。全部是**常量张量**，不参与求导。
#[derive(Debug, Clone)]
pub struct BoundaryZone<B: Backend> {
    /// 贴着这类边界的流体格点，0/1，形状 (1, 1, H, W)
    pub adj: Tensor<B, 4>,
    /// 外法向 x 分量（`adj` 之外的取值无意义，但保证是有限数）
    pub nx: Tensor<B, 4>,
    /// 外法向 y 分量
    pub ny: Tensor<B, 4>,
    /// 贴边格点数，用于启动自检打印
    pub count: usize,
}

/// 三类边界的几何。某类一个格点都没有时对应字段是 `None`。
#[derive(Debug, Clone)]
pub struct Boundary<B: Backend> {
    pub wall: Option<BoundaryZone<B>>,
    pub inlet: Option<BoundaryZone<B>>,
    pub outlet: Option<BoundaryZone<B>>,
}

impl<B: Backend> Boundary<B> {
    /// 从分区标签场预计算三类边界的贴边格点与法向。
    ///
    /// 返回 `None` 表示这张掩码里**一个边界格点都没有**（例如只含 0/1 的旧文件）。
    /// 返回 `None` 而不是三个空 `BoundaryZone`：上层据此打印「这份几何给不了边界
    /// 条件」，否则边界项会静默地一条都加不上，却看起来配置成功了。
    pub fn build(
        zones: &[f32],
        h: usize,
        w: usize,
        dx: f64,
        dy: f64,
        device: &B::Device,
    ) -> Option<Self> {
        let fluid = to_tensor::<B>(&zone_indicator(zones, Zone::Fluid), h, w, device);
        let one = |z: Zone| {
            BoundaryZone::build(
                to_tensor::<B>(&zone_indicator(zones, z), h, w, device),
                fluid.clone(),
                dx,
                dy,
            )
        };
        let b = Self {
            wall: one(Zone::Wall),
            inlet: one(Zone::Inlet),
            outlet: one(Zone::Outlet),
        };
        (b.wall.is_some() || b.inlet.is_some() || b.outlet.is_some()).then_some(b)
    }
}

impl<B: AutodiffBackend> Boundary<B> {
    /// 换到内层（非 autodiff）后端。诊断路径要用——不带过去的话 `bc_rms` 会恒为
    /// `None`，看起来像「没有边界」，而实际上约束正开着。
    pub fn inner(&self) -> Boundary<B::InnerBackend> {
        Boundary {
            wall: self.wall.as_ref().map(BoundaryZone::inner),
            inlet: self.inlet.as_ref().map(BoundaryZone::inner),
            outlet: self.outlet.as_ref().map(BoundaryZone::inner),
        }
    }
}

impl<B: AutodiffBackend> BoundaryZone<B> {
    fn inner(&self) -> BoundaryZone<B::InnerBackend> {
        BoundaryZone {
            adj: self.adj.clone().inner(),
            nx: self.nx.clone().inner(),
            ny: self.ny.clone().inner(),
            count: self.count,
        }
    }
}

impl<B: Backend> BoundaryZone<B> {
    fn build(indicator: Tensor<B, 4>, fluid: Tensor<B, 4>, dx: f64, dy: f64) -> Option<Self> {
        let (gx, gy) = grad_xy(indicator, dx, dy);
        let mag = (gx.clone() * gx.clone() + gy.clone() * gy.clone()).sqrt();

        // 非零 |∇ind| 的下界是 0.5 / max(dx, dy)（单侧三点差分），取其一半当阈值：
        // 稳稳落在 0 与最小非零值之间，与网格各向异性无关。
        let thresh = 0.25 / dx.max(dy);
        let adj = mag.clone().greater_elem(thresh).float() * fluid;
        let count = adj.clone().sum().into_scalar().elem::<f64>().round() as usize;
        if count == 0 {
            return None;
        }

        // **必须夹住**：域内部 mag 恒为 0，不夹就会出现 0/0 = NaN。而 NaN 乘以
        // 掩码的 0 仍是 NaN，会顺着 masked_mean 的求和污染整个损失——静默地。
        let safe = mag.clamp_min(1e-12);
        Some(Self {
            nx: gx / safe.clone(),
            ny: gy / safe,
            adj,
            count,
        })
    }
}

fn to_tensor<B: Backend>(v: &[f32], h: usize, w: usize, device: &B::Device) -> Tensor<B, 4> {
    Tensor::<B, 1>::from_data(TensorData::from(v), device).reshape([1, 1, h, w])
}

/// 法向速度残差 `u·n - target`，只在贴边流体格点上统计。
///
/// 三类边界共用这一个式子，只是 `target` 不同：
/// * 固壁 `target = 0` —— 不可穿透
/// * 送风口 `target = -q` —— 以速度 q 向域内送风（外法向朝外，进域即为负）
/// * 排风口 `target = +q` —— 以速度 q 抽出
///
/// **只约束法向分量，不约束切向**。风口的切向速度实际上是未知的；硬给一个
/// `(vx, vy)` 向量等于凭空编造切向分量，而且要求调用方知道每个风口的朝向。
pub fn normal_velocity_loss<B: Backend>(
    vx: Tensor<B, 4>,
    vy: Tensor<B, 4>,
    zone: &BoundaryZone<B>,
    target: f64,
) -> Tensor<B, 1> {
    let un = vx * zone.nx.clone() + vy * zone.ny.clone();
    let r = un.sub_scalar(target);
    masked_mean(r.clone() * r, Some(zone.adj.clone()))
}

/// 标量无通量残差 `∂C/∂n = ∇C·n`，只在贴边流体格点上统计。固壁上应为 0。
pub fn normal_gradient_loss<B: Backend>(
    c: Tensor<B, 4>,
    zone: &BoundaryZone<B>,
    dx: f64,
    dy: f64,
) -> Tensor<B, 1> {
    let (gx, gy) = grad_xy(c, dx, dy);
    let r = gx * zone.nx.clone() + gy * zone.ny.clone();
    masked_mean(r.clone() * r, Some(zone.adj.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    /// 左半流体、右半墙的竖直界面，外法向应当处处是 +x。
    fn vertical_wall() -> (Vec<f32>, usize, usize) {
        let (h, w) = (5usize, 6usize);
        let mut z = vec![Zone::Fluid.label(); h * w];
        for y in 0..h {
            for x in 3..w {
                z[y * w + x] = Zone::Wall.label();
            }
        }
        (z, h, w)
    }

    #[test]
    fn normal_points_out_of_the_fluid() {
        let (z, h, w) = vertical_wall();
        let d = Default::default();
        let b = Boundary::<B>::build(&z, h, w, 1.0, 1.0, &d).expect("应当认出墙");
        let wall = b.wall.expect("墙分区应当非空");

        // 贴边流体格点是 x = 2 那一列，共 h 个
        assert_eq!(wall.count, h, "贴边格点应当是紧邻墙的一列");

        let nx = wall.nx.to_data().to_vec::<f32>().unwrap();
        let ny = wall.ny.to_data().to_vec::<f32>().unwrap();
        for y in 0..h {
            let i = y * w + 2;
            assert!((nx[i] - 1.0).abs() < 1e-5, "外法向应是 +x，实得 {}", nx[i]);
            assert!(ny[i].abs() < 1e-5, "竖直界面上 ny 应为 0，实得 {}", ny[i]);
        }
        assert!(
            nx.iter().chain(ny.iter()).all(|v| v.is_finite()),
            "域内部的法向必须是有限数，否则 NaN 会污染整个损失"
        );
    }

    #[test]
    fn impermeability_is_zero_for_tangential_flow_and_positive_for_normal_flow() {
        let (z, h, w) = vertical_wall();
        let d = Default::default();
        let wall = Boundary::<B>::build(&z, h, w, 1.0, 1.0, &d)
            .unwrap()
            .wall
            .unwrap();

        // 纯切向流（只有 vy）：u·n = 0，残差必须是 0
        let zero: Tensor<B, 4> = Tensor::zeros([1, 1, h, w], &d);
        let ones: Tensor<B, 4> = Tensor::ones([1, 1, h, w], &d);
        let tangential = normal_velocity_loss(zero.clone(), ones.clone(), &wall, 0.0);
        assert!(
            tangential.into_scalar() < 1e-10,
            "平行于墙的流动不应产生不可穿透残差"
        );

        // 纯法向流（只有 vx = 1）：u·n = 1，残差均方必须是 1
        let normal = normal_velocity_loss(ones, zero, &wall, 0.0);
        let got = normal.into_scalar() as f64;
        assert!((got - 1.0).abs() < 1e-5, "垂直穿墙的残差应为 1，实得 {}", got);
    }

    #[test]
    fn inlet_target_is_satisfied_by_inflow_at_that_speed() {
        let (h, w) = (5usize, 6usize);
        let mut z = vec![Zone::Fluid.label(); h * w];
        for y in 0..h {
            for x in 3..w {
                z[y * w + x] = Zone::Inlet.label();
            }
        }
        let d = Default::default();
        let inlet = Boundary::<B>::build(&z, h, w, 1.0, 1.0, &d)
            .unwrap()
            .inlet
            .unwrap();

        // 外法向是 +x；以 2 m/s 向域内送风即 vx = -2，u·n = -2 = target
        let vx: Tensor<B, 4> = Tensor::<B, 4>::ones([1, 1, h, w], &d).mul_scalar(-2.0);
        let vy: Tensor<B, 4> = Tensor::zeros([1, 1, h, w], &d);
        let good = normal_velocity_loss(vx.clone(), vy.clone(), &inlet, -2.0).into_scalar() as f64;
        assert!(good < 1e-9, "以给定风速送风时残差应为 0，实得 {}", good);

        // 方向搞反（往外吹）：u·n = +2，与 target -2 差 4，均方 16
        let bad = normal_velocity_loss(vx.neg(), vy, &inlet, -2.0).into_scalar() as f64;
        assert!((bad - 16.0).abs() < 1e-4, "反向送风的残差应为 16，实得 {}", bad);
    }

    #[test]
    fn no_flux_is_zero_for_a_field_constant_along_the_normal() {
        let (z, h, w) = vertical_wall();
        let d = Default::default();
        let wall = Boundary::<B>::build(&z, h, w, 1.0, 1.0, &d)
            .unwrap()
            .wall
            .unwrap();

        // C 只沿 y 变化：∂C/∂n = ∂C/∂x = 0
        let vals: Vec<f32> = (0..h * w).map(|i| (i / w) as f32).collect();
        let c = to_tensor::<B>(&vals, h, w, &d);
        let ok = normal_gradient_loss(c, &wall, 1.0, 1.0).into_scalar() as f64;
        assert!(ok < 1e-10, "沿法向不变的场无通量残差应为 0，实得 {}", ok);

        // C 沿 x 线性增长：∂C/∂n = 1，均方 1
        let vals: Vec<f32> = (0..h * w).map(|i| (i % w) as f32).collect();
        let c = to_tensor::<B>(&vals, h, w, &d);
        let bad = normal_gradient_loss(c, &wall, 1.0, 1.0).into_scalar() as f64;
        assert!((bad - 1.0).abs() < 1e-5, "沿法向线性变化应给出 1，实得 {}", bad);
    }

    #[test]
    fn pure_binary_mask_yields_no_boundary_at_all() {
        // 旧的 0/1 掩码里没有任何边界标签，必须返回 None 而不是三个空分区
        let z = vec![1.0f32; 9];
        assert!(Boundary::<B>::build(&z, 3, 3, 1.0, 1.0, &Default::default()).is_none());
    }
}
