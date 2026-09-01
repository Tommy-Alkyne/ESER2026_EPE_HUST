//! 三维物理残差的解析验证。
//!
//! **每条判据都必须能区分二维和三维**——这是本文件的核心要求。用一个「二维算也对、
//! 三维算也对」的场去测，等于什么都没测：z 项接没接上，结果都一样。所以下面每个
//! 场都刻意造成「按二维算得到明显错误的数、按三维算才对」。
//!
//! Python 版没有三维物理，`golden.npz` 里也就没有可对照的参考张量，所以这里靠解析解。

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

use eser::physics::{
    divergence_loss, navier_stokes_residual, scalar, ChannelOrder, Fields, PhysicsLoss, Task, ZAxis,
};
use eser::transport::advection_diffusion_loss;

type B = NdArray<f32>;

const D: usize = 7;
const H: usize = 5;
const W: usize = 6;
const DX: f64 = 0.2;
const DY: f64 = 0.25;
const DZ: f64 = 0.1;

/// 造一个展平成 (D, 1, H, W) 的场，取值由物理坐标 `(z, y, x)` 给出。
///
/// batch 取 1，于是展平后的第 0 维就是深度维，与 `flatten_depth` 的 `b*D + d` 一致。
fn field(f: impl Fn(f64, f64, f64) -> f64) -> Tensor<B, 4> {
    let mut v = Vec::with_capacity(D * H * W);
    for k in 0..D {
        for j in 0..H {
            for i in 0..W {
                v.push(f(k as f64 * DZ, j as f64 * DY, i as f64 * DX) as f32);
            }
        }
    }
    Tensor::<B, 1>::from_data(TensorData::from(v.as_slice()), &Default::default())
        .reshape([D, 1, H, W])
}

fn zaxis() -> ZAxis {
    ZAxis { b: 1, d: D, dz: DZ }
}

/// 只统计内部格点：边界层的 replicate padding 让差分退化成单侧，那不是 z 项写错。
fn interior() -> Option<Tensor<B, 4>> {
    let mut v = vec![0.0f32; D * H * W];
    for k in 1..D - 1 {
        for j in 1..H - 1 {
            for i in 1..W - 1 {
                v[(k * H + j) * W + i] = 1.0;
            }
        }
    }
    Some(
        Tensor::<B, 1>::from_data(TensorData::from(v.as_slice()), &Default::default())
            .reshape([D, 1, H, W]),
    )
}

/// `u = x, v = 0, w = -z`：三维散度恒为 0，二维散度恒为 1。
///
/// 这就是那个「二维会给出明显错误答案」的场。少算 `∂w/∂z` 不会报任何错，
/// 只会让网络学出一个二维意义上无散、三维意义上违反质量守恒的场。
#[test]
fn three_d_divergence_needs_the_dw_dz_term() {
    let f = Fields::<B> {
        vx: Some(field(|_z, _y, x| x)),
        vy: Some(field(|_, _, _| 0.0)),
        vz: Some(field(|z, _, _| -z)),
        ..Default::default()
    };

    let three_d = scalar(divergence_loss(&f, interior(), DX, DY, Some(zaxis())).unwrap());
    let two_d = scalar(divergence_loss(&f, interior(), DX, DY, None).unwrap());

    assert!(
        three_d < 1e-8,
        "三维散度应当恒为 0（∂u/∂x + ∂w/∂z = 1 - 1），实得均方 {:.3e}",
        three_d
    );
    assert!(
        (two_d - 1.0).abs() < 1e-4,
        "同一个场按二维算应当给出 1（漏掉 ∂w/∂z），实得 {:.6}。\
         两者若相等说明 z 项根本没接上",
        two_d
    );
}

/// `C = z`，`w = 1`，无源无时变：三维残差是 1（`w·∂C/∂z`），二维是 0。
///
/// 方向反过来了：这次是**二维给 0、三维给非 0**。两个方向各测一次，
/// 免得实现里把 z 项加成了恒等于 0 的东西也照样「通过」。
#[test]
fn three_d_advection_carries_the_vertical_gradient() {
    let f = Fields::<B> {
        c: Some(field(|z, _, _| z)),
        vx: Some(field(|_, _, _| 0.0)),
        vy: Some(field(|_, _, _| 0.0)),
        vz: Some(field(|_, _, _| 1.0)),
        ..Default::default()
    };

    let three_d = scalar(
        advection_diffusion_loss(&f, interior(), 0.0, DX, DY, None, None, Some(zaxis())).unwrap(),
    );
    let two_d =
        scalar(advection_diffusion_loss(&f, interior(), 0.0, DX, DY, None, None, None).unwrap());

    assert!(
        (three_d - 1.0).abs() < 1e-4,
        "w·∂C/∂z = 1，残差均方应当是 1，实得 {:.6}",
        three_d
    );
    assert!(
        two_d < 1e-8,
        "同一个场按二维算看不到任何竖向输运，应当是 0，实得 {:.3e}",
        two_d
    );
}

/// 竖向扩散：`C = z²` 时 `∂²C/∂z² = 2`，残差应当是 `-D·2`。
#[test]
fn three_d_diffusion_sees_the_vertical_curvature() {
    let f = Fields::<B> {
        c: Some(field(|z, _, _| z * z)),
        vx: Some(field(|_, _, _| 0.0)),
        vy: Some(field(|_, _, _| 0.0)),
        vz: Some(field(|_, _, _| 0.0)),
        ..Default::default()
    };
    let d = 0.5;
    let got = scalar(
        advection_diffusion_loss(&f, interior(), d, DX, DY, None, None, Some(zaxis())).unwrap(),
    );
    let want = (d * 2.0) * (d * 2.0);
    assert!(
        (got - want).abs() < 1e-3,
        "竖向扩散残差均方应当是 (D·∂²C/∂z²)² = {:.4}，实得 {:.4}",
        want,
        got
    );
}

/// 三维必须多出第三个动量方程，而且它真的含 `∂p/∂z`。
#[test]
fn three_d_momentum_has_a_z_component_driven_by_pressure() {
    let zero = || field(|_, _, _| 0.0);
    // 静止流体 + 沿 z 线性变化的压力：mom_z = (1/ρ)·∂p/∂z，其余两个分量为 0
    let f = Fields::<B> {
        p: Some(field(|z, _, _| 3.0 * z)),
        vx: Some(zero()),
        vy: Some(zero()),
        vz: Some(zero()),
        ..Default::default()
    };

    let moms = navier_stokes_residual(&f, 1.0, 0.0, DX, DY, Some(zaxis())).unwrap();
    assert_eq!(moms.len(), 3, "三维应当给出三个动量分量");

    let m = interior().unwrap();
    let pick = |t: Tensor<B, 4>| {
        let s = (t.clone() * m.clone()).sum().into_scalar() as f64;
        s / m.clone().sum().into_scalar() as f64
    };
    assert!(pick(moms[0].clone()).abs() < 1e-6, "静止流体的 mom_x 应为 0");
    assert!(pick(moms[1].clone()).abs() < 1e-6, "静止流体的 mom_y 应为 0");
    assert!(
        (pick(moms[2].clone()) - 3.0).abs() < 1e-4,
        "mom_z 应当等于 ∂p/∂z = 3，实得 {:.6}。为 0 说明压力的 z 梯度没算",
        pick(moms[2].clone())
    );

    // 二维路径不该凭空多出一个分量
    let two = navier_stokes_residual(&f, 1.0, 0.0, DX, DY, None).unwrap();
    assert_eq!(two.len(), 2, "二维只有 x / y 两个动量分量");
}

/// 没有 `vz` 时（二维通道口径）z 项一概不算，而不是按 0 算。
#[test]
fn missing_vz_falls_back_to_the_two_d_path() {
    let f = Fields::<B> {
        vx: Some(field(|_z, _y, x| x)),
        vy: Some(field(|_, _, _| 0.0)),
        vz: None,
        ..Default::default()
    };
    let with_z = scalar(divergence_loss(&f, interior(), DX, DY, Some(zaxis())).unwrap());
    let no_z = scalar(divergence_loss(&f, interior(), DX, DY, None).unwrap());
    assert!(
        (with_z - no_z).abs() < 1e-9,
        "通道口径里没有 Vz 时，给不给 ZAxis 结果都该一样：{} vs {}",
        with_z,
        no_z
    );
}

/// 深度不足 3 层时必须报错，而不是算出一个没有意义的数。
#[test]
fn too_few_layers_is_an_error_not_a_wrong_number() {
    for d in [1usize, 2] {
        let z = ZAxis { b: 1, d, dz: 0.1 };
        let e = z.check().unwrap_err();
        assert!(e.contains("3 层") || e.contains("层"), "报错要说清原因：{}", e);
    }
    assert!(ZAxis { b: 1, d: 3, dz: 0.1 }.check().is_ok());
    assert!(
        ZAxis { b: 1, d: 5, dz: 0.0 }.check().is_err(),
        "dz = 0 会让所有 z 项变成除零"
    );
}

/// 逐层掩码必须对上各自那一层。
///
/// `compute_3d` 把 (D,1,H,W) 的掩码复制 B 份对齐 (B*D,C,H,W)。写成「层内复制」
/// （每层重复 B 次）不会报错，只会让每个样本配到错误的那一层掩码——而如果掩码
/// 各层相同，这个错永远测不出来。所以这里的掩码**每层都不一样**。
#[test]
fn per_layer_mask_lines_up_with_its_own_layer() {
    let device = Default::default();
    let (b, c) = (3usize, 1usize);

    // 掩码：只有第 2 层是全 1，其余全 0
    let keep = 2usize;
    let mut mv = vec![0.0f32; D * H * W];
    for j in 0..H {
        for i in 0..W {
            mv[(keep * H + j) * W + i] = 1.0;
        }
    }
    let mask = Tensor::<B, 1>::from_data(TensorData::from(mv.as_slice()), &device)
        .reshape([D, 1, H, W]);

    // 输出：第 k 层恒等于 k，target 全 0。数据项 = 掩码选中那一层的 k²
    let mut ov = Vec::with_capacity(b * c * D * H * W);
    for _ in 0..b {
        for _ in 0..c {
            for k in 0..D {
                for _ in 0..H * W {
                    ov.push(k as f32);
                }
            }
        }
    }
    let out = Tensor::<B, 1>::from_data(TensorData::from(ov.as_slice()), &device)
        .reshape([b, c, D, H, W]);
    let target = Tensor::<B, 5>::zeros([b, c, D, H, W], &device);

    let loss = PhysicsLoss::<B> {
        task: Task::Flow,
        channel_order: ChannelOrder::Pvv,
        div_weight: 0.0,
        ns_weight: 0.0,
        smooth_weight: 0.0,
        bc_weight: 0.0,
        physics_warmup: 0.0,
        dx: DX,
        dy: DY,
        dz: DZ,
        mask: Some(mask),
        ..Default::default()
    };

    let (_, parts) = loss.compute_3d(out, target).expect("三维损失应当算得出来");
    // reconstruction_loss 是 0.6·MSE + 0.4·L1，误差恒为 keep
    let want = 0.6 * (keep * keep) as f64 + 0.4 * keep as f64;
    assert!(
        (parts.base - want).abs() < 1e-4,
        "掩码只留第 {} 层，数据项应当是 {:.3}，实得 {:.3}。\
         对不上说明掩码没对到它自己那一层",
        keep,
        want,
        parts.base
    );
}

/// `--ndim 3` 但通道口径是二维的（P, Vx, Vy）：z 项本来就不参与，
/// 层数只有 2 层也照样能跑，行为与加三维算子之前完全一致。
#[test]
fn three_d_network_with_two_d_channels_still_works_on_two_layers() {
    let device = Default::default();
    let (b, c, d, h, w) = (2usize, 3usize, 2usize, 4usize, 4usize);
    let out = Tensor::<B, 5>::ones([b, c, d, h, w], &device);
    let target = Tensor::<B, 5>::zeros([b, c, d, h, w], &device);

    let loss = PhysicsLoss::<B> {
        task: Task::Flow,
        channel_order: ChannelOrder::Pvv, // 没有 Vz
        dx: DX,
        dy: DY,
        dz: DZ,
        physics_warmup: 0.0,
        ..Default::default()
    };
    assert!(
        loss.compute_3d(out, target).is_ok(),
        "二维通道口径下不该因为层数少而拒绝训练——z 项本来就不算"
    );
}

/// 但真要算 z 项（通道口径带 Vz）却只给 2 层，必须报错。
#[test]
fn real_three_d_physics_on_two_layers_is_rejected() {
    let device = Default::default();
    let (b, c, d, h, w) = (2usize, 4usize, 2usize, 4usize, 4usize);
    let out = Tensor::<B, 5>::ones([b, c, d, h, w], &device);
    let target = Tensor::<B, 5>::zeros([b, c, d, h, w], &device);

    let loss = PhysicsLoss::<B> {
        task: Task::Flow,
        channel_order: ChannelOrder::Pvvv, // 有 Vz
        dx: DX,
        dy: DY,
        dz: DZ,
        physics_warmup: 0.0,
        ..Default::default()
    };
    let e = loss
        .compute_3d(out, target)
        .expect_err("2 层算不出 ∂/∂z，必须报错而不是给个没意义的数");
    assert!(e.contains("层"), "报错要说清是层数不够：{}", e);
}

/// 走完整的 `compute_3d`，确认 z 项**真的进了损失**。
///
/// 补的是变异验证发现的一个洞：把 `resolve_z` 改成永远返回 `None`（三维物理静默
/// 关闭）时，前面所有测试都还是绿的——因为它们都直接传 `Some(ZAxis)`，绕过了
/// 那个决定「这次到底算不算 z 项」的函数。而静默关闭正是这一整块最要命的失效模式：
/// 日志、曲线、收敛全都正常，只是三维约束一条都没加。
#[test]
fn compute_3d_really_turns_on_the_z_terms() {
    let device = Default::default();
    let (b, d, h, w) = (2usize, 5usize, 4usize, 4usize);

    // 输出四通道 (P, Vx, Vy, Vz)：前三个恒为 0，Vz = z。
    // 于是二维散度恒为 0，三维散度恒为 ∂w/∂z = 1。
    let mut v = Vec::with_capacity(b * 4 * d * h * w);
    for _ in 0..b {
        for ch in 0..4 {
            for k in 0..d {
                for _ in 0..h * w {
                    v.push(if ch == 3 { (k as f64 * DZ) as f32 } else { 0.0 });
                }
            }
        }
    }
    let out = Tensor::<B, 1>::from_data(TensorData::from(v.as_slice()), &device)
        .reshape([b, 4, d, h, w]);
    let target = Tensor::<B, 5>::zeros([b, 4, d, h, w], &device);

    // 掩码留内部层，避开 replicate 边界；erode_mask 还会再收一圈 x/y
    let mut mv = vec![0.0f32; d * h * w];
    for k in 1..d - 1 {
        for j in 0..h {
            for i in 0..w {
                mv[(k * h + j) * w + i] = 1.0;
            }
        }
    }
    let mask = Tensor::<B, 1>::from_data(TensorData::from(mv.as_slice()), &device)
        .reshape([d, 1, h, w]);

    let base = PhysicsLoss::<B> {
        task: Task::Flow,
        div_weight: 1.0,
        ns_weight: 0.0,
        smooth_weight: 0.0,
        bc_weight: 0.0,
        physics_warmup: 0.0,
        dx: DX,
        dy: DY,
        dz: DZ,
        mask: Some(mask),
        ..Default::default()
    };

    let (_, three_d) = PhysicsLoss::<B> {
        channel_order: ChannelOrder::Pvvv,
        ..base.clone()
    }
    .compute_3d(out.clone(), target.clone())
    .expect("三维应当算得出来");

    // 同一份数据按二维通道口径（没有 Vz）：散度应当恒为 0
    let (_, two_d) = PhysicsLoss::<B> {
        channel_order: ChannelOrder::Pvv,
        ..base
    }
    .compute_3d(out, target)
    .expect("二维口径也应当算得出来");

    assert!(
        two_d.div < 1e-8,
        "二维口径看不到竖向速度，散度应为 0，实得 {:.3e}",
        two_d.div
    );
    assert!(
        (three_d.div - 1.0).abs() < 1e-3,
        "三维散度应当是 (∂w/∂z)² = 1，实得 {:.6}。为 0 说明 z 项没进损失",
        three_d.div
    );
}
