//! P2 验收：物理损失逐项对上黄金参考数据（第 3 层物理残差、第 4 层对流扩散）。
//!
//! 第 1、2 层（差分算子、掩码几何）在 `golden.rs` 里已经对上了。**下层没对上时，
//! 上层的偏差没有诊断意义**，所以两个文件都必须绿。
//!
//! 这里比的全是标量损失值。标量最容易「看起来差不多」——散度残差 12.7 和 12.9
//! 肉眼没区别，但后者可能是把 dx 和 dy 用反了。所以判据是 1e-5 相对容差，不是
//! 「量级对得上」。

mod common;

use common::{assert_close4, assert_close_f64, open_npz, read, read_scalar, read_vec, tensor4, B};
use eser::ops::erode_mask;
use eser::physics::{
    denormalize, divergence, divergence_loss, navier_stokes_loss, reconstruction_loss, scalar,
    smoothness_loss, split_fields, ChannelOrder, Fields, PhysicsLoss, Scales, Task,
};
use eser::transport::{
    advection_diffusion_loss, advection_diffusion_residual, central_time_derivative,
};

// 与 export_golden.py 的 layer3_physics 一致
const PHYS_DX: f64 = 0.021;
const PHYS_DY: f64 = 0.043;
const PHYS_RHO: f64 = 1.3;
const PHYS_NU: f64 = 7e-3;

// 与 layer4_transport 一致
const TR_DX: f64 = 0.019;
const TR_DY: f64 = 0.031;
const TR_D: f64 = 3.2e-3;

// ---------------------------------------------------------------- 第 3 层

#[test]
fn denormalize_matches_reference() {
    // 物理项必须在有量纲场上算：逐通道缩放因子不同，在归一化空间求 div 是错的
    let mut npz = open_npz();
    let x = tensor4(&mut npz, "phys/fields_pvvt");
    let y_min = read_vec(&mut npz, "phys/y_min");
    let y_max = read_vec(&mut npz, "phys/y_max");
    assert_close4(
        "phys/denormalized",
        denormalize(x, Some(&y_min), Some(&y_max)),
        &read(&mut npz, "phys/denormalized"),
    );
}

#[test]
fn interior_mask_matches_reference() {
    // 这份掩码刻意贴着左右边界（地铁站站台层就是这样横贯整幅网格）
    let mut npz = open_npz();
    let mask = tensor4(&mut npz, "phys/mask");
    assert_close4(
        "phys/interior",
        erode_mask(mask, 1),
        &read(&mut npz, "phys/interior"),
    );
}

#[test]
fn divergence_and_losses_match_reference() {
    let mut npz = open_npz();
    let fields4 = tensor4(&mut npz, "phys/fields_pvvt");
    let interior = tensor4(&mut npz, "phys/interior");
    let f = split_fields(&fields4, ChannelOrder::Pvvt);

    assert_close4(
        "phys/divergence_field",
        divergence(
            f.vx.clone().unwrap(),
            f.vy.clone().unwrap(),
            PHYS_DX,
            PHYS_DY,
        ),
        &read(&mut npz, "phys/divergence_field"),
    );

    let div = divergence_loss(&f, Some(interior.clone()), PHYS_DX, PHYS_DY).unwrap();
    assert_close_f64(
        "phys/div_loss",
        scalar(div),
        read_scalar(&mut npz, "phys/div_loss"),
    );

    let ns = navier_stokes_loss(
        &f,
        Some(interior.clone()),
        PHYS_RHO,
        PHYS_NU,
        PHYS_DX,
        PHYS_DY,
    )
    .unwrap();
    assert_close_f64(
        "phys/ns_loss",
        scalar(ns),
        read_scalar(&mut npz, "phys/ns_loss"),
    );

    // 平滑项作用在全部 4 个通道上（温度通道不参与残差，但参与平滑）
    let smooth = smoothness_loss(fields4, Some(interior), PHYS_DX, PHYS_DY);
    assert_close_f64(
        "phys/smooth_loss",
        scalar(smooth),
        read_scalar(&mut npz, "phys/smooth_loss"),
    );
}

#[test]
fn missing_field_yields_none_not_wrong_number() {
    // 只喂 1 个通道时既没有 vy 也没有 p，两个损失都必须是 None
    let mut npz = open_npz();
    let fields4 = tensor4(&mut npz, "phys/fields_pvvt");
    let [b, _, h, w] = fields4.dims();
    let one = fields4.slice([0..b, 0..1, 0..h, 0..w]);
    let f = split_fields(&one, ChannelOrder::Pvvt);
    assert!(
        divergence_loss(&f, None, PHYS_DX, PHYS_DY).is_none(),
        "缺速度场时必须返回 None，而不是拿压力当速度算"
    );
    assert!(navier_stokes_loss(&f, None, 1.0, 1e-2, PHYS_DX, PHYS_DY).is_none());
}

#[test]
fn reconstruction_loss_matches_reference() {
    let mut npz = open_npz();
    let out = tensor4(&mut npz, "phys/fields_pvvt");
    let tgt = tensor4(&mut npz, "phys/target");
    let mask = tensor4(&mut npz, "phys/mask");

    assert_close_f64(
        "phys/recon_weighted",
        scalar(reconstruction_loss(
            out.clone(),
            tgt.clone(),
            Some(mask.clone()),
            true,
        )),
        read_scalar(&mut npz, "phys/recon_weighted"),
    );
    assert_close_f64(
        "phys/recon_plain",
        scalar(reconstruction_loss(out, tgt, Some(mask), false)),
        read_scalar(&mut npz, "phys/recon_plain"),
    );
}

/// 与 export_golden.py 里 `combined_flow_*` 那一组调用参数完全一致。
fn flow_loss(mask: burn::tensor::Tensor<B, 4>, y_min: Vec<f32>, y_max: Vec<f32>) -> PhysicsLoss<B> {
    PhysicsLoss {
        epoch: 37,
        total_epochs: 200,
        div_weight: 0.05,
        ns_weight: 0.005,
        smooth_weight: 0.005,
        channel_order: ChannelOrder::Pvvt,
        rho: PHYS_RHO,
        nu: PHYS_NU,
        dx: PHYS_DX,
        dy: PHYS_DY,
        physics_warmup: 0.1,
        scales: Scales {
            div: Some(12.0),
            ns: Some(3400.0),
            smooth: Some(7.0),
            advdiff: None,
        },
        mask: Some(mask),
        y_min: Some(y_min),
        y_max: Some(y_max),
        ..Default::default()
    }
}

#[test]
fn combined_flow_loss_matches_reference() {
    let mut npz = open_npz();
    let out = tensor4(&mut npz, "phys/fields_pvvt");
    let tgt = tensor4(&mut npz, "phys/target");
    let mask = tensor4(&mut npz, "phys/mask");
    let y_min = read_vec(&mut npz, "phys/y_min");
    let y_max = read_vec(&mut npz, "phys/y_max");

    let (total, parts) = flow_loss(mask, y_min, y_max).compute(out, tgt);

    assert_close_f64(
        "phys/combined_flow_total",
        scalar(total),
        read_scalar(&mut npz, "phys/combined_flow_total"),
    );
    for (name, got) in [
        ("base", parts.base),
        ("div", parts.div),
        ("ns", parts.ns),
        ("smooth", parts.smooth),
        ("w_div", parts.w_div),
        ("w_ns", parts.w_ns),
        ("w_smooth", parts.w_smooth),
    ] {
        let key = format!("phys/combined_flow_{}", name);
        let want = read_scalar(&mut npz, &key);
        assert_close_f64(&key, got, want);
    }
}

#[test]
fn zero_weight_really_skips_the_term() {
    // 地铁站方案 §3.5 去掉动量残差就是靠 ns_weight=0，必须真跳过而不是算完乘 0
    let mut npz = open_npz();
    let out = tensor4(&mut npz, "phys/fields_pvvt");
    let tgt = tensor4(&mut npz, "phys/target");
    let mask = tensor4(&mut npz, "phys/mask");
    let y_min = read_vec(&mut npz, "phys/y_min");
    let y_max = read_vec(&mut npz, "phys/y_max");

    let cfg = PhysicsLoss {
        epoch: 37,
        total_epochs: 200,
        ns_weight: 0.0,
        channel_order: ChannelOrder::Pvvt,
        dx: PHYS_DX,
        dy: PHYS_DY,
        mask: Some(mask),
        y_min: Some(y_min),
        y_max: Some(y_max),
        ..Default::default()
    };
    let (_, parts) = cfg.compute(out, tgt);

    assert_close_f64(
        "phys/nsweight0_ns",
        parts.ns,
        read_scalar(&mut npz, "phys/nsweight0_ns"),
    );
    assert_eq!(parts.ns, 0.0, "ns_weight=0 时动量残差应恰好为 0");
    assert_close_f64(
        "phys/nsweight0_div",
        parts.div,
        read_scalar(&mut npz, "phys/nsweight0_div"),
    );
    assert!(parts.div > 0.0, "关掉动量项不应连散度项一起关掉");
}

#[test]
fn physics_report_matches_reference() {
    let mut npz = open_npz();
    let out = tensor4(&mut npz, "phys/fields_pvvt");
    let mask = tensor4(&mut npz, "phys/mask");
    let y_min = read_vec(&mut npz, "phys/y_min");
    let y_max = read_vec(&mut npz, "phys/y_max");

    let rep = PhysicsLoss {
        channel_order: ChannelOrder::Pvvt,
        rho: PHYS_RHO,
        nu: PHYS_NU,
        dx: PHYS_DX,
        dy: PHYS_DY,
        mask: Some(mask),
        y_min: Some(y_min),
        y_max: Some(y_max),
        ..Default::default()
    }
    .report(out);

    assert_close_f64(
        "phys/report_div_rms",
        rep.div_rms.expect("流场任务必须报散度残差"),
        read_scalar(&mut npz, "phys/report_div_rms"),
    );
    assert_close_f64(
        "phys/report_ns_rms",
        rep.ns_rms.expect("流场任务必须报动量残差"),
        read_scalar(&mut npz, "phys/report_ns_rms"),
    );
    assert_eq!(rep.advdiff_rms, None, "流场任务不该报对流扩散残差");
}

// ---------------------------------------------------------------- 第 4 层

/// 读第 4 层共用的浓度场与速度场。
fn transport_inputs(
    npz: &mut ndarray_npy::NpzReader<std::fs::File>,
) -> (
    burn::tensor::Tensor<B, 4>,
    burn::tensor::Tensor<B, 4>,
    burn::tensor::Tensor<B, 4>,
) {
    (
        tensor4(npz, "trans/c"),
        tensor4(npz, "trans/vx"),
        tensor4(npz, "trans/vy"),
    )
}

#[test]
fn advection_diffusion_residual_matches_reference() {
    let mut npz = open_npz();
    let (c, vx, vy) = transport_inputs(&mut npz);
    let dcdt = tensor4(&mut npz, "trans/dcdt");
    let source = tensor4(&mut npz, "trans/source");

    assert_close4(
        "trans/residual_steady",
        advection_diffusion_residual(
            c.clone(),
            vx.clone(),
            vy.clone(),
            TR_D,
            TR_DX,
            TR_DY,
            None,
            None,
        ),
        &read(&mut npz, "trans/residual_steady"),
    );
    assert_close4(
        "trans/residual_transient",
        advection_diffusion_residual(
            c.clone(),
            vx.clone(),
            vy.clone(),
            TR_D,
            TR_DX,
            TR_DY,
            Some(dcdt.clone()),
            None,
        ),
        &read(&mut npz, "trans/residual_transient"),
    );
    assert_close4(
        "trans/residual_with_source",
        advection_diffusion_residual(c, vx, vy, TR_D, TR_DX, TR_DY, Some(dcdt), Some(source)),
        &read(&mut npz, "trans/residual_with_source"),
    );
}

#[test]
fn advection_diffusion_loss_matches_reference() {
    let mut npz = open_npz();
    let (c, vx, vy) = transport_inputs(&mut npz);
    let dcdt = tensor4(&mut npz, "trans/dcdt");
    let n = c.dims()[2];
    let interior = erode_mask(
        burn::tensor::Tensor::<B, 4>::ones([1, 1, n, n], &Default::default()),
        1,
    );

    let fields = Fields {
        c: Some(c),
        vx: Some(vx),
        vy: Some(vy),
        p: None,
    };
    let loss = advection_diffusion_loss(
        &fields,
        Some(interior),
        TR_D,
        TR_DX,
        TR_DY,
        Some(dcdt),
        None,
    )
    .expect("三个场齐全时不该返回 None");
    assert_close_f64(
        "trans/loss",
        scalar(loss),
        read_scalar(&mut npz, "trans/loss"),
    );

    // 缺速度场时必须返回 None，而不是静默算错
    let only_c = Fields {
        c: fields.c.clone(),
        ..Default::default()
    };
    assert!(
        advection_diffusion_loss(&only_c, None, TR_D, TR_DX, TR_DY, None, None).is_none(),
        "缺速度场时必须返回 None"
    );
}

#[test]
fn central_time_derivative_matches_reference() {
    let mut npz = open_npz();
    let c = tensor4(&mut npz, "trans/c");
    let got =
        central_time_derivative(c.clone(), c.mul_scalar(1.1), 0.25).expect("dt 为正时不该报错");
    assert_close4("trans/central_dt", got, &read(&mut npz, "trans/central_dt"));

    let c2 = tensor4(&mut npz, "trans/c");
    assert!(
        central_time_derivative(c2.clone(), c2, 0.0).is_err(),
        "dt<=0 必须被拒绝——分母是 2Δt，传 0 会得到 inf 而不是报错"
    );
}

#[test]
fn combined_dispersion_loss_matches_reference() {
    let mut npz = open_npz();
    let (c, vx, vy) = transport_inputs(&mut npz);
    let tgt = tensor4(&mut npz, "trans/target");
    let n = c.dims()[2];
    let mask = burn::tensor::Tensor::<B, 4>::ones([1, 1, n, n], &Default::default());

    let (total, parts) = PhysicsLoss {
        epoch: 90,
        total_epochs: 200,
        channel_order: ChannelOrder::C,
        task: Task::Dispersion,
        velocity: Some((vx, vy)),
        diffusivity: TR_D,
        advdiff_weight: 0.08,
        y_min: Some(vec![0.0]),
        y_max: Some(vec![5.0]),
        dx: TR_DX,
        dy: TR_DY,
        mask: Some(mask),
        ..Default::default()
    }
    .compute(c, tgt);

    assert_close_f64(
        "trans/combined_total",
        scalar(total),
        read_scalar(&mut npz, "trans/combined_total"),
    );
    for (name, got) in [
        ("base", parts.base),
        ("advdiff", parts.advdiff),
        ("smooth", parts.smooth),
        ("div", parts.div),
        ("ns", parts.ns),
    ] {
        let key = format!("trans/combined_{}", name);
        let want = read_scalar(&mut npz, &key);
        assert_close_f64(&key, got, want);
    }
    assert_eq!(
        (parts.div, parts.ns),
        (0.0, 0.0),
        "dispersion 任务不该再算散度/动量项"
    );
    assert!(
        parts.advdiff > 0.0,
        "dispersion 任务必须真的算了对流扩散残差"
    );
}

#[test]
fn dispersion_without_velocity_stays_zero() {
    // 速度场缺失时残差归零，而不是拿浓度输出当速度乱算
    let mut npz = open_npz();
    let c = tensor4(&mut npz, "trans/c");
    let tgt = tensor4(&mut npz, "trans/target");
    let (_, parts) = PhysicsLoss {
        channel_order: ChannelOrder::C,
        task: Task::Dispersion,
        velocity: None,
        advdiff_weight: 0.1,
        dx: TR_DX,
        dy: TR_DY,
        ..Default::default()
    }
    .compute(c, tgt);
    assert_eq!(parts.advdiff, 0.0, "缺速度场时必须归零，不能静默算错");
}
