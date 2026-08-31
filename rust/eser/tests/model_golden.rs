//! P3 验收：固定权重下的网络前向逐元素对上黄金数据（第 6 层）。
//!
//! **权重由参考数据给定，不是自己初始化的**——两个框架的随机初始化不可能一致，
//! 所以只有喂同一份权重、比同一个输入的输出，才谈得上「前向一致」。
//!
//! 这一项对上，说明下面这些全都对：DenseLayer 的拼接顺序、Transition 的下采样 /
//! 上采样路径、skip 的接法与对齐、Transformer 瓶颈（含 pre-norm、SiLU MLP、
//! PyTorch 布局的 in_proj、sin-cos 位置编码）、SpatialAttention、
//! BatchNorm 推理模式，以及贯穿全网的 align_corners=False 插值。

mod common;

use burn::tensor::{Tensor, TensorData};
use common::{assert_close4, assert_close5, open_npz, read, read_scalar, state_dict, tensor4, B};
use eser::attention::TransformerBottleneck;
use eser::model::{DenseEd, DenseEdConfig};
use eser::model3d::{DenseEd3d, DenseEd3dConfig};
use eser::store::LoadState;

/// 与 `export_golden.py` 的 layer6_model 一一对应。
fn config(tag: &str) -> DenseEdConfig {
    let mut cfg = match tag {
        "m3blk" => DenseEdConfig::new(2, 3, vec![2, 2, 2]),
        "m5blk" => DenseEdConfig::new(3, 4, vec![2, 2, 2, 2, 2]),
        "m_odd" => DenseEdConfig::new(1, 1, vec![2, 2, 2]),
        other => panic!("未知配置 {}", other),
    };
    cfg.growth_rate = 4;
    cfg.init_features = 8;
    cfg.output_size = Some(match tag {
        "m3blk" => [32, 32],
        "m5blk" => [40, 40],
        _ => [33, 29],
    });
    cfg
}

fn build(tag: &str) -> DenseEd<B> {
    let mut npz = open_npz();
    let sd = state_dict(&mut npz, tag);
    let mut model = config(tag)
        .init::<B>(&Default::default())
        .expect("建模失败");
    model
        .load_state(&sd, "")
        .unwrap_or_else(|e| panic!("{} 加载权重失败：{}", tag, e));
    // 有键没被取走 = 有一层还留着随机初始化，模型照样前向、照样出数
    sd.ensure_fully_used()
        .unwrap_or_else(|e| panic!("{}：{}", tag, e));
    model
}

#[test]
fn dense_ed_forward_matches_reference_3blk() {
    let tag = "m3blk";
    let model = build(tag);
    let mut npz = open_npz();
    let x = tensor4(&mut npz, &format!("{}/input", tag));
    assert_close4(
        &format!("{}/output", tag),
        model.forward(x),
        &read(&mut npz, &format!("{}/output", tag)),
    );
}

#[test]
fn dense_ed_forward_matches_reference_5blk() {
    // 两级编解码：多一层 skip、多一次上采样，接错了这里才会暴露
    let tag = "m5blk";
    let model = build(tag);
    let mut npz = open_npz();
    let x = tensor4(&mut npz, &format!("{}/input", tag));
    assert_close4(
        &format!("{}/output", tag),
        model.forward(x),
        &read(&mut npz, &format!("{}/output", tag)),
    );
}

#[test]
fn dense_ed_forward_matches_reference_odd_grid() {
    // 33x29：非方形且是奇数，skip 尺寸对齐和输出兜底两条插值路径都会走到
    let tag = "m_odd";
    let model = build(tag);
    let mut npz = open_npz();
    let x = tensor4(&mut npz, &format!("{}/input", tag));
    assert_close4(
        &format!("{}/output", tag),
        model.forward(x),
        &read(&mut npz, &format!("{}/output", tag)),
    );
}

#[test]
fn wrong_prefix_is_caught_not_silently_ignored() {
    // 前缀写错时那一层会留着随机初始化，前向照样能跑通。
    // 这条防线是 ensure_fully_used，不是「结果看起来合理」
    let mut npz = open_npz();
    let sd = state_dict(&mut npz, "m3blk");
    let mut model = config("m3blk")
        .init::<B>(&Default::default())
        .expect("建模失败");
    assert!(
        model.load_state(&sd, "oops").is_err(),
        "前缀写错必须报错，而不是加载 0 个权重后返回 Ok"
    );
}

// ---------------------------------------------------------------- 3D 路径

#[test]
fn dense_ed_3d_forward_matches_reference() {
    // 与 export_golden.py 的 m3d 一致：in=2 / out=3 / blocks=[2,2,2] / (2,24,24)
    let mut npz = open_npz();
    let sd = state_dict(&mut npz, "m3d");
    let mut cfg = DenseEd3dConfig::new(2, 3, vec![2, 2, 2]);
    cfg.growth_rate = 4;
    cfg.init_features = 8;
    cfg.output_size = Some([2, 24, 24]);
    let mut model: DenseEd3d<B> = cfg.init(&Default::default()).expect("建模失败");
    model.load_state(&sd, "").expect("加载 3D 权重失败");
    sd.ensure_fully_used().expect("有 3D 权重没被加载");

    let src = read(&mut npz, "m3d/input");
    let s = src.shape().to_vec();
    let flat: Vec<f32> = src.iter().copied().collect();
    let x = Tensor::<B, 1>::from_data(TensorData::from(flat.as_slice()), &Default::default())
        .reshape([s[0], s[1], s[2], s[3], s[4]]);

    assert_close5(
        "m3d/output",
        model.forward(x),
        &read(&mut npz, "m3d/output"),
    );
}

// ---------------------------------------------------- Transformer 瓶颈单独验

/// 建一个装好参考权重的瓶颈。channels=16 / heads=4，与 `layer6_attention` 一致。
fn bottleneck() -> TransformerBottleneck<B> {
    let mut npz = open_npz();
    let sd = state_dict(&mut npz, "attn");
    let mut m = TransformerBottleneck::<B>::new(16, 4, true, 8192, &Default::default());
    m.load_state(&sd, "").expect("加载瓶颈权重失败");
    sd.ensure_fully_used().expect("有瓶颈权重没被加载");
    m
}

#[test]
fn attention_reference_is_actually_selective() {
    // 这一项守的是**参考数据本身的鉴别力**：整网那几个配置里 softmax 几乎是均匀
    // 分布（权重 ≈ 1/64），Q/K 接反都验不出来。专门这组必须让 softmax 有选择性，
    // 否则下面两个测试就是摆设
    let mut npz = open_npz();
    let spread = read_scalar(&mut npz, "attn/weight_spread");
    assert!(
        spread > 0.1,
        "注意力权重极差只有 {:.4}，softmax 退化成均匀分布，这组参考数据验不动注意力核",
        spread
    );
}

#[test]
fn transformer_bottleneck_forward_matches_reference() {
    let m = bottleneck();
    let mut npz = open_npz();
    let x = tensor4(&mut npz, "attn/input");
    assert_close4("attn/output", m.forward(x), &read(&mut npz, "attn/output"));
}

#[test]
fn transformer_bottleneck_3d_forward_matches_reference() {
    // 3D 的位置编码按 3 根轴分通道（余数给最后一根），与 2D 不是同一条路径
    let m = bottleneck();
    let mut npz = open_npz();
    let src = read(&mut npz, "attn/input3d");
    let s = src.shape().to_vec();
    let flat: Vec<f32> = src.iter().copied().collect();
    let x = Tensor::<B, 1>::from_data(TensorData::from(flat.as_slice()), &Default::default())
        .reshape([s[0], s[1], s[2], s[3], s[4]]);

    assert_close5(
        "attn/output3d",
        m.forward(x),
        &read(&mut npz, "attn/output3d"),
    );
}
