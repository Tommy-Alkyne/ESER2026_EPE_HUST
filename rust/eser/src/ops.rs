//! 张量算子层：差分算子、掩码统计、腐蚀。
//!
//! 移植自 `utils/physics_loss.py`。**这是整个重构里风险最高的一块**——写错不会
//! 报错，只会让物理残差静默失真，模型「看起来在训练」但约束是错的。
//!
//! 所以每个函数都必须逐元素对上 `tests/golden/golden.npz` 里的 `op/*` 与
//! `mask/*` 参考数据，容差 1e-5。见 `tests/golden.rs`。
//!
//! ## 为什么要手写 replicate padding
//!
//! Burn 的 `Tensor::pad` 只支持**常量值填充**，没有 replicate / reflect 模式。
//! 而零填充等价于把墙外当成 0 值流场，会在边界引入虚假的巨大梯度。所以这里用
//! 切片 + 拼接手工复制边缘行列。

use burn::tensor::backend::Backend;
use burn::tensor::module::{conv2d, max_pool2d};
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Tensor, TensorData};

/// 边缘复制填充：H 与 W 各向外扩 `pad` 个格点，取值复制自最外一行/列。
///
/// 等价于 PyTorch 的 `F.pad(x, (pad,)*4, mode='replicate')`。
/// 先补上下再补左右——顺序无所谓，因为角点的取值在两种顺序下都等于原角点值。
pub fn replicate_pad_2d<B: Backend>(x: Tensor<B, 4>, pad: usize) -> Tensor<B, 4> {
    if pad == 0 {
        return x;
    }
    let [b, c, h, w] = x.dims();

    // 上下：把第 0 行与第 h-1 行各复制 pad 份
    let top = x.clone().slice([0..b, 0..c, 0..1, 0..w]);
    let bottom = x.clone().slice([0..b, 0..c, h - 1..h, 0..w]);
    let mut parts = Vec::with_capacity(2 * pad + 1);
    for _ in 0..pad {
        parts.push(top.clone());
    }
    parts.push(x);
    for _ in 0..pad {
        parts.push(bottom.clone());
    }
    let x = Tensor::cat(parts, 2);

    // 左右：同理复制第 0 列与第 w-1 列（高度已经变成 h + 2*pad）
    let h2 = h + 2 * pad;
    let left = x.clone().slice([0..b, 0..c, 0..h2, 0..1]);
    let right = x.clone().slice([0..b, 0..c, 0..h2, w - 1..w]);
    let mut parts = Vec::with_capacity(2 * pad + 1);
    for _ in 0..pad {
        parts.push(left.clone());
    }
    parts.push(x);
    for _ in 0..pad {
        parts.push(right.clone());
    }
    Tensor::cat(parts, 3)
}

/// 把一个 3x3 核构造成 (C, 1, 3, 3) 的分组卷积权重。
fn grouped_kernel<B: Backend>(
    values: [[f32; 3]; 3],
    channels: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let flat: Vec<f32> = values.iter().flat_map(|r| r.iter().copied()).collect();
    let k = Tensor::<B, 1>::from_data(TensorData::from(flat.as_slice()), device)
        .reshape([1, 1, 3, 3]);
    // PyTorch 侧是 kernel.repeat(C,1,1,1)，即沿输出通道维复制 C 份
    Tensor::cat((0..channels).map(|_| k.clone()).collect::<Vec<_>>(), 0)
}

/// replicate padding + 无 padding 的逐通道 3x3 卷积。
///
/// 对应 Python 的 `_conv_replicate`。注意 PyTorch 的 `conv2d` 实际做的是
/// **互相关**（不翻转核），Burn 同理——这里用的一阶差分核是反对称的，一旦哪边
/// 翻转了核，符号会反过来，黄金对照会立刻抓到。
pub fn conv_replicate<B: Backend>(x: Tensor<B, 4>, kernel: [[f32; 3]; 3]) -> Tensor<B, 4> {
    let [_, c, _, _] = x.dims();
    let device = x.device();
    let weight = grouped_kernel::<B>(kernel, c, &device);
    let padded = replicate_pad_2d(x, 1);
    conv2d(padded, weight, None, ConvOptions::new([1, 1], [0, 0], [1, 1], c))
}

const KX: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]];
const KY: [[f32; 3]; 3] = [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]];
const KXX: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]];
const KYY: [[f32; 3]; 3] = [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]];

/// 一阶中心差分，返回 `(d/dx, d/dy)`。各向异性网格下分别除以 dx / dy。
pub fn grad_xy<B: Backend>(field: Tensor<B, 4>, dx: f64, dy: f64) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let gx = conv_replicate(field.clone(), KX).div_scalar(dx);
    let gy = conv_replicate(field, KY).div_scalar(dy);
    (gx, gy)
}

/// 五点 Laplacian。dx ≠ dy 时两个方向分别按 dx² / dy² 缩放。
pub fn laplacian<B: Backend>(field: Tensor<B, 4>, dx: f64, dy: f64) -> Tensor<B, 4> {
    let lxx = conv_replicate(field.clone(), KXX).div_scalar(dx * dx);
    let lyy = conv_replicate(field, KYY).div_scalar(dy * dy);
    lxx + lyy
}

/// 掩码内均值。分母是**掩码内像素数**，不是全图——否则损失会被域外的零值稀释。
///
/// `mask` 为 `None` 时退化成普通 mean。掩码形状可与 `x` 广播。
pub fn masked_mean<B: Backend>(x: Tensor<B, 4>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 1> {
    match mask {
        None => x.mean(),
        Some(m) => {
            let m = broadcast_to::<B>(m, x.dims());
            let num = (x * m.clone()).sum();
            let den = m.sum().clamp_min(1.0);
            num / den
        }
    }
}

/// 把掩码按需广播到目标形状（Burn 的算术不像 PyTorch 那样隐式广播所有维度）。
fn broadcast_to<B: Backend>(m: Tensor<B, 4>, target: [usize; 4]) -> Tensor<B, 4> {
    let dims = m.dims();
    if dims == target {
        return m;
    }
    let mut out = m;
    for (axis, (&have, &want)) in dims.iter().zip(target.iter()).enumerate() {
        if have == want {
            continue;
        }
        assert_eq!(have, 1, "掩码第 {} 维是 {}，无法广播到 {}", axis, have, want);
        out = Tensor::cat((0..want).map(|_| out.clone()).collect::<Vec<_>>(), axis);
    }
    out
}

/// 把掩码向内收缩 `radius` 个格点。
///
/// 3x3 差分模板在贴壁格点上会取到墙外的值，直接在原掩码上评估物理残差，墙外的
/// 数值就会从边界漏进损失里。收缩之后每个评估点的整个模板都落在计算域内部。
///
/// **图像边界也算域外**：Python 侧踩过一个真实缺陷——用 `max_pool2d(padding=r)`
/// 时补的是 -inf，贴着图像边缘的格点根本腐蚀不掉，而它们的模板同样伸到图像外。
/// 所以这里显式用 1（域外）补边再池化，与修好后的 Python 版一致。
pub fn erode_mask<B: Backend>(mask: Tensor<B, 4>, radius: usize) -> Tensor<B, 4> {
    if radius == 0 {
        return mask;
    }
    let kernel = 2 * radius + 1;
    let inv = mask.neg().add_scalar(1.0); // 1 - mask
    let inv = constant_pad_2d(inv, radius, 1.0); // 域外补 1
    let pooled = max_pool2d(inv, [kernel, kernel], [1, 1], [0, 0], [1, 1]);
    pooled.neg().add_scalar(1.0) // 1 - pooled
}

/// 常量值填充（H / W 各向外扩 `pad`）。
fn constant_pad_2d<B: Backend>(x: Tensor<B, 4>, pad: usize, value: f32) -> Tensor<B, 4> {
    if pad == 0 {
        return x;
    }
    let [b, c, h, w] = x.dims();
    let device = x.device();

    let row = Tensor::<B, 4>::full([b, c, pad, w], value, &device);
    let x = Tensor::cat(vec![row.clone(), x, row], 2);

    let col = Tensor::<B, 4>::full([b, c, h + 2 * pad, pad], value, &device);
    Tensor::cat(vec![col.clone(), x, col], 3)
}
