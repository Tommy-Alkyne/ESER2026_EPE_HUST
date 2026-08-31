//! 双线性 / 三线性重采样，对齐 PyTorch 的 `align_corners=False`。
//!
//! **Burn 自带的 `interpolate(InterpolateMode::Bilinear)` 与 PyTorch 的默认语义
//! 不是一回事**——实测同一输入 2 倍上采样，最大绝对偏差 0.35，不是舍入级别的差，
//! 是坐标映射约定本身不同（Burn 走 `align_corners=True` 那一套）。
//!
//! 这在本项目里不是小事：模型每一级 `_Transition` 上采样、每一次 skip 尺寸对齐、
//! 以及最后的 `output_size` 兜底都走插值。用错约定不会报错，只会让整张预测场
//! 系统性地偏半个格点——正好是壁面边界层最要命的地方。
//!
//! # PyTorch 的 align_corners=False 是怎么算的
//!
//! ```text
//! scale = in / out
//! src   = max((dst + 0.5) * scale - 0.5, 0)
//! i0    = floor(src)，i1 = min(i0 + 1, in - 1)，w = src - i0
//! out[dst] = (1 - w) * x[i0] + w * x[i1]
//! ```
//!
//! 而 `align_corners=True` 用的是 `src = dst * (in - 1) / (out - 1)`：两端对齐到
//! 像素中心。两者在 in=out 时相同，只要放缩就不同。
//!
//! 实现上按轴分解（双线性本身就是可分离的张量积），一份 1D 代码同时供 2D 与 3D 用。
//! 用 `select` 而不是手写下标循环，梯度才能回传——插值层在编解码器中间，梯度断在
//! 这里等于整个解码器不训练。

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

/// 沿 `dim` 轴线性重采样到 `out` 个点。`in == out` 时原样返回（零开销）。
fn resample_axis<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    dim: usize,
    out: usize,
) -> Tensor<B, D> {
    let n_in = x.dims()[dim];
    assert!(out > 0, "目标尺寸必须为正");
    if n_in == out {
        return x;
    }
    assert!(n_in > 0, "输入尺寸为 0，无法重采样");

    let scale = n_in as f64 / out as f64;
    let mut lo = Vec::with_capacity(out);
    let mut hi = Vec::with_capacity(out);
    let mut frac = Vec::with_capacity(out);
    for j in 0..out {
        // 负数夹到 0：PyTorch 在非 cubic 模式下就是这么处理左边界的
        let src = ((j as f64 + 0.5) * scale - 0.5).max(0.0);
        let f = src.floor();
        let i0 = f as usize;
        lo.push(i0 as i32);
        // 右边界不再前进一格，此时权重作用在同一个点上，结果与只取 x[i0] 相同
        hi.push(if i0 + 1 < n_in {
            i0 as i32 + 1
        } else {
            i0 as i32
        });
        frac.push((src - f) as f32);
    }

    let device = x.device();
    let idx = |v: &[i32]| Tensor::<B, 1, Int>::from_data(TensorData::from(v), &device);
    let a = x.clone().select(dim, idx(&lo));
    let b = x.select(dim, idx(&hi));

    let mut wshape = [1usize; D];
    wshape[dim] = out;
    let w = Tensor::<B, 1>::from_data(TensorData::from(frac.as_slice()), &device).reshape(wshape);

    a * (w.clone().neg().add_scalar(1.0)) + b * w
}

/// 等价于 `F.interpolate(x, size, mode='bilinear', align_corners=False)`。
pub fn resize_bilinear<B: Backend>(x: Tensor<B, 4>, size: [usize; 2]) -> Tensor<B, 4> {
    let x = resample_axis(x, 2, size[0]);
    resample_axis(x, 3, size[1])
}

/// 等价于 `F.interpolate(x, size, mode='trilinear', align_corners=False)`。
///
/// 本项目的 3D 模型只在 H/W 上放缩（`scale_factor=(1, 2, 2)`），深度轴通常原样
/// 透传——`resample_axis` 在 `in == out` 时直接返回，那条路径零开销。
pub fn resize_trilinear<B: Backend>(x: Tensor<B, 5>, size: [usize; 3]) -> Tensor<B, 5> {
    let x = resample_axis(x, 2, size[0]);
    let x = resample_axis(x, 3, size[1]);
    resample_axis(x, 4, size[2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn ramp(n: usize) -> Tensor<B, 4> {
        let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Tensor::<B, 1>::from_data(TensorData::from(v.as_slice()), &Default::default())
            .reshape([1, 1, 1, n])
    }

    #[test]
    fn same_size_is_identity() {
        let x = ramp(7);
        let got = resize_bilinear(x.clone(), [1, 7]);
        assert_eq!(
            got.into_data().to_vec::<f32>().unwrap(),
            x.into_data().to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn linear_ramp_stays_linear_under_upsampling() {
        // x = [0,1,2,3] 上采样到 8 点：src(j) = max(0.5j - 0.25, 0)。
        // 线性函数在线性插值下必须精确复现，只有被夹取的两端会平掉
        let got: Vec<f32> = resize_bilinear(ramp(4), [1, 8])
            .into_data()
            .to_vec()
            .unwrap();
        let want = [0.0f32, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0];
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "下标 {}：{} != {}", i, g, w);
        }
    }

    #[test]
    fn does_not_use_align_corners_convention() {
        // 这个测试是防止哪天有人图省事换回 Burn 自带的 interpolate。
        // 同一组输入，align_corners=True 给 src(1) = 3/7 ≈ 0.4286，
        // align_corners=False 给 0.25——差 0.18，绝不是舍入误差
        let got: Vec<f32> = resize_bilinear(ramp(4), [1, 8])
            .into_data()
            .to_vec()
            .unwrap();
        assert!(
            (got[1] - 0.25).abs() < 1e-6,
            "下标 1 应为 0.25，实得 {}（≈0.4286 说明用的是 align_corners=True）",
            got[1]
        );
        assert!(
            (got[7] - 3.0).abs() < 1e-6,
            "右端点夹取后应为 3.0，实得 {}",
            got[7]
        );
    }

    #[test]
    fn downsampling_averages_neighbours() {
        // 4 -> 2：scale=2，src = (j+0.5)*2-0.5 = 0.5, 2.5
        // j=0: 0.5*x[0] + 0.5*x[1] = 0.5；j=1: 0.5*x[2] + 0.5*x[3] = 2.5
        let got: Vec<f32> = resize_bilinear(ramp(4), [1, 2])
            .into_data()
            .to_vec()
            .unwrap();
        assert!((got[0] - 0.5).abs() < 1e-6, "实得 {}", got[0]);
        assert!((got[1] - 2.5).abs() < 1e-6, "实得 {}", got[1]);
    }
}
