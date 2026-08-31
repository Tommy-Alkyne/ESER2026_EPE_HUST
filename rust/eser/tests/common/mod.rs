//! 黄金参考数据的读取与比对工具，供各层验收测试共用。
//!
//! 判据与 `manifest.json` 里声明的一致：`|a-b| <= tol + tol*|b|`，tol = 1e-5。
//! 失败时必须报出最大偏差与位置，只说「不相等」没法定位是哪一层写错了。
//!
//! # 这批数据是**冻结**的
//!
//! `tests/golden/golden.npz` 的 502 项参考张量当初由 `tests/export_golden.py` 从
//! PyTorch 版跑出来。**Python 版已经退役、脚本已删**，所以这批数据既不能再生成、
//! 也没有 `--verify` 可以回验了。
//!
//! 这不影响它作为回归防线的价值：它记录的是「Rust 侧当年逐元素对上了 PyTorch」
//! 这个事实，改了 Rust 实现照样会红。但要清楚它的边界——
//!
//! * 参考值本身**再没有第二个实现能交叉验证**了；
//! * 想扩充覆盖（比如新增一个算子的参考），得自己在 Rust 侧构造并论证正确性，
//!   不能再靠「跑一遍 Python 就有了」。

// 每个集成测试二进制只用到其中一部分函数，未用到的不算死代码
#![allow(dead_code)]

use std::fs::File;
use std::path::PathBuf;

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use eser::store::StateDict;
use ndarray::ArrayD;
use ndarray_npy::NpzReader;

/// 黄金数据全是 f32，验收统一用 CPU 后端——先保证算对，再谈算得快。
pub type B = NdArray<f32>;

pub const TOL: f64 = 1e-5;

/// 打开黄金数据。路径相对本 crate（`rust/eser/`）往上两级回到仓库根。
pub fn open_npz() -> NpzReader<File> {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "..",
        "..",
        "tests",
        "golden",
        "golden.npz",
    ]
    .iter()
    .collect();
    let file = File::open(&path).unwrap_or_else(|e| {
        panic!(
            "打不开黄金数据 {:?}: {}\n先跑 python -m tests.export_golden",
            path, e
        )
    });
    NpzReader::new(file).expect("golden.npz 不是合法的 npz")
}

pub fn read(npz: &mut NpzReader<File>, name: &str) -> ArrayD<f32> {
    // numpy savez 在 zip 里给每个键加了 .npy 后缀（实测条目名形如 "op/input.npy"）
    npz.by_name(&format!("{}.npy", name))
        .unwrap_or_else(|e| panic!("黄金数据里没有 {:?}: {}", name, e))
}

/// 读一维参考数组（y_min / y_max 这类逐通道常量）。
pub fn read_vec(npz: &mut NpzReader<File>, name: &str) -> Vec<f32> {
    read(npz, name).iter().copied().collect()
}

/// 读标量参考值（npz 里是 0 维数组）。
pub fn read_scalar(npz: &mut NpzReader<File>, name: &str) -> f64 {
    let a = read(npz, name);
    assert_eq!(a.len(), 1, "{} 不是标量，形状 {:?}", name, a.shape());
    a.iter().copied().next().unwrap() as f64
}

pub fn to_tensor4(a: &ArrayD<f32>) -> Tensor<B, 4> {
    let s = a.shape();
    assert_eq!(s.len(), 4, "期望 4D，实得 {:?}", s);
    let flat: Vec<f32> = a.iter().copied().collect();
    Tensor::<B, 1>::from_data(TensorData::from(flat.as_slice()), &Default::default())
        .reshape([s[0], s[1], s[2], s[3]])
}

/// 直接从 npz 读一个 4D 张量。
pub fn tensor4(npz: &mut NpzReader<File>, name: &str) -> Tensor<B, 4> {
    to_tensor4(&read(npz, name))
}

/// 逐元素比对已经展平的数据。判据是 manifest 里声明的 `|a-b| <= tol + tol*|b|`
/// ——**不是绝对容差**。参考量级差好几个数量级（残差场 1e-2、注意力输出 25），
/// 一刀切的绝对容差要么放过真错，要么把浮点噪声报成错。
pub fn assert_close_slice(name: &str, got_v: &[f32], want: &ArrayD<f32>) {
    let want_v: Vec<f32> = want.iter().copied().collect();
    assert_eq!(
        got_v.len(),
        want_v.len(),
        "{}: 元素数 {} != 参考 {}",
        name,
        got_v.len(),
        want_v.len()
    );

    let mut worst = 0.0f64;
    let mut worst_i = 0usize;
    let mut bad = 0usize;
    // **NaN 必须单独拦**：`NaN > limit` 恒为 false，用「大于阈值才算超差」统计的话，
    // 一整张全 NaN 的输出会被判成「零处超差」而全绿。这条判据护着项目里每一处黄金
    // 比对，写漏了等于整套验收对 NaN 失效
    if let Some(i) = got_v
        .iter()
        .zip(want_v.iter())
        .position(|(g, w)| g.is_nan() != w.is_nan() || (*g - *w).is_nan() && !w.is_nan())
    {
        panic!(
            "{}: 下标 {} 出现 NaN（got={} want={}）——NaN 不会被 `>` 判成超差，必须单独拦",
            name, i, got_v[i], want_v[i]
        );
    }
    for (i, (&g, &w)) in got_v.iter().zip(want_v.iter()).enumerate() {
        let d = (g as f64 - w as f64).abs();
        let limit = TOL + TOL * (w as f64).abs();
        if d > limit {
            bad += 1;
        }
        if d - limit > worst {
            worst = d - limit;
            worst_i = i;
        }
    }
    assert_eq!(
        bad,
        0,
        "{}: {} / {} 处超差；最差在下标 {}，got={} want={}，超出容差 {:.3e}",
        name,
        bad,
        got_v.len(),
        worst_i,
        got_v[worst_i],
        want_v[worst_i],
        worst
    );
}

/// 4D 张量的逐元素比对，先查形状再比数值。
pub fn assert_close4(name: &str, got: Tensor<B, 4>, want: &ArrayD<f32>) {
    let dims = got.dims();
    assert_eq!(
        &dims[..],
        want.shape(),
        "{}: 形状 {:?} != 参考 {:?}",
        name,
        dims,
        want.shape()
    );
    let got_v: Vec<f32> = got.into_data().to_vec().unwrap();
    assert_close_slice(name, &got_v, want);
}

/// 5D 张量的逐元素比对（3D 模型与三线性插值用）。
pub fn assert_close5(name: &str, got: Tensor<B, 5>, want: &ArrayD<f32>) {
    let dims = got.dims();
    assert_eq!(
        &dims[..],
        want.shape(),
        "{}: 形状 {:?} != 参考 {:?}",
        name,
        dims,
        want.shape()
    );
    let got_v: Vec<f32> = got.into_data().to_vec().unwrap();
    assert_close_slice(name, &got_v, want);
}

pub fn assert_close_scalar(name: &str, got: Tensor<B, 1>, want: &ArrayD<f32>) {
    let g = got.into_data().to_vec::<f32>().unwrap()[0] as f64;
    assert_close_f64(name, g, want.iter().copied().next().unwrap() as f64);
}

/// 比对已经取出来的标量（损失分项都是 f64）。
pub fn assert_close_f64(name: &str, got: f64, want: f64) {
    // `NaN <= limit` 是 false，下面的 assert 本身拦得住 NaN；单独判一次是为了让报错
    // 说清「这是 NaN」，而不是「偏差 NaN 超过容差」那种读起来像精度问题的话
    assert!(
        !got.is_nan(),
        "{}: got 是 NaN（want={}）——多半是上游算出了 NaN，不是精度问题",
        name,
        want
    );
    let limit = TOL + TOL * want.abs();
    assert!(
        (got - want).abs() <= limit,
        "{}: got={} want={}，偏差 {:.3e} 超过容差 {:.3e}",
        name,
        got,
        want,
        (got - want).abs(),
        limit
    );
}

/// 逐元素比对一维展平的掩码。
pub fn assert_mask_eq(name: &str, got: &[f32], want: &ArrayD<f32>) {
    let want_v: Vec<f32> = want.iter().copied().collect();
    assert_eq!(
        got.len(),
        want_v.len(),
        "{}: 长度 {} != 参考 {}",
        name,
        got.len(),
        want_v.len()
    );
    // 同样要拦 NaN：`NaN > 1e-6` 是 false，掩码里混进 NaN 会静默通过
    let bad: Vec<usize> = got
        .iter()
        .zip(want_v.iter())
        .enumerate()
        .filter(|(_, (g, w))| {
            let d = (**g - **w).abs();
            d.is_nan() || d > 1e-6
        })
        .map(|(i, _)| i)
        .collect();
    assert!(
        bad.is_empty(),
        "{}: {} 处不符，首个在下标 {}（got={} want={}）",
        name,
        bad.len(),
        bad[0],
        got[bad[0]],
        want_v[bad[0]]
    );
}

/// 把 `<tag>/param/*` 下的全部参数收成一份 PyTorch 风格的权重表。
///
/// 只有 parameter，没有 BatchNorm 的 running 统计量——那些在 PyTorch 里是 buffer，
/// `named_parameters()` 不包含。导出脚本把它们设成了 mean=0 / var=1，正好是 Burn
/// 的默认初值，所以推理结果仍然对得上。
pub fn state_dict(npz: &mut NpzReader<File>, tag: &str) -> StateDict {
    let prefix = format!("{}/param/", tag);
    let names: Vec<String> = npz
        .names()
        .expect("读不出 npz 的条目名")
        .into_iter()
        .filter(|n| n.starts_with(&prefix))
        .collect();
    assert!(!names.is_empty(), "黄金数据里没有 {} 的参数", tag);

    let mut sd = StateDict::new();
    for full in names {
        let key = full
            .strip_suffix(".npy")
            .unwrap_or(&full)
            .strip_prefix(&prefix)
            .expect("前缀过滤过了")
            .to_string();
        let arr: ArrayD<f32> = npz.by_name(&full).expect("读参数失败");
        sd.insert(key, arr.shape().to_vec(), arr.iter().copied().collect());
    }
    sd
}
