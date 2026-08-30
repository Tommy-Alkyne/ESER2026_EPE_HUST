//! 对照 Python 版冻结的黄金参考数据，逐元素验收 Rust 算子。
//!
//! 参考数据由 `tests/export_golden.py` 生成，共 479 项。本文件覆盖第 1、2 层
//! （差分算子、掩码几何）——也就是 P1 的验收范围。
//!
//! 判据与 `manifest.json` 里声明的一致：`|a-b| <= tol + tol*|b|`，tol = 1e-5。
//!
//! **验收标准不是「跑起来了」**。物理残差写错不会报错，只会让模型看起来在训练
//! 但约束是错的，所以这里必须逐元素对上，没有商量余地。

use std::fs::File;
use std::path::PathBuf;

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;

use eser::geometry::{coverage, default_t_geometry, rect_mask, resample_raster};
use eser::ops::{conv_replicate, erode_mask, grad_xy, laplacian, masked_mean,
                replicate_pad_2d};

type B = NdArray<f32>;

const TOL: f64 = 1e-5;

/// 打开黄金数据。路径相对本 crate（`rust/eser/`）往上两级回到仓库根。
fn open_npz() -> NpzReader<File> {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "..", "tests", "golden",
                         "golden.npz"].iter().collect();
    let file = File::open(&path).unwrap_or_else(|e| {
        panic!("打不开黄金数据 {:?}: {}\n先跑 python -m tests.export_golden", path, e)
    });
    NpzReader::new(file).expect("golden.npz 不是合法的 npz")
}

fn read(npz: &mut NpzReader<File>, name: &str) -> ArrayD<f32> {
    // numpy savez 在 zip 里给每个键加了 .npy 后缀（实测条目名形如 "op/input.npy"）
    npz.by_name(&format!("{}.npy", name))
        .unwrap_or_else(|e| panic!("黄金数据里没有 {:?}: {}", name, e))
}

fn to_tensor4(a: &ArrayD<f32>) -> Tensor<B, 4> {
    let s = a.shape();
    assert_eq!(s.len(), 4, "期望 4D，实得 {:?}", s);
    let flat: Vec<f32> = a.iter().copied().collect();
    Tensor::<B, 1>::from_data(TensorData::from(flat.as_slice()), &Default::default())
        .reshape([s[0], s[1], s[2], s[3]])
}

/// 逐元素比对，失败时报出最大偏差与位置，而不是只说「不相等」。
fn assert_close4(name: &str, got: Tensor<B, 4>, want: &ArrayD<f32>) {
    let dims = got.dims();
    assert_eq!(
        &dims[..], want.shape(),
        "{}: 形状 {:?} != 参考 {:?}", name, dims, want.shape()
    );
    let got_v: Vec<f32> = got.into_data().to_vec().unwrap();
    let want_v: Vec<f32> = want.iter().copied().collect();

    let mut worst = 0.0f64;
    let mut worst_i = 0usize;
    let mut bad = 0usize;
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
        bad, 0,
        "{}: {} / {} 处超差；最差在下标 {}，got={} want={}，超出容差 {:.3e}",
        name, bad, got_v.len(), worst_i, got_v[worst_i], want_v[worst_i], worst
    );
}

fn assert_close_scalar(name: &str, got: Tensor<B, 1>, want: &ArrayD<f32>) {
    let g = got.into_data().to_vec::<f32>().unwrap()[0] as f64;
    let w = want.iter().copied().next().unwrap() as f64;
    let limit = TOL + TOL * w.abs();
    assert!(
        (g - w).abs() <= limit,
        "{}: got={} want={}，偏差 {:.3e} 超过容差 {:.3e}",
        name, g, w, (g - w).abs(), limit
    );
}

// ---------------------------------------------------------------- 第 1 层

#[test]
fn diff_kernels_match_reference() {
    // 核本身也要对——符号写反是最常见的低级错误，而反对称核翻转后只是变号
    let mut npz = open_npz();
    let kx = read(&mut npz, "op/kernel_x");
    let ky = read(&mut npz, "op/kernel_y");
    let expect_kx = [0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0, 0.0];
    let expect_ky = [0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0];
    let kxv: Vec<f32> = kx.iter().copied().collect();
    let kyv: Vec<f32> = ky.iter().copied().collect();
    assert_eq!(kxv, expect_kx, "x 方向差分核与参考不符");
    assert_eq!(kyv, expect_ky, "y 方向差分核与参考不符");
}

#[test]
fn replicate_padding_matches_reference() {
    // 间接验证：conv_replicate 的结果对上，就说明 padding 与卷积都对。
    // 单独把 padding 拎出来再验一次形状与边缘复制语义。
    let mut npz = open_npz();
    let x = to_tensor4(&read(&mut npz, "op/input"));
    let [b, c, h, w] = x.dims();
    let padded = replicate_pad_2d(x.clone(), 1);
    assert_eq!(padded.dims(), [b, c, h + 2, w + 2], "padding 后形状不对");

    // 角点必须等于原角点，边必须等于原边（这正是 replicate 与零填充的区别）
    let pv: Vec<f32> = padded.into_data().to_vec().unwrap();
    let xv: Vec<f32> = x.into_data().to_vec().unwrap();
    let pw = w + 2;
    let at_p = |bi: usize, ci: usize, y: usize, xx: usize| {
        pv[((bi * c + ci) * (h + 2) + y) * pw + xx]
    };
    let at_x = |bi: usize, ci: usize, y: usize, xx: usize| {
        xv[((bi * c + ci) * h + y) * w + xx]
    };
    assert_eq!(at_p(0, 0, 0, 0), at_x(0, 0, 0, 0), "左上角没有复制原角点");
    assert_eq!(at_p(0, 0, 0, pw - 1), at_x(0, 0, 0, w - 1), "右上角不对");
    assert_eq!(at_p(1, 2, h + 1, 0), at_x(1, 2, h - 1, 0), "左下角不对");
    assert_eq!(at_p(0, 1, 0, 3), at_x(0, 1, 0, 2), "上边缘没有复制第 0 行");
}

#[test]
fn conv_replicate_matches_reference() {
    let mut npz = open_npz();
    let x = to_tensor4(&read(&mut npz, "op/input"));
    const KX: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]];
    const KY: [[f32; 3]; 3] = [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]];
    assert_close4("op/conv_replicate_x", conv_replicate(x.clone(), KX),
                  &read(&mut npz, "op/conv_replicate_x"));
    assert_close4("op/conv_replicate_y", conv_replicate(x, KY),
                  &read(&mut npz, "op/conv_replicate_y"));
}

#[test]
fn grad_and_laplacian_match_reference() {
    let mut npz = open_npz();
    let x = to_tensor4(&read(&mut npz, "op/input"));
    // 与导出脚本一致：刻意用各向异性且非整数的间距
    let (dx, dy) = (0.037_f64, 0.11_f64);

    let (gx, gy) = grad_xy(x.clone(), dx, dy);
    assert_close4("op/grad_x", gx, &read(&mut npz, "op/grad_x"));
    assert_close4("op/grad_y", gy, &read(&mut npz, "op/grad_y"));
    assert_close4("op/laplacian", laplacian(x, dx, dy),
                  &read(&mut npz, "op/laplacian"));
}

#[test]
fn single_channel_path_matches_reference() {
    // Python 侧单通道走的是不带 groups 的分支，两条路径都要覆盖
    let mut npz = open_npz();
    let full = to_tensor4(&read(&mut npz, "op/input"));
    let [b, _, h, w] = full.dims();
    let x1 = full.slice([0..b, 0..1, 0..h, 0..w]);
    let (gx, _) = grad_xy(x1, 0.037, 0.11);
    assert_close4("op/grad_x_1ch", gx, &read(&mut npz, "op/grad_x_1ch"));
}

// ---------------------------------------------------------------- 第 2 层

#[test]
fn erode_mask_matches_reference() {
    let mut npz = open_npz();
    for case in ["all_ones", "band_touching_edges", "interior_block",
                 "single_pixel", "checker"] {
        let src = to_tensor4(&read(&mut npz, &format!("mask/{}/src", case)));
        for r in [1usize, 2] {
            let name = format!("mask/{}/erode{}", case, r);
            let want = read(&mut npz, &name);
            assert_close4(&name, erode_mask(src.clone(), r), &want);
        }
    }
}

#[test]
fn erode_treats_image_border_as_outside() {
    // 回归防线：Python 侧踩过的真实缺陷——贴着图像边缘的格点腐蚀不掉。
    // 全 1 的 9x9 腐蚀一圈后应当只剩 7x7 = 49 个格点。
    let ones = Tensor::<B, 4>::ones([1, 1, 9, 9], &Default::default());
    let eroded = erode_mask(ones, 1);
    let sum: f32 = eroded.clone().sum().into_data().to_vec::<f32>().unwrap()[0];
    assert_eq!(sum, 49.0, "全 1 掩码腐蚀 1 圈应剩 49 个格点，实得 {}", sum);

    // 贴边的计算域（地铁站站台层横贯整幅网格）两端必须被腐蚀掉
    let mut data = vec![0.0f32; 81];
    for y in 3..6 {
        for x in 0..9 {
            data[y * 9 + x] = 1.0;
        }
    }
    let band = Tensor::<B, 1>::from_data(TensorData::from(data.as_slice()),
                                         &Default::default())
        .reshape([1, 1, 9, 9]);
    let v: Vec<f32> = erode_mask(band, 1).into_data().to_vec().unwrap();
    assert_eq!(v[4 * 9], 0.0, "贴左边的格点没被腐蚀掉");
    assert_eq!(v[4 * 9 + 8], 0.0, "贴右边的格点没被腐蚀掉");
    assert_eq!(v[4 * 9 + 4], 1.0, "域内格点被误腐蚀");
}

#[test]
fn masked_mean_matches_reference() {
    let mut npz = open_npz();
    let v = to_tensor4(&read(&mut npz, "mask/masked_mean_input"));
    let band = to_tensor4(&read(&mut npz, "mask/band_touching_edges/src"));

    assert_close_scalar("mask/masked_mean_banded",
                        masked_mean(v.clone(), Some(band)),
                        &read(&mut npz, "mask/masked_mean_banded"));
    assert_close_scalar("mask/masked_mean_none",
                        masked_mean(v, None),
                        &read(&mut npz, "mask/masked_mean_none"));
}

#[test]
fn masked_mean_denominator_is_mask_area() {
    // 分母必须是掩码内像素数。如果错用全图像素数，掩码只盖一半时结果会差一倍。
    let x = Tensor::<B, 4>::ones([1, 1, 4, 4], &Default::default());
    let mut m = vec![0.0f32; 16];
    m[..4].fill(1.0); // 只有第一行
    let mask = Tensor::<B, 1>::from_data(TensorData::from(m.as_slice()),
                                         &Default::default())
        .reshape([1, 1, 4, 4]);
    let got: f32 = masked_mean(x, Some(mask)).into_data().to_vec::<f32>().unwrap()[0];
    assert!((got - 1.0).abs() < 1e-6,
            "全 1 场在任意掩码下均值都应是 1.0（分母是掩码面积），实得 {}", got);
}


// ---------------------------------------------------- 第 2 层：几何掩码

/// 逐元素比对一维展平的掩码。
fn assert_mask_eq(name: &str, got: &[f32], want: &ArrayD<f32>) {
    let want_v: Vec<f32> = want.iter().copied().collect();
    assert_eq!(got.len(), want_v.len(), "{}: 长度 {} != 参考 {}",
               name, got.len(), want_v.len());
    let bad: Vec<usize> = got.iter().zip(want_v.iter()).enumerate()
        .filter(|(_, (g, w))| (**g - **w).abs() > 1e-6)
        .map(|(i, _)| i).collect();
    assert!(bad.is_empty(), "{}: {} 处不符，首个在下标 {}（got={} want={}）",
            name, bad.len(), bad[0], got[bad[0]], want_v[bad[0]]);
}

#[test]
fn rect_geometry_matches_reference() {
    let mut npz = open_npz();
    let segs = default_t_geometry();
    for (h, w) in [(65usize, 65usize), (137, 137), (96, 240)] {
        let name = format!("geom/rect_{}x{}", h, w);
        let want = read(&mut npz, &name);
        let got = rect_mask(h, w, &segs).expect("生成默认 T 型掩码失败");
        assert_mask_eq(&name, &got, &want);
    }
}

#[test]
fn rect_geometry_is_resolution_invariant() {
    // 归一化比例的全部意义就在这里：任意分辨率覆盖同一物理区域。
    // 旧版硬编码像素坐标时，65x65 覆盖 63.8%、257x257 只剩 12.8%。
    let segs = default_t_geometry();
    let covs: Vec<f64> = [65usize, 137, 257, 512]
        .iter()
        .map(|&n| coverage(&rect_mask(n, n, &segs).unwrap()))
        .collect();
    let spread = covs.iter().cloned().fold(f64::MIN, f64::max)
        - covs.iter().cloned().fold(f64::MAX, f64::min);
    assert!(spread < 0.01, "各分辨率覆盖率应基本一致，实得 {:?}（极差 {:.4}）",
            covs, spread);
}

#[test]
fn raster_geometry_matches_reference() {
    let mut npz = open_npz();
    let src_arr = read(&mut npz, "geom/raster_src");
    let (sh, sw) = (src_arr.shape()[0], src_arr.shape()[1]);
    let src: Vec<f32> = src_arr.iter().copied().collect();

    for (h, w) in [(40usize, 100usize), (80, 200), (20, 50)] {
        let name = format!("geom/raster_{}x{}", h, w);
        let want = read(&mut npz, &name);
        let got = resample_raster(&src, sh, sw, h, w).expect("栅格重采样失败");
        assert_mask_eq(&name, &got, &want);
    }
}

#[test]
fn raster_resample_stays_binary() {
    let mut npz = open_npz();
    let src_arr = read(&mut npz, "geom/raster_src");
    let (sh, sw) = (src_arr.shape()[0], src_arr.shape()[1]);
    let src: Vec<f32> = src_arr.iter().copied().collect();
    let got = resample_raster(&src, sh, sw, 160, 400).unwrap();
    assert!(got.iter().all(|&v| v == 0.0 || v == 1.0),
            "重采样后必须仍是纯 0/1——任何插值都会造出没有物理意义的 0.5");
}
