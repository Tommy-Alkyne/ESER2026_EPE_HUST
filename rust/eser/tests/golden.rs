//! 对照 Python 版冻结的黄金参考数据，逐元素验收 Rust 算子。
//!
//! 参考数据由 `tests/export_golden.py` 生成，共 502 项。本文件覆盖第 1、2 层
//! （差分算子、掩码几何）——也就是 P1 的验收范围。
//!
//! 判据与比对工具在 `tests/common/mod.rs`：`|a-b| <= tol + tol*|b|`，tol = 1e-5。
//!
//! **验收标准不是「跑起来了」**。物理残差写错不会报错，只会让模型看起来在训练
//! 但约束是错的，所以这里必须逐元素对上，没有商量余地。

mod common;

use burn::tensor::{Tensor, TensorData};

use common::{
    assert_close4, assert_close5, assert_close_scalar, assert_close_slice, assert_mask_eq,
    open_npz, read, to_tensor4, B,
};
use eser::geometry::{coverage, default_t_geometry, rect_mask, resample_raster};
use eser::interp::{resize_bilinear, resize_trilinear};
use eser::ops::{conv_replicate, erode_mask, grad_xy, laplacian, masked_mean, replicate_pad_2d};

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
    let at_p =
        |bi: usize, ci: usize, y: usize, xx: usize| pv[((bi * c + ci) * (h + 2) + y) * pw + xx];
    let at_x = |bi: usize, ci: usize, y: usize, xx: usize| xv[((bi * c + ci) * h + y) * w + xx];
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
    assert_close4(
        "op/conv_replicate_x",
        conv_replicate(x.clone(), KX),
        &read(&mut npz, "op/conv_replicate_x"),
    );
    assert_close4(
        "op/conv_replicate_y",
        conv_replicate(x, KY),
        &read(&mut npz, "op/conv_replicate_y"),
    );
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
    assert_close4(
        "op/laplacian",
        laplacian(x, dx, dy),
        &read(&mut npz, "op/laplacian"),
    );
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

#[test]
fn bilinear_interpolation_matches_reference() {
    // Burn 自带的 interpolate 用 align_corners=True 的坐标映射，与 PyTorch 默认
    // 不一致（实测 2 倍上采样最大偏差 0.35）。这几项对上才说明手写的那套对了
    let mut npz = open_npz();
    let x = to_tensor4(&read(&mut npz, "op/input"));
    let [_, _, h, w] = x.dims();

    assert_close4(
        "op/interp_up2",
        resize_bilinear(x.clone(), [h * 2, w * 2]),
        &read(&mut npz, "op/interp_up2"),
    );
    assert_close4(
        "op/interp_to_9x11",
        resize_bilinear(x.clone(), [9, 11]),
        &read(&mut npz, "op/interp_to_9x11"),
    );
    assert_close4(
        "op/interp_to_40x38",
        resize_bilinear(x, [40, 38]),
        &read(&mut npz, "op/interp_to_40x38"),
    );
}

#[test]
fn trilinear_interpolation_matches_reference() {
    let mut npz = open_npz();
    let src = read(&mut npz, "op/interp3d_src");
    let s = src.shape().to_vec();
    let flat: Vec<f32> = src.iter().copied().collect();
    let x = Tensor::<B, 1>::from_data(TensorData::from(flat.as_slice()), &Default::default())
        .reshape([s[0], s[1], s[2], s[3], s[4]]);

    // 深度轴 scale=1，应当原样透传
    assert_close5(
        "op/interp3d_up",
        resize_trilinear(x, [s[2], s[3] * 2, s[4] * 2]),
        &read(&mut npz, "op/interp3d_up"),
    );
}

// ---------------------------------------------------------------- 第 2 层

#[test]
fn erode_mask_matches_reference() {
    let mut npz = open_npz();
    for case in [
        "all_ones",
        "band_touching_edges",
        "interior_block",
        "single_pixel",
        "checker",
    ] {
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
    let band = Tensor::<B, 1>::from_data(TensorData::from(data.as_slice()), &Default::default())
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

    assert_close_scalar(
        "mask/masked_mean_banded",
        masked_mean(v.clone(), Some(band)),
        &read(&mut npz, "mask/masked_mean_banded"),
    );
    assert_close_scalar(
        "mask/masked_mean_none",
        masked_mean(v, None),
        &read(&mut npz, "mask/masked_mean_none"),
    );
}

#[test]
fn masked_mean_denominator_is_mask_area() {
    // 分母必须是掩码内像素数。如果错用全图像素数，掩码只盖一半时结果会差一倍。
    let x = Tensor::<B, 4>::ones([1, 1, 4, 4], &Default::default());
    let mut m = vec![0.0f32; 16];
    m[..4].fill(1.0); // 只有第一行
    let mask = Tensor::<B, 1>::from_data(TensorData::from(m.as_slice()), &Default::default())
        .reshape([1, 1, 4, 4]);
    let got: f32 = masked_mean(x, Some(mask))
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0];
    assert!(
        (got - 1.0).abs() < 1e-6,
        "全 1 场在任意掩码下均值都应是 1.0（分母是掩码面积），实得 {}",
        got
    );
}

// ---------------------------------------------------- 第 2 层：几何掩码

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
    assert!(
        spread < 0.01,
        "各分辨率覆盖率应基本一致，实得 {:?}（极差 {:.4}）",
        covs,
        spread
    );
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
    assert!(
        got.iter().all(|&v| v == 0.0 || v == 1.0),
        "重采样后必须仍是纯 0/1——任何插值都会造出没有物理意义的 0.5"
    );
}

#[test]
#[should_panic(expected = "出现 NaN")]
fn nan_output_is_not_silently_accepted() {
    // **NaN 不会被 `>` 判成超差**：`NaN > limit` 恒为 false，于是用「大于阈值才算超差」
    // 统计违规数时，一整张全 NaN 的输出会被判成「零处超差」而全绿——那等于整套黄金
    // 验收对 NaN 失效。这条测试守的是比对函数本身，不是某个算子
    let want =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let got = vec![f32::NAN; 4];
    assert_close_slice("nan-probe", &got, &want);
}

#[test]
#[should_panic(expected = "处不符")]
fn nan_mask_is_not_silently_accepted() {
    // 掩码比对同理
    let want =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![1.0f32, 0.0, 1.0, 0.0]).unwrap();
    assert_mask_eq("nan-mask", &[f32::NAN, 0.0, 1.0, 0.0], &want);
}
