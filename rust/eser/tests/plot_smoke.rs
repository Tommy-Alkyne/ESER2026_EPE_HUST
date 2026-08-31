//! 绘图冒烟测试（P7）：用合成数据实际生成一组图片，供人工目视比对。
//!
//! 不对照黄金数据——`utils/plot.py` 没有导出过绘图相关的参考张量（P0 的黄金
//! 数据集只覆盖算子/物理残差/网络/训练，不含绘图），色阶/掩码的数值行为已经
//! 在 `src/plot.rs` 的纯函数单元测试里验证过。这里只验证「真的能画出图、文件
//! 落盘」，产物写到系统临时目录，不污染仓库。

use std::fs;
use std::path::PathBuf;

use eser::geometry::{default_t_geometry, rect_mask};
use eser::plot::{plot_curves, plot_prediction, PlotStyle};

fn out_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("eser_plot_smoke");
    fs::create_dir_all(&dir).expect("创建临时输出目录失败");
    dir
}

/// 合成一组 (channels, h, w) 场：真值是光滑函数，预测=真值+结构化小扰动，
/// 残差因此非零且有空间结构，比纯随机噪声更适合目视检查色阶/掩码是否对齐。
fn synth_fields(channels: usize, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut target = vec![0.0f32; channels * h * w];
    let mut prediction = vec![0.0f32; channels * h * w];
    for c in 0..channels {
        for row in 0..h {
            for col in 0..w {
                let x = col as f64 / w as f64;
                let y = row as f64 / h as f64;
                let t = (2.0 * std::f64::consts::PI * x * (c as f64 + 1.0)).sin()
                    * (2.0 * std::f64::consts::PI * y).cos()
                    + 0.3 * c as f64;
                let noise = 0.06 * (10.0 * x + 7.0 * y + c as f64).sin();
                let idx = c * h * w + row * w + col;
                target[idx] = t as f32;
                prediction[idx] = (t + noise) as f32;
            }
        }
    }
    (target, prediction)
}

/// T 形计算域掩码（复用 `eser::geometry`，与几何层验收用的是同一套实现）。
fn t_mask(h: usize, w: usize) -> Vec<bool> {
    let segments = default_t_geometry();
    let raster = rect_mask(h, w, &segments).expect("生成 T 形掩码失败");
    raster.iter().map(|&v| v > 0.5).collect()
}

fn assert_nonempty_file(path: &std::path::Path) {
    let meta = fs::metadata(path).unwrap_or_else(|e| panic!("产物没有落盘 {:?}: {}", path, e));
    assert!(meta.len() > 0, "产物是空文件: {:?}", path);
}

/// imshow 风格 + T 形掩码，PNG 输出。三通道验证「三行同色阶」与「掩码置灰」。
#[test]
fn imshow_with_mask_png() {
    let (h, w) = (65, 65);
    let (target, prediction) = synth_fields(3, h, w);
    let mask = t_mask(h, w);
    let path = out_dir().join("imshow_mask.png");
    plot_prediction(
        &path,
        &target,
        &prediction,
        (3, h, w),
        Some(&mask),
        &["P", "Vx", "Vy"],
        PlotStyle::Imshow,
    )
    .expect("imshow+掩码 出图失败");
    assert_nonempty_file(&path);
}

/// contourf 风格（51 档量化）+ 掩码，SVG 矢量输出（PDF 的替代方案）。
#[test]
fn contourf_with_mask_svg() {
    let (h, w) = (65, 65);
    let (target, prediction) = synth_fields(3, h, w);
    let mask = t_mask(h, w);
    let path = out_dir().join("contourf_mask.svg");
    plot_prediction(
        &path,
        &target,
        &prediction,
        (3, h, w),
        Some(&mask),
        &["P", "Vx", "Vy"],
        PlotStyle::ContourF { levels: 51 },
    )
    .expect("contourf+掩码 出图失败");
    assert_nonempty_file(&path);
}

/// 单通道、无掩码：验证「支持任意通道数」（Python 旧版硬编码 3 通道会在这里
/// 直接 IndexError，是 plot.py 文档里点名修过的缺陷）。
#[test]
fn single_channel_no_mask_svg() {
    let (h, w) = (40, 80); // 非正方形网格，顺带验证宽高比处理
    let (target, prediction) = synth_fields(1, h, w);
    let path = out_dir().join("single_channel.svg");
    plot_prediction(
        &path,
        &target,
        &prediction,
        (1, h, w),
        None,
        &["P"],
        PlotStyle::Imshow,
    )
    .expect("单通道出图失败");
    assert_nonempty_file(&path);
}

/// 训练曲线：R²/RMSE 各画一张；训练/验证序列长度不一致（`--post` 模式下的真实
/// 场景）；全空序列不应产生文件。
#[test]
fn training_curves() {
    let dir = out_dir();

    let train_r2: Vec<f32> = (0..20).map(|i| 1.0 - 0.9 / (i as f32 + 1.0)).collect();
    let test_r2: Vec<f32> = (0..12).map(|i| 1.0 - 1.1 / (i as f32 + 1.0)).collect(); // 长度不同
    let r2_path = dir.join("r2.svg");
    plot_curves(
        &r2_path,
        "R2-score",
        &[("Train", &train_r2), ("Test", &test_r2)],
    )
    .expect("R2 曲线出图失败");
    assert_nonempty_file(&r2_path);

    let train_rmse: Vec<f32> = (0..20).map(|i| 1.0 / (i as f32 + 1.0)).collect();
    let test_rmse: Vec<f32> = (0..20).map(|i| 1.2 / (i as f32 + 1.0)).collect();
    let rmse_path = dir.join("rmse.png");
    plot_curves(
        &rmse_path,
        "RMSE",
        &[("Train", &train_rmse), ("Test", &test_rmse)],
    )
    .expect("RMSE 曲线出图失败");
    assert_nonempty_file(&rmse_path);

    // 全空序列：不应该产生文件（与 Python 版 `_curve` 一致）。
    let empty_path = dir.join("should_not_exist.svg");
    let _ = fs::remove_file(&empty_path);
    let empty: Vec<f32> = vec![];
    plot_curves(
        &empty_path,
        "unused",
        &[("Train", &empty), ("Test", &empty)],
    )
    .expect("空序列理应正常返回 Ok，而不是报错");
    assert!(
        !empty_path.exists(),
        "全空序列不该产生输出文件，但 {:?} 被创建了",
        empty_path
    );
}

/// 形状/掩码不匹配要报错，而不是 panic 或悄悄画错。
#[test]
fn rejects_mask_shape_mismatch() {
    let (h, w) = (10, 10);
    let (target, prediction) = synth_fields(1, h, w);
    let bad_mask = vec![true; h * w - 1]; // 故意错一格
    let path = out_dir().join("should_not_exist_bad_mask.png");
    let _ = fs::remove_file(&path);
    let err = plot_prediction(
        &path,
        &target,
        &prediction,
        (1, h, w),
        Some(&bad_mask),
        &["P"],
        PlotStyle::Imshow,
    );
    assert!(err.is_err(), "掩码长度不匹配应该报错");
    assert!(!path.exists(), "报错后不应该留下产物文件");
}
