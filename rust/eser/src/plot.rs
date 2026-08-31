//! 绘图（P7）：真值/预测/残差同色阶对比图、训练曲线。
//! 移植自 `utils/plot.py` / `utils/plot_3d.py`。
//!
//! ## 数据接口
//! 只吃 `&[f32]` + 形状 + 可选 `&[bool]` 掩码，不绑定 Burn 的 `Tensor`——绘图
//! 不该依赖训练后端。调用方自己把 Tensor `into_data().to_vec()` 拍平传进来。
//! Python 版 `_select_middle_depth` 的 3D→2D 取中间层同理：那是数据切片，不是
//! 绘图逻辑，留给调用方在传入前做。
//!
//! ## plotters 的能力边界（据此做的取舍）
//! - **没有 PDF 后端**，只有位图（`BitMapBackend`）与 SVG（`SVGBackend`，矢量）。
//!   本模块用**扩展名驱动**输出格式：`.svg` 当出版级矢量替代，`.png` 当位图。
//!   代价：SVG 不是 PDF，论文排版管线如果认 `\includegraphics{*.pdf}` 需要额外
//!   一步转换（`rsvg-convert` / `inkscape` 等），不是零成本替代。
//! - **截至 2026 年初无抗锯齿**：位图边缘会有锯齿，矢量 SVG 本身不存在这个问题
//!   （渲染器负责抗锯齿），但网格线/文字仍按 plotters 自身栅格化逻辑走。
//! - **没有内建 jet / coolwarm 色表**，也没有中文字体。本模块用解析公式手写
//!   jet（MATLAB 经典公式的分段线性近似）与 coolwarm（三点 RGB 线性插值，锚点
//!   取自 matplotlib coolwarm 的端点/中点颜色），标签一律用英文——中文需要外挂
//!   CJK 字体文件，这里没有可用字体资源，不做。
//! - **没有真正的 `contourf`**：matplotlib 的 contourf 在网格顶点间插值后画等值
//!   线包围的区域；这里退化成**逐像素色阶量化**（把归一化值取整量化进 N 个色阶
//!   区间），网格分辨率低时会看出方块感，和 matplotlib 的平滑等值线不是一回事。
//!
//! ## 色阶规则（照抄 Python 版行为，不自作主张改）
//! 真值/预测两行共用同一个 `[vmin, vmax]`（否则两行没法直接比较）；残差行单独用
//! 以 0 为中心的对称色阶 `[-rlim, rlim]`。掩码外的像素在渲染前就置成 NaN，色阶
//! 范围计算与渲染都跳过非有限值——域外的哨兵值不会把色阶拉偏，渲染时统一置灰。

use std::error::Error;
use std::path::Path;

use plotters::coord::Shift;
use plotters::prelude::*;

/// 绘图相关的统一错误类型：网络能来自 IO/字体/编码等多种子系统的错误，用
/// trait object 而不是自定义 enum——这里不需要按错误类型分支处理。
pub type PlotResult = Result<(), Box<dyn Error>>;

/// 掩码外像素的置灰色。
const MASKED_GRAY: RGBColor = RGBColor(200, 200, 200);

/// 三行标题，与 Python 版 `row_titles` 逐字对应。
const ROW_TITLES: [&str; 3] = ["CFD truth", "Surrogate", "Residual"];

/// 单张对比图的渲染风格。
#[derive(Debug, Clone, Copy)]
pub enum PlotStyle {
    /// 逐像素连续色阶，等价于 Python 版 `plot_fn='imshow'`。
    Imshow,
    /// 分档填色，模拟 `contourf(levels=linspace(lo,hi,levels))`：把归一化位置
    /// 量化进 `levels` 个色阶区间。是像素级量化，不是真正等值线插值——见模块
    /// 顶部说明。Python 版默认 51（对应 50 个色带）。
    ContourF { levels: usize },
}

/// jet 用于真值/预测行；coolwarm（发散）用于残差行——与 Python 版固定搭配一致，
/// 不做成调用方可选参数（没有需求）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Colormap {
    Jet,
    Coolwarm,
}

// ============================================================
// 纯函数：色阶计算 / 掩码剔除。全部不依赖渲染，可独立单元测试。
// ============================================================

/// 真值/预测共用的色阶范围：忽略非有限值。
///
/// 掩码外的像素在调用前已经被 [`apply_mask`] 置成 NaN，这里天然跳过——域外的
/// 哨兵值不会把色阶拉偏。全部非有限（比如整张图都被掩掉）时退回 `(0.0, 1.0)`，
/// 与 Python 版空栈分支一致。
///
/// 上下限相等（或极接近）时把上限顶高 `1e-8`，对应 Python 版
/// `np.isclose(vmin, vmax)` 分支（`np.isclose` 默认 `rtol=1e-5, atol=1e-8`）。
pub fn shared_scale(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in a.iter().chain(b.iter()) {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    if isclose(lo as f64, hi as f64) {
        // Python 版直接 `vmax = vmin + 1e-8`：那是 float64，1e-8 是可分辨的增量。
        // 这里数据是 f32——量级稍大时 f32 的 ULP 就超过 1e-8，`lo + 1e-8` 会被
        // 舍入吃掉，等于没顶高（实测 f32(2.0 + 1e-8) == f32(2.0)）。改用与 |lo|
        // 量级挂钩、且有绝对值下限的增量，保证在 f32 精度下真的能分辨出来。
        hi = lo + (lo.abs() * 1e-4).max(1e-6);
    }
    (lo, hi)
}

/// `numpy.isclose` 的等价实现：`|a-b| <= atol + rtol*|b|`，默认参数。
fn isclose(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

/// 残差行的对称色阶半宽：`[-rlim, rlim]`，以 0 为中心。
///
/// 取有限残差绝对值的最大者；全部非有限时退回 `1e-8`（对应 Python 版空数组
/// 分支），再与 `1e-12` 取较大值（避免色阶宽度为 0）。与 Python 版逐条对应：
/// ```python
/// res = residual[np.isfinite(residual)]
/// rlim = float(np.max(np.abs(res))) if res.size else 1e-8
/// rlim = max(rlim, 1e-12)
/// ```
pub fn symmetric_scale(residual: &[f32]) -> f32 {
    let mut rlim: f32 = 0.0;
    let mut any = false;
    for &v in residual {
        if v.is_finite() {
            rlim = rlim.max(v.abs());
            any = true;
        }
    }
    if !any {
        rlim = 1e-8;
    }
    rlim.max(1e-12)
}

/// 按布尔掩码把域外像素置 NaN。
///
/// `mask[i] == true` 表示该像素**在计算域内**（保留原值），`false` 表示域
/// 外——这是「计算域内位」而不是「屏蔽位」，与 Python 版 `mask.astype(bool)`
/// 语义一致。置成 NaN 之后，[`shared_scale`] / [`symmetric_scale`] 与渲染都会
/// 自动跳过这些像素，渲染时统一置灰。
pub fn apply_mask(data: &[f32], mask: &[bool]) -> Vec<f32> {
    data.iter()
        .zip(mask.iter())
        .map(|(&v, &keep)| if keep { v } else { f32::NAN })
        .collect()
}

/// 训练曲线里「最近 5 个点均值」——与 Python 版 `values[-5:].mean()` 一致，
/// 序列不足 5 个点时取全部。要求 `values` 非空。
fn tail_mean(values: &[f32]) -> f32 {
    let start = values.len().saturating_sub(5);
    let tail = &values[start..];
    tail.iter().sum::<f32>() / tail.len() as f32
}

// ============================================================
// 颜色映射：把归一化标量映射成 RGB。纯函数，可独立单元测试。
// ============================================================

/// 把标量值映射为颜色。非有限值（掩码外）固定灰色，不落进任何色阶——避免
/// NaN 被颜色公式当成 0 或极值处理。`ContourF` 时先把归一化位置量化进
/// `levels` 个区间。
fn value_color(v: f32, lo: f32, hi: f32, cm: Colormap, style: PlotStyle) -> RGBColor {
    if !v.is_finite() {
        return MASKED_GRAY;
    }
    let span = hi - lo;
    let mut t = if span.abs() < f32::EPSILON {
        0.5
    } else {
        ((v - lo) / span).clamp(0.0, 1.0)
    };
    if let PlotStyle::ContourF { levels } = style {
        t = quantize_levels(t, levels);
    }
    match cm {
        Colormap::Jet => jet_color(t),
        Colormap::Coolwarm => coolwarm_color(t),
    }
}

/// 把 `[0,1]` 量化到 `levels` 个离散色阶（色带中心对齐）。`levels <= 1` 时原样
/// 返回（没有分档意义）。
fn quantize_levels(t: f32, levels: usize) -> f32 {
    if levels <= 1 {
        return t;
    }
    let n = (levels - 1) as f32;
    (t * n).round() / n
}

/// MATLAB 经典 jet 色阶的解析近似（分段线性公式）。视觉上与 matplotlib 'jet'
/// 接近，但不是逐点相同的查找表——真正的 jet LUT 有 256 个手工调过的采样点。
fn jet_color(t: f32) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    let r = (1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0);
    RGBColor(
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

/// coolwarm 风格发散色阶：蓝 `(59,76,192)` → 白 `(221,221,221)` → 红
/// `(180,4,38)`，锚点取自 matplotlib coolwarm 的端点/中点颜色，用直接 RGB
/// 线性插值代替其真正的 CIE Lab 插值——「0 附近发白、两端发散」的趋势一致，
/// 过渡曲线不完全相同。
fn coolwarm_color(t: f32) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    const BLUE: (f32, f32, f32) = (59.0, 76.0, 192.0);
    const MID: (f32, f32, f32) = (221.0, 221.0, 221.0);
    const RED: (f32, f32, f32) = (180.0, 4.0, 38.0);
    let (c0, c1, frac) = if t < 0.5 {
        (BLUE, MID, t / 0.5)
    } else {
        (MID, RED, (t - 0.5) / 0.5)
    };
    let lerp = |a: f32, b: f32| a + (b - a) * frac;
    RGBColor(
        lerp(c0.0, c1.0).round() as u8,
        lerp(c0.1, c1.1).round() as u8,
        lerp(c0.2, c1.2).round() as u8,
    )
}

// ============================================================
// 渲染：真值/预测/残差三行对比图
// ============================================================

/// 每个通道预先算好的渲染输入：掩码外已置 NaN 的真值/预测/残差，以及三行各自
/// 的色阶范围。
struct ChannelPlot {
    target: Vec<f32>,
    prediction: Vec<f32>,
    residual: Vec<f32>,
    vmin: f32,
    vmax: f32,
    rlim: f32,
}

/// 三行对比图：真值 / 预测 / 残差，每列一个物理通道。
///
/// # 参数
/// - `target` / `prediction`：按 `(channels, height, width)` 行主序展平。
/// - `mask`：可选 `(height, width)` 计算域掩码，`true` 表示域内；域外置灰且
///   不参与色阶计算。
/// - `channel_names`：长度必须等于 `channels`，否则退化成 `ch0, ch1, ...`。
/// - `path` 扩展名决定输出格式：`.svg`（矢量，出版替代）或 `.png`（位图）。
///
/// 3D 场（`(C,D,H,W)`）的中间层切片由调用方在传入前完成，见模块顶部说明。
#[allow(clippy::too_many_arguments)]
pub fn plot_prediction(
    path: &Path,
    target: &[f32],
    prediction: &[f32],
    shape: (usize, usize, usize),
    mask: Option<&[bool]>,
    channel_names: &[&str],
    style: PlotStyle,
) -> PlotResult {
    let (channels, h, w) = shape;
    if channels == 0 || h == 0 || w == 0 {
        return Err(format!("形状不能含 0：{:?}", shape).into());
    }
    let hw = h * w;
    if target.len() != channels * hw || prediction.len() != channels * hw {
        return Err(format!(
            "数据长度与形状 {:?} 不符：target={}, prediction={}, 期望各 {}",
            shape,
            target.len(),
            prediction.len(),
            channels * hw
        )
        .into());
    }
    if let Some(m) = mask {
        if m.len() != hw {
            return Err(format!("掩码长度 {} 与 (h,w)=({},{}) 不符", m.len(), h, w).into());
        }
    }

    let names: Vec<String> = (0..channels)
        .map(|i| {
            if channel_names.len() == channels {
                channel_names[i].to_string()
            } else {
                format!("ch{}", i)
            }
        })
        .collect();

    let rows: Vec<ChannelPlot> = (0..channels)
        .map(|c| {
            let t = &target[c * hw..(c + 1) * hw];
            let p = &prediction[c * hw..(c + 1) * hw];
            let (t, p) = match mask {
                Some(m) => (apply_mask(t, m), apply_mask(p, m)),
                None => (t.to_vec(), p.to_vec()),
            };
            let residual: Vec<f32> = t.iter().zip(p.iter()).map(|(&a, &b)| a - b).collect();
            let (vmin, vmax) = shared_scale(&t, &p);
            let rlim = symmetric_scale(&residual);
            ChannelPlot {
                target: t,
                prediction: p,
                residual,
                vmin,
                vmax,
                rlim,
            }
        })
        .collect();

    // 主图按数据宽高比留正确的宽高（等价于 Python 版 `ax.set_aspect('equal')`）。
    let aspect = h as f64 / w as f64;
    let main_w: u32 = 220;
    let main_h: u32 = ((main_w as f64) * aspect).round().max(40.0) as u32;
    let bar_w: u32 = 46;
    let width = (main_w + bar_w) * channels as u32;
    let height = main_h * 3;

    match path.extension().and_then(|e| e.to_str()) {
        Some("svg") => render_prediction(
            SVGBackend::new(path, (width, height)).into_drawing_area(),
            &rows,
            w,
            h,
            &names,
            style,
            main_w,
            main_h,
        ),
        Some("png") => render_prediction(
            BitMapBackend::new(path, (width, height)).into_drawing_area(),
            &rows,
            w,
            h,
            &names,
            style,
            main_w,
            main_h,
        ),
        _ => Err(format!(
            "不支持的输出扩展名 {:?}：plotters 无 PDF 后端，出版级用 .svg（矢量），否则用 .png",
            path
        )
        .into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn render_prediction<DB>(
    root: DrawingArea<DB, Shift>,
    rows: &[ChannelPlot],
    w: usize,
    h: usize,
    names: &[String],
    style: PlotStyle,
    main_w: u32,
    main_h: u32,
) -> PlotResult
where
    DB: DrawingBackend,
    DB::ErrorType: std::fmt::Debug,
{
    let channels = rows.len();
    root.fill(&WHITE)
        .map_err(|e| format!("填充背景失败: {:?}", e))?;
    let grid = root.split_evenly((3, channels));

    for r in 0..3 {
        for (ci, row) in rows.iter().enumerate() {
            let cell = &grid[r * channels + ci];
            let (data, lo, hi, cm): (&[f32], f32, f32, Colormap) = match r {
                0 => (&row.target, row.vmin, row.vmax, Colormap::Jet),
                1 => (&row.prediction, row.vmin, row.vmax, Colormap::Jet),
                _ => (&row.residual, -row.rlim, row.rlim, Colormap::Coolwarm),
            };

            let (main, bar) = cell.split_horizontally(main_w);
            main.fill(&WHITE)
                .map_err(|e| format!("填充子图失败: {:?}", e))?;

            // 逐像素画色块：origin 取 (col, row) 直接对应数据下标，Cartesian2d
            // 默认向上为正，天然等价于 Python 版 `imshow(..., origin='lower')`。
            draw_pixels(&main, w, h, data, lo, hi, cm, style)?;
            draw_colorbar(&bar, lo, hi, cm, style, main_h)?;

            if r == 0 {
                draw_corner_text(&main, &names[ci], 2)?;
            }
            if ci == 0 {
                let y = if r == 0 { 16 } else { 2 };
                draw_corner_text(&main, ROW_TITLES[r], y)?;
            }
        }
    }

    root.present().map_err(|e| format!("写出失败: {:?}", e))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn draw_pixels<DB>(
    main: &DrawingArea<DB, Shift>,
    w: usize,
    h: usize,
    data: &[f32],
    lo: f32,
    hi: f32,
    cm: Colormap,
    style: PlotStyle,
) -> PlotResult
where
    DB: DrawingBackend,
    DB::ErrorType: std::fmt::Debug,
{
    let mut chart = ChartBuilder::on(main)
        .margin(1)
        .x_label_area_size(0)
        .y_label_area_size(0)
        .build_cartesian_2d(0i32..w as i32, 0i32..h as i32)
        .map_err(|e| format!("建图失败: {:?}", e))?;

    chart
        .draw_series((0..h).flat_map(|row| {
            (0..w).map(move |col| {
                let v = data[row * w + col];
                let color = value_color(v, lo, hi, cm, style);
                Rectangle::new(
                    [(col as i32, row as i32), (col as i32 + 1, row as i32 + 1)],
                    color.filled(),
                )
            })
        }))
        .map_err(|e| format!("画像素失败: {:?}", e))?;
    Ok(())
}

fn draw_colorbar<DB>(
    area: &DrawingArea<DB, Shift>,
    lo: f32,
    hi: f32,
    cm: Colormap,
    style: PlotStyle,
    height_px: u32,
) -> PlotResult
where
    DB: DrawingBackend,
    DB::ErrorType: std::fmt::Debug,
{
    area.fill(&WHITE)
        .map_err(|e| format!("填充色阶条失败: {:?}", e))?;
    let n = match style {
        PlotStyle::ContourF { levels } => levels.max(2),
        PlotStyle::Imshow => 100,
    };

    let mut chart = ChartBuilder::on(area)
        .margin_left(4)
        .margin_right(2)
        .x_label_area_size(0)
        .y_label_area_size(0)
        .build_cartesian_2d(0f32..1f32, lo..hi)
        .map_err(|e| format!("建色阶条失败: {:?}", e))?;

    chart
        .draw_series((0..n).map(|i| {
            let v0 = lo + (hi - lo) * i as f32 / n as f32;
            let v1 = lo + (hi - lo) * (i + 1) as f32 / n as f32;
            let color = value_color((v0 + v1) * 0.5, lo, hi, cm, style);
            Rectangle::new([(0f32, v0), (1f32, v1)], color.filled())
        }))
        .map_err(|e| format!("画色阶条失败: {:?}", e))?;

    // 简化：只标端点数值，不做 matplotlib ScalarFormatter 的科学计数偏移记号。
    area.draw(&Text::new(
        fmt_value(hi),
        (2, 2),
        ("sans-serif", 10).into_font(),
    ))
    .map_err(|e| format!("画色阶上限失败: {:?}", e))?;
    let bottom_y = (height_px as i32 - 12).max(2);
    area.draw(&Text::new(
        fmt_value(lo),
        (2, bottom_y),
        ("sans-serif", 10).into_font(),
    ))
    .map_err(|e| format!("画色阶下限失败: {:?}", e))?;
    Ok(())
}

fn draw_corner_text<DB>(area: &DrawingArea<DB, Shift>, text: &str, y: i32) -> PlotResult
where
    DB: DrawingBackend,
    DB::ErrorType: std::fmt::Debug,
{
    let backing_w = 6 + text.len() as i32 * 6;
    area.draw(&Rectangle::new(
        [(0, y - 1), (backing_w, y + 12)],
        WHITE.mix(0.75).filled(),
    ))
    .map_err(|e| format!("画标签底色失败: {:?}", e))?;
    area.draw(&Text::new(
        text.to_string(),
        (2, y),
        ("sans-serif", 11).into_font(),
    ))
    .map_err(|e| format!("画标签失败: {:?}", e))?;
    Ok(())
}

fn fmt_value(v: f32) -> String {
    if v == 0.0 {
        "0".to_string()
    } else {
        format!("{:.2e}", v)
    }
}

// ============================================================
// 渲染：训练曲线（R² / RMSE / 物理残差 RMS）
// ============================================================

/// 一条或多条命名序列画在同一张图里；legend 附最近 5 点均值——与 Python 版
/// `_curve` 一致。序列长度可以互不相同（`--post` 模式下训练/验证长度不一致时
/// Python 版就是这样处理的），空序列直接跳过。全部序列都空时不产生输出文件，
/// 与 Python 版 `if not drawn: plt.close(); return` 一致。
pub fn plot_curves(path: &Path, ylabel: &str, series: &[(&str, &[f32])]) -> PlotResult {
    let drawable: Vec<(&str, &[f32])> = series
        .iter()
        .filter(|(_, v)| !v.is_empty())
        .copied()
        .collect();
    if drawable.is_empty() {
        return Ok(());
    }

    match path.extension().and_then(|e| e.to_str()) {
        Some("svg") => render_curves(
            SVGBackend::new(path, (900, 600)).into_drawing_area(),
            ylabel,
            &drawable,
        ),
        Some("png") => render_curves(
            BitMapBackend::new(path, (900, 600)).into_drawing_area(),
            ylabel,
            &drawable,
        ),
        _ => Err(format!(
            "不支持的输出扩展名 {:?}：plotters 无 PDF 后端，出版级用 .svg（矢量），否则用 .png",
            path
        )
        .into()),
    }
}

fn render_curves<DB>(
    root: DrawingArea<DB, Shift>,
    ylabel: &str,
    series: &[(&str, &[f32])],
) -> PlotResult
where
    DB: DrawingBackend,
    DB::ErrorType: std::fmt::Debug,
{
    root.fill(&WHITE)
        .map_err(|e| format!("填充背景失败: {:?}", e))?;

    let max_len = series
        .iter()
        .map(|(_, v)| v.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for (_, v) in series {
        for &x in *v {
            if x.is_finite() {
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
    }
    // **区间兜底走 curve_y_range**，不要在这里再抄一份：抽出去只给测试用、
    // 绘图代码留一份拷贝的话，测的和用的是两份代码，改坏哪一份测试都发现不了
    let (lo, hi) = curve_y_range_from(lo, hi);
    let pad = (hi - lo) * 0.05;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(1f32..(max_len.max(2) as f32), (lo - pad)..(hi + pad))
        .map_err(|e| format!("建图失败: {:?}", e))?;

    chart
        .configure_mesh()
        .y_desc(ylabel)
        .x_desc("logged step (every log_freq epochs)")
        .draw()
        .map_err(|e| format!("画网格失败: {:?}", e))?;

    for (i, (label, values)) in series.iter().enumerate() {
        let tail = tail_mean(values);
        let pts: Vec<(f32, f32)> = values
            .iter()
            .enumerate()
            .map(|(k, &v)| ((k + 1) as f32, v))
            .collect();
        chart
            .draw_series(LineSeries::new(pts, Palette99::pick(i).stroke_width(2)))
            .map_err(|e| format!("画线失败: {:?}", e))?
            .label(format!("{}: {:.4}", label, tail))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], Palette99::pick(i)));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|e| format!("画图例失败: {:?}", e))?;

    root.present().map_err(|e| format!("写出失败: {:?}", e))?;
    Ok(())
}

/// 曲线图 y 轴区间的兜底（供测试与调用方核对）。退化区间必须被撑开到 f32 能分辨。
pub fn curve_y_range(values: &[f32]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for v in values {
        if v.is_finite() {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    curve_y_range_from(lo, hi)
}

/// 已经算出 lo/hi 之后的区间兜底。**唯一的一份实现**，绘图与测试共用。
fn curve_y_range_from(lo: f32, hi: f32) -> (f32, f32) {
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    if (hi - lo).abs() < 1e-8 {
        // 同一个 f32 陷阱（对照 shared_scale）：数据是 f32，|v| ≳ 0.25 时
        // `hi += 1e-8` 会被舍入吃掉，区间仍是零宽。plotters 对零宽区间返回的是
        // 相对偏移而不是绝对像素，整条曲线会被画到画布外，出来一张只有坐标轴的空图
        let bump = (hi.abs().max(lo.abs()) * 1e-4).max(1e-6);
        (lo - bump, hi + bump)
    } else {
        (lo, hi)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn constant_curve_gets_a_representable_y_range() {
        // 常量序列的 y 轴区间必须在 **f32 精度下真的能分辨**。
        // `hi += 1e-8` 在 |v| ≳ 0.25 时会被舍入吃掉，区间仍是零宽，
        // plotters 于是把整条曲线画到画布外，出来一张空图
        for v in [0.0f32, 0.25, 1.0, 42.0, -137.5, 1e4] {
            let (lo, hi) = curve_y_range(&[v; 5]);
            assert!(
                hi > lo,
                "常量 {} 的 y 轴区间仍是零宽（{} ..= {}）",
                v,
                lo,
                hi
            );
            assert!(lo <= v && v <= hi, "常量 {} 落在区间之外", v);
        }
    }

    #[test]
    fn all_non_finite_curve_falls_back_to_unit_range() {
        let (lo, hi) = curve_y_range(&[f32::NAN, f32::INFINITY]);
        assert_eq!((lo, hi), (0.0, 1.0), "全非有限时应退回单位区间");
    }

    use super::*;

    // ---------- shared_scale ----------

    #[test]
    fn shared_scale_basic_min_max_across_both_arrays() {
        let a = [1.0_f32, 5.0, 3.0];
        let b = [0.0_f32, 2.0, 9.0];
        assert_eq!(shared_scale(&a, &b), (0.0, 9.0));
    }

    #[test]
    fn shared_scale_ignores_non_finite() {
        let a = [1.0_f32, f32::NAN, 5.0];
        let b = [f32::NAN, 2.0, f32::NAN];
        assert_eq!(shared_scale(&a, &b), (1.0, 5.0));
    }

    #[test]
    fn shared_scale_bumps_equal_bounds() {
        let a = [2.0_f32, 2.0, 2.0];
        let (lo, hi) = shared_scale(&a, &a);
        assert_eq!(lo, 2.0);
        assert!(hi > lo, "vmin==vmax 必须把 vmax 顶高，否则色阶宽度为 0");
    }

    #[test]
    fn shared_scale_all_masked_falls_back_to_unit_range() {
        let a = [f32::NAN, f32::NAN];
        assert_eq!(shared_scale(&a, &a), (0.0, 1.0));
    }

    /// 掩码剔除的判别性测试：掩码外的值（这里是 100.0）必须被排除在色阶范围
    /// 之外，否则色阶会被域外的哨兵值拉偏。
    #[test]
    fn apply_mask_excludes_masked_values_from_scale() {
        let data = [1.0_f32, 100.0, 3.0, 4.0];
        let mask = [true, false, true, true];
        let masked = apply_mask(&data, &mask);
        assert!(masked[1].is_nan());
        let (lo, hi) = shared_scale(&masked, &masked);
        assert_eq!(lo, 1.0);
        assert_eq!(hi, 4.0, "掩码外的 100.0 不应该进入色阶范围");
    }

    // ---------- symmetric_scale ----------

    #[test]
    fn symmetric_scale_takes_max_abs_of_finite_residuals() {
        let r = [-3.0_f32, 2.0, -1.0];
        assert_eq!(symmetric_scale(&r), 3.0);
    }

    #[test]
    fn symmetric_scale_ignores_non_finite() {
        let r = [f32::NAN, -7.0, f32::INFINITY, 2.0];
        assert_eq!(symmetric_scale(&r), 7.0);
    }

    #[test]
    fn symmetric_scale_empty_or_all_masked_falls_back() {
        let empty: [f32; 0] = [];
        assert_eq!(symmetric_scale(&empty), 1e-8);
        let all_nan = [f32::NAN, f32::NAN];
        assert_eq!(symmetric_scale(&all_nan), 1e-8);
    }

    #[test]
    fn symmetric_scale_floors_tiny_values() {
        let r = [1e-15_f32, -1e-16];
        assert_eq!(symmetric_scale(&r), 1e-12);
    }

    /// 组合判别性测试：残差为 0 的点在对称色阶下必须落在色阶中点（coolwarm
    /// 的白灰锚点）。如果 symmetric_scale 错误地改成用 [min, max] 而不是对称
    /// 的 [-rlim, rlim]，非对称分布的残差会让 0 偏离中点，这里就会失败。
    #[test]
    fn residual_zero_maps_to_colorbar_center() {
        let residual = [-3.0_f32, 2.0, -1.0, 0.0]; // 非对称分布：min=-3, max=2
        let rlim = symmetric_scale(&residual);
        let c = value_color(0.0, -rlim, rlim, Colormap::Coolwarm, PlotStyle::Imshow);
        assert!(
            c.0.abs_diff(221) <= 1 && c.1.abs_diff(221) <= 1 && c.2.abs_diff(221) <= 1,
            "0 应落在 coolwarm 中点(221,221,221)附近，实得 {:?}",
            c
        );
    }

    // ---------- 颜色映射 ----------

    #[test]
    fn value_color_masks_non_finite_to_gray() {
        let c = value_color(f32::NAN, 0.0, 1.0, Colormap::Jet, PlotStyle::Imshow);
        assert_eq!(c, MASKED_GRAY);
        let c = value_color(
            f32::INFINITY,
            0.0,
            1.0,
            Colormap::Coolwarm,
            PlotStyle::Imshow,
        );
        assert_eq!(c, MASKED_GRAY);
    }

    #[test]
    fn jet_color_endpoints_are_blue_and_red_dominant() {
        let blue_end = jet_color(0.0);
        let red_end = jet_color(1.0);
        assert!(blue_end.2 > blue_end.0, "t=0 应蓝色主导: {:?}", blue_end);
        assert!(red_end.0 > red_end.2, "t=1 应红色主导: {:?}", red_end);
    }

    #[test]
    fn quantize_levels_produces_expected_bin_count() {
        let levels = 5;
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..=1000 {
            let t = i as f32 / 1000.0;
            let q = quantize_levels(t, levels);
            seen.insert((q * 1e6).round() as i64);
        }
        assert_eq!(seen.len(), levels, "5 档量化应恰好落在 5 个离散值上");
    }

    #[test]
    fn quantize_levels_passthrough_when_not_enough_levels() {
        assert_eq!(quantize_levels(0.37, 1), 0.37);
        assert_eq!(quantize_levels(0.37, 0), 0.37);
    }

    // ---------- 训练曲线辅助 ----------

    #[test]
    fn tail_mean_uses_last_five_or_fewer_points() {
        assert_eq!(tail_mean(&[1.0, 2.0, 3.0]), 2.0);
        let v: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        assert_eq!(tail_mean(&v), 8.0); // 最后 5 个：6..10，均值 8
    }

    // ---------- 形状校验 ----------

    #[test]
    fn plot_prediction_rejects_mismatched_length() {
        let target = vec![0.0_f32; 4];
        let prediction = vec![0.0_f32; 3]; // 故意错一个
        let err = plot_prediction(
            Path::new("unused.svg"),
            &target,
            &prediction,
            (1, 2, 2),
            None,
            &[],
            PlotStyle::Imshow,
        );
        assert!(err.is_err());
    }

    #[test]
    fn plot_prediction_rejects_zero_shape() {
        let err = plot_prediction(
            Path::new("unused.svg"),
            &[],
            &[],
            (1, 0, 2),
            None,
            &[],
            PlotStyle::Imshow,
        );
        assert!(err.is_err());
    }
}
