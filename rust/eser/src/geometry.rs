//! 计算域几何：矩形段拼接（管道类）与 `.npy` 栅格（任意形状，如地铁站平面）。
//!
//! 移植自 `utils/geometry.py`。几何用**归一化比例**描述，任意分辨率下覆盖同一
//! 物理区域——旧版把掩码写死成 137x137 的像素坐标，换分辨率时静默夹取，掩码
//! 几何完全错位却不报错（65x65 覆盖 63.8%，257x257 只剩 12.8%）。

/// 一段矩形，用 `(y0, y1, x0, x1)` 的归一化比例描述，取值范围 [0, 1]，右开区间。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub y0: f64,
    pub y1: f64,
    pub x0: f64,
    pub x1: f64,
}

impl Segment {
    pub const fn new(y0: f64, y1: f64, x0: f64, x1: f64) -> Self {
        Self { y0, y1, x0, x1 }
    }
}

/// 默认 T 型几何。比例由原 137x137 硬编码坐标反推，因此在 137x137 上与旧版逐像素等价。
pub fn default_t_geometry() -> Vec<Segment> {
    vec![
        // 竖管：原 rows 0:136, cols 0:35 @137
        Segment::new(0.0, 136.0 / 137.0, 0.0, 35.0 / 137.0),
        // 横管：原 rows 51:87, cols 35:137 @137
        Segment::new(51.0 / 137.0, 87.0 / 137.0, 35.0 / 137.0, 1.0),
    ]
}

/// Python `int(round(x))` 的等价实现——**银行家舍入**（round-half-to-even）。
///
/// Rust 的 `f64::round()` 是「四舍五入远离零」，`round(0.5)` 给 1；而 Python 3 给 0。
/// 几何比例乘上分辨率后正好落在 .5 的情况是存在的，两者不一致会让掩码边界差一格，
/// 而且**不会报任何错**。这里显式对齐 Python 的语义。
///
/// 只处理非负输入（几何比例在 [0,1]，乘以分辨率后恒非负）。
fn py_round_nonneg(x: f64) -> usize {
    debug_assert!(x >= 0.0, "几何比例不应为负: {}", x);
    let floor = x.floor();
    let frac = x - floor;
    let fi = floor as usize;
    if frac == 0.5 {
        // 恰好落在 .5：向偶数靠。这一条就是 Python 与 Rust 的分歧点。
        if fi.is_multiple_of(2) { fi } else { fi + 1 }
    } else if frac > 0.5 {
        fi + 1
    } else {
        fi
    }
}

/// 校验矩形段合法性。返回错误而不是 panic，方便上层给出可读提示。
pub fn validate_segments(segments: &[Segment]) -> Result<(), String> {
    if segments.is_empty() {
        return Err("几何定义为空".into());
    }
    for (i, s) in segments.iter().enumerate() {
        for (name, v) in [("y0", s.y0), ("y1", s.y1), ("x0", s.x0), ("x1", s.x1)] {
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("第 {} 段的 {} = {} 超出 [0, 1]", i, name, v));
            }
        }
        if s.y1 <= s.y0 || s.x1 <= s.x0 {
            return Err(format!("第 {} 段区间为空: {:?}", i, s));
        }
    }
    Ok(())
}

/// 由矩形段生成 (h, w) 的 0/1 掩码，行优先展平。
pub fn rect_mask(h: usize, w: usize, segments: &[Segment]) -> Result<Vec<f32>, String> {
    validate_segments(segments)?;
    let mut mask = vec![0.0f32; h * w];
    for s in segments {
        let y0 = py_round_nonneg(s.y0 * h as f64).min(h);
        let y1 = py_round_nonneg(s.y1 * h as f64).min(h);
        let x0 = py_round_nonneg(s.x0 * w as f64).min(w);
        let x1 = py_round_nonneg(s.x1 * w as f64).min(w);
        for y in y0..y1 {
            for x in x0..x1 {
                mask[y * w + x] = 1.0;
            }
        }
    }
    if mask.iter().all(|&v| v == 0.0) {
        return Err(format!("在 {}x{} 网格上生成的掩码为空", h, w));
    }
    Ok(mask)
}

/// 校验栅格掩码：必须是纯 0/1 且非全零。
pub fn validate_raster(mask: &[f32], h: usize, w: usize) -> Result<(), String> {
    if mask.len() != h * w {
        return Err(format!("栅格长度 {} 与形状 {}x{} 不符", mask.len(), h, w));
    }
    if let Some(&bad) = mask.iter().find(|&&v| v != 0.0 && v != 1.0) {
        return Err(format!("栅格掩码只能含 0/1，出现了 {}", bad));
    }
    if mask.iter().all(|&v| v == 0.0) {
        return Err("栅格掩码全为 0，计算域为空".into());
    }
    Ok(())
}

/// 栅格掩码最近邻重采样到 (h, w)。
///
/// 掩码是 0/1 的，任何插值都会在边界造出 0.5 这种没有物理意义的值，所以只能用
/// 最近邻。下标计算与 Python 的 `(arange(n) * src) // n` 逐位一致（整数向下取整）。
pub fn resample_raster(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>, String> {
    validate_raster(src, src_h, src_w)?;
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        let sy = ((y * src_h) / h.max(1)).min(src_h - 1);
        for x in 0..w {
            let sx = ((x * src_w) / w.max(1)).min(src_w - 1);
            out[y * w + x] = src[sy * src_w + sx];
        }
    }
    if out.iter().all(|&v| v == 0.0) {
        return Err(format!(
            "栅格掩码重采样到 {}x{} 后为空，说明目标网格远粗于掩码里的通道宽度",
            h, w
        ));
    }
    Ok(out)
}

/// 掩码覆盖率，用于启动时打印自检信息。
pub fn coverage(mask: &[f32]) -> f64 {
    if mask.is_empty() {
        return 0.0;
    }
    mask.iter().map(|&v| v as f64).sum::<f64>() / mask.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_round_matches_bankers_rounding() {
        // 这几个值 Rust 的 f64::round() 会给出不同答案
        assert_eq!(py_round_nonneg(0.5), 0, "0.5 应向偶数舍入到 0");
        assert_eq!(py_round_nonneg(1.5), 2);
        assert_eq!(py_round_nonneg(2.5), 2, "2.5 应向偶数舍入到 2");
        assert_eq!(py_round_nonneg(3.5), 4);
        assert_eq!(py_round_nonneg(0.4), 0);
        assert_eq!(py_round_nonneg(0.6), 1);
        assert_eq!(py_round_nonneg(7.0), 7);
    }

    #[test]
    fn rejects_bad_geometry() {
        assert!(rect_mask(10, 10, &[]).is_err(), "空几何应被拒绝");
        assert!(rect_mask(10, 10, &[Segment::new(0.5, 0.2, 0.0, 1.0)]).is_err(),
                "空区间应被拒绝");
        assert!(rect_mask(10, 10, &[Segment::new(-0.1, 0.5, 0.0, 1.0)]).is_err(),
                "越界比例应被拒绝");
    }

    #[test]
    fn rejects_bad_raster() {
        assert!(validate_raster(&[0.0; 9], 3, 3).is_err(), "全 0 栅格应被拒绝");
        assert!(validate_raster(&[0.5; 9], 3, 3).is_err(), "非 0/1 栅格应被拒绝");
        assert!(validate_raster(&[1.0; 8], 3, 3).is_err(), "长度不符应被拒绝");
    }
}
