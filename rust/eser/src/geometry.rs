//! 计算域几何：矩形段拼接（管道类）与 `.npy` 栅格（任意形状，如地铁站平面）。
//!
//! 栅格支持**边界分区标签**（见 [`Zone`]）：0 域外 / 1 流体 / 2 壁 / 3 入口 / 4 出口。
//! 只含 0/1 的旧掩码逐位等价于「全是流体」，域掩码不变。
//!
//! 移植自 `utils/geometry.py`。几何用**归一化比例**描述，任意分辨率下覆盖同一
//! 物理区域——旧版把掩码写死成 137x137 的像素坐标，换分辨率时静默夹取，掩码
//! 几何完全错位却不报错（65x65 覆盖 63.8%，257x257 只剩 12.8%）。

/// 栅格掩码的分区标签。
///
/// 加它是因为**旧的 0/1 掩码分不出「墙」和「送风口」**——两者同为 0，物理损失
/// 腐蚀一圈后完全不看边界，网络对「风机开多大」没有任何物理约束，只能靠数据硬记。
///
/// **向后兼容是硬约束**：只含 0/1 的栅格里 1 一律解读成 [`Zone::Fluid`]，
/// [`GeometrySpec::mask`](crate::predict::GeometrySpec::mask) 的输出逐位不变。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Zone {
    /// 域外，不参与任何计算
    Outside = 0,
    /// 流体计算域。**只有它算「域内」**——所有既有损失的掩码就是它
    Fluid = 1,
    /// 固壁：不可穿透 `u·n = 0` + 标量无通量 `∂C/∂n = 0`
    Wall = 2,
    /// 送风口：给定速度 `u = u_given`
    Inlet = 3,
    /// 排风口 / 自由出流
    Outlet = 4,
}

impl Zone {
    /// 从栅格里的浮点标签取枚举。非整数或越界都是错误——**静默当成别的分区，
    /// 边界条件就会加在错误的格点上，而且不报错**。
    pub fn from_label(v: f32) -> Result<Self, String> {
        match v {
            0.0 => Ok(Self::Outside),
            1.0 => Ok(Self::Fluid),
            2.0 => Ok(Self::Wall),
            3.0 => Ok(Self::Inlet),
            4.0 => Ok(Self::Outlet),
            other => Err(format!(
                "栅格标签只能是 0(域外)/1(流体)/2(壁)/3(入口)/4(出口)，出现了 {}",
                other
            )),
        }
    }

    pub fn label(self) -> f32 {
        self as u8 as f32
    }
}

/// 从分区标签场取某一类的 0/1 指示场。
pub fn zone_indicator(zones: &[f32], z: Zone) -> Vec<f32> {
    let lab = z.label();
    zones.iter().map(|&v| if v == lab { 1.0 } else { 0.0 }).collect()
}

/// 分区标签场里每一类的格点数，按 [`Zone`] 顺序返回，用于启动自检打印。
pub fn zone_counts(zones: &[f32]) -> [usize; 5] {
    let mut c = [0usize; 5];
    for &v in zones {
        let i = v as usize;
        if i < 5 {
            c[i] += 1;
        }
    }
    c
}

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

/// 校验栅格掩码：取值必须是 [`Zone`] 的合法标签，且流体区非空。
///
/// 旧行为（只认 0/1）是这个的特例：0.5 这类非整数、长度不符、流体为空全都照样报错，
/// 放宽的只是「允许出现 2/3/4 三个边界标签」。
pub fn validate_raster(mask: &[f32], h: usize, w: usize) -> Result<(), String> {
    if mask.len() != h * w {
        return Err(format!("栅格长度 {} 与形状 {}x{} 不符", mask.len(), h, w));
    }
    for &v in mask {
        Zone::from_label(v)?;
    }
    if !mask.iter().any(|&v| v == Zone::Fluid.label()) {
        return Err("栅格里没有一个流体格点（标签 1），计算域为空".into());
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
    resample_raster_planes(src, 1, src_h, src_w, h, w)
}

/// [`resample_raster`] 的多层版本：逐层各自重采样，**层数不动**。
///
/// 深度方向不做最近邻：层与层是物理上不同的楼层（站厅 / 站台 / 隧道），插值或抽取
/// 出来的「中间层」没有任何物理意义。
///
/// 「非空」只对**整个体**要求一次，不是逐层要求：地铁站里整层是实心楼板完全合法，
/// 逐层要求会把这种几何直接拒掉。
pub fn resample_raster_planes(
    src: &[f32],
    d: usize,
    src_h: usize,
    src_w: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>, String> {
    if src.len() != d * src_h * src_w {
        return Err(format!(
            "栅格长度 {} 与形状 {}x{}x{} 不符",
            src.len(),
            d,
            src_h,
            src_w
        ));
    }
    for &v in src {
        Zone::from_label(v)?;
    }
    if !src.iter().any(|&v| v == Zone::Fluid.label()) {
        return Err("栅格里没有一个流体格点（标签 1），计算域为空".into());
    }

    let mut out = vec![0.0f32; d * h * w];
    for k in 0..d {
        let plane = &src[k * src_h * src_w..(k + 1) * src_h * src_w];
        for y in 0..h {
            let sy = ((y * src_h) / h.max(1)).min(src_h - 1);
            for x in 0..w {
                let sx = ((x * src_w) / w.max(1)).min(src_w - 1);
                out[(k * h + y) * w + x] = plane[sy * src_w + sx];
            }
        }
    }
    if !out.iter().any(|&v| v == Zone::Fluid.label()) {
        return Err(format!(
            "栅格掩码重采样到 {}x{} 后流体区为空，说明目标网格远粗于掩码里的通道宽度",
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
        assert!(validate_raster(&[0.5; 9], 3, 3).is_err(), "非整数标签应被拒绝");
        assert!(validate_raster(&[1.0; 8], 3, 3).is_err(), "长度不符应被拒绝");
        assert!(validate_raster(&[5.0; 9], 3, 3).is_err(), "越界标签应被拒绝");
        // 全是墙、没有一个流体格点 —— 计算域为空，必须报错而不是静默跑
        assert!(
            validate_raster(&[2.0; 9], 3, 3).is_err(),
            "只有边界标签、没有流体，应被拒绝"
        );
    }

    #[test]
    fn accepts_zone_labels() {
        let z = [0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0];
        validate_raster(&z, 3, 3).expect("合法分区标签应被接受");
        assert_eq!(zone_counts(&z), [1, 5, 1, 1, 1]);
        assert_eq!(
            zone_indicator(&z, Zone::Wall),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn zone_labels_survive_nearest_resample() {
        // 最近邻对标签场是唯一正确的重采样：任何插值都会造出 1.5 这种不存在的分区
        let src = [1.0, 2.0, 3.0, 4.0];
        let got = resample_raster(&src, 2, 2, 4, 4).unwrap();
        for v in &got {
            Zone::from_label(*v).expect("重采样后仍须是合法标签");
        }
        assert_eq!(got[0], 1.0);
        assert_eq!(got[15], 4.0);
    }
}
