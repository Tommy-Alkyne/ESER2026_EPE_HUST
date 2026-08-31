//! P6 推理侧：`surrogate_predict.py` 的等价物。
//!
//! 「一份权重表 + 一组超参 -> 对某个 HDF5 做前向、反归一化、按计算域裁剪、写盘」。
//!
//! # 这一层的三个静默失效点
//!
//! 1. **超参白名单漏项**。Python 版是
//!    `hp.update({k: v for k, v in ckpt_args.items() if k in hp and v is not None})`
//!    ——**不在白名单里的键被静默丢弃**。`t_geometry` 曾经就漏在外面，于是车站模型
//!    的输出被默认 T 型几何裁掉（覆盖率 45% vs 25%），一句警告都没有。
//!    [`HP_KEYS`] 就是那份白名单，[`merge_ckpt_args`] 会把丢掉的键**报出来**
//!    （Python 版只是静默丢弃，这一点是本移植版故意加严的）。
//! 2. **忘了反归一化**。网络输出在 [0,1]，直接写盘的话下游拿不回压力/速度量纲，
//!    而数值本身「看起来很正常」。
//! 3. **掩码来源与训练时不一致**。推理换一份几何不会报错，只会把结果裁成另一个
//!    计算域。所以几何缺省一律从 checkpoint 读，命令行只作为显式覆盖。
//!
//! # 与 Python 版的差异（有意为之）
//!
//! * 没有 CLI（本 crate 是库，`Cargo.toml` 不在 P6 的改动范围内）。命令行参数对应
//!   [`PredictOptions`] 与 [`Hyperparams`] 的字段。
//! * 旧命名 `LastTransUp.convT2.weight` 的通道反推没有移植：Rust 侧的
//!   [`crate::store::LoadState`] 只认新命名，旧 checkpoint 连权重都灌不进来，
//!   单独支持通道反推没有意义。

use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor, TensorData};
use rust_hdf5::H5File;

use crate::data::{denormalize, CfdDataset, OutputStats};
use crate::geometry::{coverage, default_t_geometry, rect_mask, resample_raster, Segment};
use crate::model::{DenseEd, DenseEdConfig};
use crate::model3d::{DenseEd3d, DenseEd3dConfig};
use crate::physics::ChannelOrder;
use crate::store::StateDict;

// ================================================================ 极小 JSON

/// 一个够用就好的 JSON 解析器。
///
/// 放在推理模块里，是因为 P6 只允许新增两个源文件，而 `--t-geometry` 的矩形段与
/// `experiments/*/args.txt` 都是 JSON——本 crate 没有 serde 依赖，也不该为这点
/// 需求新加一个（`Cargo.toml` 不在改动范围内）。
///
/// 特意支持 Python `json.dump` 会写出的**非标准字面量** `NaN` / `Infinity` /
/// `-Infinity`：`train.py` 用 `json.dump(vars(args))` 落盘，一个 nan 就会让标准
/// 解析器整份读失败，而诊断报告那时只会说「读取失败」，下游全变成 `--`。
pub mod json {
    /// JSON 值。对象用 `Vec` 而不是 `HashMap`：保序、无需额外依赖，键也就几十个。
    ///
    /// **整数与浮点分开**：Python 的 `json` 也是这么还原的，而诊断报告要按
    /// `'{}'.format(v)` 逐字对上原版——`200` 与 `200.0` 打出来不是一回事。
    #[derive(Debug, Clone, PartialEq)]
    pub enum Json {
        Null,
        Bool(bool),
        Int(i64),
        Num(f64),
        Str(String),
        Array(Vec<Json>),
        Object(Vec<(String, Json)>),
    }

    impl Json {
        pub fn parse(text: &str) -> Result<Json, String> {
            let bytes: Vec<char> = text.chars().collect();
            let mut p = Parser { s: &bytes, i: 0 };
            p.skip_ws();
            let v = p.value()?;
            p.skip_ws();
            if p.i != p.s.len() {
                return Err(format!("第 {} 个字符之后还有多余内容", p.i));
            }
            Ok(v)
        }

        /// 取对象成员。不是对象或没有这个键时返回 `None`。
        pub fn get(&self, key: &str) -> Option<&Json> {
            match self {
                Json::Object(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }

        /// 取对象成员，且值不是 `null`。对应 Python 里 `cfg.get(k)` 之后的真值判断。
        pub fn get_some(&self, key: &str) -> Option<&Json> {
            self.get(key).filter(|v| !matches!(v, Json::Null))
        }

        pub fn as_f64(&self) -> Option<f64> {
            match self {
                Json::Num(v) => Some(*v),
                Json::Int(v) => Some(*v as f64),
                Json::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
                _ => None,
            }
        }

        pub fn as_str(&self) -> Option<&str> {
            match self {
                Json::Str(s) => Some(s),
                _ => None,
            }
        }

        pub fn as_bool(&self) -> Option<bool> {
            match self {
                Json::Bool(b) => Some(*b),
                _ => None,
            }
        }

        pub fn as_array(&self) -> Option<&[Json]> {
            match self {
                Json::Array(v) => Some(v),
                _ => None,
            }
        }

        pub fn is_null(&self) -> bool {
            matches!(self, Json::Null)
        }

        /// Python `bool(x)` 的语义：null / false / 0 / "" / 空容器都是假。
        /// 诊断脚本里大量出现 `if cfg.get('no_mask'):` 这种真值判断，语义必须一致。
        pub fn truthy(&self) -> bool {
            match self {
                Json::Null => false,
                Json::Bool(b) => *b,
                Json::Int(v) => *v != 0,
                Json::Num(v) => *v != 0.0,
                Json::Str(s) => !s.is_empty(),
                Json::Array(v) => !v.is_empty(),
                Json::Object(kv) => !kv.is_empty(),
            }
        }

        /// 按 Python `'{}'.format(v)` 的样子渲染，供诊断报告逐字对上原版输出。
        pub fn py_repr(&self) -> String {
            match self {
                Json::Null => "None".into(),
                Json::Bool(b) => if *b { "True" } else { "False" }.into(),
                Json::Int(v) => v.to_string(),
                Json::Num(v) => fmt_py_float(*v),
                Json::Str(s) => s.clone(),
                Json::Array(v) => format!(
                    "[{}]",
                    v.iter()
                        .map(|e| e.py_literal())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                Json::Object(kv) => format!(
                    "{{{}}}",
                    kv.iter()
                        .map(|(k, v)| format!("{}: {}", py_str_repr(k), v.py_literal()))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }
        }

        /// 容器**内部**元素的渲染：Python 打印列表时字符串带引号，标量不带。
        fn py_literal(&self) -> String {
            match self {
                Json::Str(s) => py_str_repr(s),
                other => other.py_repr(),
            }
        }
    }

    /// Python `repr(str)`：默认单引号，串里有单引号而没有双引号时才改用双引号。
    /// Rust 的 `{:?}` 一律用双引号，直接拿来会和 Python 的输出差两个字符。
    fn py_str_repr(s: &str) -> String {
        let quote = if s.contains('\'') && !s.contains('"') {
            '"'
        } else {
            '\''
        };
        let mut out = String::with_capacity(s.len() + 2);
        out.push(quote);
        for c in s.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if c == quote => {
                    out.push('\\');
                    out.push(c);
                }
                c => out.push(c),
            }
        }
        out.push(quote);
        out
    }

    /// Python `repr(float)` 的等价实现。
    ///
    /// 两个实现的默认浮点打印**不是一回事**，而诊断报告要逐行对上：
    ///
    /// | 值 | Python | Rust `{}` |
    /// |---|---|---|
    /// | `1e-5` | `1e-05` | `0.00001` |
    /// | `1.0` | `1.0` | `1` |
    /// | `1e16` | `1e+16` | `10000000000000000` |
    ///
    /// 规则来自 CPython 的 `format_float_short`：最短往返表示的数字串 `digits`
    /// 与小数点位置 `decpt`（值 = `0.digits × 10^decpt`），`decpt <= -4 || decpt > 16`
    /// 时走指数形式，指数至少两位。最短往返的数字串直接借 Rust 的 `{:e}`。
    pub fn fmt_py_float(v: f64) -> String {
        if v.is_nan() {
            return "nan".into();
        }
        if v.is_infinite() {
            return if v > 0.0 { "inf" } else { "-inf" }.into();
        }
        let sci = format!("{:e}", v); // 形如 "-1.5e-5" / "0e0"
        let (mantissa, exp) = sci.split_once('e').expect("{:e} 必然带 e");
        let exp: i32 = exp.parse().expect("{:e} 的指数必然是整数");
        let (sign, mantissa) = match mantissa.strip_prefix('-') {
            Some(rest) => ("-", rest),
            None => ("", mantissa),
        };
        let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
        let decpt = exp + 1;

        if decpt <= -4 || decpt > 16 {
            let head = &digits[..1];
            let tail = &digits[1..];
            let frac = if tail.is_empty() {
                String::new()
            } else {
                format!(".{}", tail)
            };
            let e = decpt - 1;
            return format!(
                "{}{}{}e{}{:02}",
                sign,
                head,
                frac,
                if e < 0 { '-' } else { '+' },
                e.abs()
            );
        }
        let n = digits.len() as i32;
        let body = if decpt <= 0 {
            format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
        } else if decpt >= n {
            // 没有小数部分时 Python 会补 `.0`（Py_DTSF_ADD_DOT_0）
            format!("{}{}.0", digits, "0".repeat((decpt - n) as usize))
        } else {
            format!(
                "{}.{}",
                &digits[..decpt as usize],
                &digits[decpt as usize..]
            )
        };
        format!("{}{}", sign, body)
    }

    struct Parser<'a> {
        s: &'a [char],
        i: usize,
    }

    impl Parser<'_> {
        fn skip_ws(&mut self) {
            while self.i < self.s.len() && self.s[self.i].is_whitespace() {
                self.i += 1;
            }
        }

        fn peek(&self) -> Option<char> {
            self.s.get(self.i).copied()
        }

        fn eat(&mut self, lit: &str) -> bool {
            let n = lit.chars().count();
            if self.i + n <= self.s.len()
                && self.s[self.i..self.i + n].iter().copied().eq(lit.chars())
            {
                self.i += n;
                true
            } else {
                false
            }
        }

        fn value(&mut self) -> Result<Json, String> {
            self.skip_ws();
            match self.peek() {
                None => Err("内容为空".into()),
                Some('{') => self.object(),
                Some('[') => self.array(),
                Some('"') => Ok(Json::Str(self.string()?)),
                Some('t') if self.eat("true") => Ok(Json::Bool(true)),
                Some('f') if self.eat("false") => Ok(Json::Bool(false)),
                Some('n') if self.eat("null") => Ok(Json::Null),
                // Python json.dump 的非标准字面量
                Some('N') if self.eat("NaN") => Ok(Json::Num(f64::NAN)),
                Some('I') if self.eat("Infinity") => Ok(Json::Num(f64::INFINITY)),
                Some('-') if self.eat("-Infinity") => Ok(Json::Num(f64::NEG_INFINITY)),
                Some(_) => self.number(),
            }
        }

        fn object(&mut self) -> Result<Json, String> {
            self.i += 1; // '{'
            let mut out = Vec::new();
            self.skip_ws();
            if self.peek() == Some('}') {
                self.i += 1;
                return Ok(Json::Object(out));
            }
            loop {
                self.skip_ws();
                let k = self.string()?;
                self.skip_ws();
                if self.peek() != Some(':') {
                    return Err(format!("第 {} 个字符处缺少 ':'", self.i));
                }
                self.i += 1;
                let v = self.value()?;
                out.push((k, v));
                self.skip_ws();
                match self.peek() {
                    Some(',') => self.i += 1,
                    Some('}') => {
                        self.i += 1;
                        return Ok(Json::Object(out));
                    }
                    _ => return Err(format!("第 {} 个字符处对象未正常闭合", self.i)),
                }
            }
        }

        fn array(&mut self) -> Result<Json, String> {
            self.i += 1; // '['
            let mut out = Vec::new();
            self.skip_ws();
            if self.peek() == Some(']') {
                self.i += 1;
                return Ok(Json::Array(out));
            }
            loop {
                out.push(self.value()?);
                self.skip_ws();
                match self.peek() {
                    Some(',') => self.i += 1,
                    Some(']') => {
                        self.i += 1;
                        return Ok(Json::Array(out));
                    }
                    _ => return Err(format!("第 {} 个字符处数组未正常闭合", self.i)),
                }
            }
        }

        fn string(&mut self) -> Result<String, String> {
            if self.peek() != Some('"') {
                return Err(format!("第 {} 个字符处期望字符串", self.i));
            }
            self.i += 1;
            let mut out = String::new();
            loop {
                let c = self
                    .peek()
                    .ok_or_else(|| "字符串没有闭合的引号".to_string())?;
                self.i += 1;
                match c {
                    '"' => return Ok(out),
                    '\\' => {
                        let e = self.peek().ok_or_else(|| "转义符后没有内容".to_string())?;
                        self.i += 1;
                        match e {
                            '"' => out.push('"'),
                            '\\' => out.push('\\'),
                            '/' => out.push('/'),
                            'b' => out.push('\u{8}'),
                            'f' => out.push('\u{c}'),
                            'n' => out.push('\n'),
                            'r' => out.push('\r'),
                            't' => out.push('\t'),
                            'u' => out.push(self.unicode_escape()?),
                            other => return Err(format!("未知转义 \\{}", other)),
                        }
                    }
                    other => out.push(other),
                }
            }
        }

        fn hex4(&mut self) -> Result<u32, String> {
            if self.i + 4 > self.s.len() {
                return Err("\\u 后不足 4 位十六进制".into());
            }
            let text: String = self.s[self.i..self.i + 4].iter().collect();
            self.i += 4;
            u32::from_str_radix(&text, 16).map_err(|e| format!("\\u{} 不是十六进制: {}", text, e))
        }

        fn unicode_escape(&mut self) -> Result<char, String> {
            let hi = self.hex4()?;
            // 代理对：高位 D800..DBFF 后面必须跟一个 \uDC00..DFFF
            if (0xD800..0xDC00).contains(&hi) {
                if !self.eat("\\u") {
                    return Err("高代理项后缺少低代理项".into());
                }
                let lo = self.hex4()?;
                if !(0xDC00..0xE000).contains(&lo) {
                    return Err(format!("非法的低代理项 {:04X}", lo));
                }
                let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                return char::from_u32(cp).ok_or_else(|| format!("非法码位 {:X}", cp));
            }
            char::from_u32(hi).ok_or_else(|| format!("非法码位 {:X}", hi))
        }

        fn number(&mut self) -> Result<Json, String> {
            let start = self.i;
            if self.peek() == Some('-') || self.peek() == Some('+') {
                self.i += 1;
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
                    self.i += 1;
                } else {
                    break;
                }
            }
            let text: String = self.s[start..self.i].iter().collect();
            // 没有小数点也没有指数的字面量在 Python 侧还原成 int，打印时不带 `.0`
            if !text.contains(['.', 'e', 'E']) {
                if let Ok(i) = text.parse::<i64>() {
                    return Ok(Json::Int(i));
                }
            }
            text.parse::<f64>()
                .map(Json::Num)
                .map_err(|e| format!("{:?} 不是合法数字: {}", text, e))
        }
    }
}

use json::Json;

// ================================================================ 几何 spec

/// `--t-geometry` / `--domain-mask` 的取值，对应 `geometry.parse_geometry` 的返回。
///
/// **保存的是「怎么造掩码」而不是掩码本身**：分辨率要等到拿到预测场才知道。
#[derive(Debug, Clone, PartialEq)]
pub enum GeometrySpec {
    /// 缺省 T 型管
    Default,
    /// 矩形段拼接（归一化比例）
    Segments(Vec<Segment>),
    /// `.npy` 栅格（地铁站这类拼不出来的平面），会做最近邻重采样
    Raster { mask: Vec<f32>, h: usize, w: usize },
}

impl GeometrySpec {
    /// 解析 `--t-geometry` 的字符串取值。对应 `geometry.parse_geometry`。
    ///
    /// * `None` / `""` -> [`GeometrySpec::Default`]
    /// * `*.npy` -> 栅格文件
    /// * 以 `{` 开头 -> 内联 JSON
    /// * 其余 -> 指向 JSON 文件的路径
    pub fn parse(spec: Option<&str>) -> Result<Self, String> {
        let text = match spec {
            None => return Ok(Self::Default),
            Some(s) if s.trim().is_empty() => return Ok(Self::Default),
            Some(s) => s.trim().to_string(),
        };
        if text.to_ascii_lowercase().ends_with(".npy") {
            let (mask, h, w) = read_npy_2d(Path::new(&text))?;
            return Ok(Self::Raster { mask, h, w });
        }
        let body = if text.starts_with('{') {
            text
        } else {
            std::fs::read_to_string(&text)
                .map_err(|e| format!("读不了几何定义文件 {:?}: {}", text, e))?
        };
        Self::from_json(&Json::parse(&body).map_err(|e| format!("几何定义不是合法 JSON: {}", e))?)
    }

    /// 从已解析的 JSON 对象取矩形段：`{"vertical": [y0, y1, x0, x1], ...}`。
    pub fn from_json(v: &Json) -> Result<Self, String> {
        let Json::Object(kv) = v else {
            return Err("几何定义必须是 {段名: [y0, y1, x0, x1]} 形式的对象".into());
        };
        let mut segments = Vec::with_capacity(kv.len());
        for (name, value) in kv {
            let arr = value
                .as_array()
                .ok_or_else(|| format!("几何段 {} 必须是数组", name))?;
            if arr.len() != 4 {
                return Err(format!(
                    "几何段 {} 必须是 (y0, y1, x0, x1) 四元组，实得 {} 项",
                    name,
                    arr.len()
                ));
            }
            let mut b = [0.0f64; 4];
            for (i, e) in arr.iter().enumerate() {
                b[i] = e
                    .as_f64()
                    .ok_or_else(|| format!("几何段 {} 的第 {} 项不是数字", name, i))?;
            }
            segments.push(Segment::new(b[0], b[1], b[2], b[3]));
        }
        Ok(Self::Segments(segments))
    }

    /// 生成 (h, w) 的 0/1 掩码，行优先展平。对应 `geometry.t_mask_numpy`。
    pub fn mask(&self, h: usize, w: usize) -> Result<Vec<f32>, String> {
        match self {
            Self::Default => rect_mask(h, w, &default_t_geometry()),
            Self::Segments(s) => rect_mask(h, w, s),
            Self::Raster { mask, h: sh, w: sw } => resample_raster(mask, *sh, *sw, h, w),
        }
    }
}

/// 读一个 2D 的 `.npy` 栅格并统一成 f32。
///
/// 自己解析而不是引依赖：`ndarray-npy` 只在 dev-dependencies 里（测试用），而
/// `Cargo.toml` 不在 P6 的改动范围内。格式本身很简单——魔数、头长度、一行 Python
/// 字典字面量，然后是裸数据。
fn read_npy_2d(path: &Path) -> Result<(Vec<f32>, usize, usize), String> {
    let raw = std::fs::read(path).map_err(|e| format!("读不了栅格掩码 {:?}: {}", path, e))?;
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        return Err(format!("{:?} 不是 .npy 文件（魔数不符）", path));
    }
    let major = raw[6];
    let (header_len, body_start) = match major {
        1 => (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10usize),
        2 | 3 => (
            u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize,
            12usize,
        ),
        other => return Err(format!("不支持的 .npy 版本 {}", other)),
    };
    let head_end = body_start + header_len;
    if head_end > raw.len() {
        return Err(format!("{:?} 的头部长度越界", path));
    }
    let header = String::from_utf8_lossy(&raw[body_start..head_end]).to_string();

    if header.contains("'fortran_order': True") {
        return Err(format!(
            "{:?} 是列优先存储（fortran_order=True），请用 np.ascontiguousarray 重存",
            path
        ));
    }
    let descr = between(&header, "'descr': '", "'")
        .ok_or_else(|| format!("{:?} 的头部里没有 descr", path))?;
    let shape_text = between(&header, "'shape': (", ")")
        .ok_or_else(|| format!("{:?} 的头部里没有 shape", path))?;
    let dims: Vec<usize> = shape_text
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| format!("shape 解析失败: {}", e))
        })
        .collect::<Result<_, _>>()?;
    if dims.len() != 2 {
        return Err(format!(
            "栅格掩码必须是 2D (H, W)，{:?} 的形状是 {:?}",
            path, dims
        ));
    }
    let (h, w) = (dims[0], dims[1]);
    let body = &raw[head_end..];
    let n = h * w;

    // 字节序：'<' 小端、'|' 单字节无所谓；大端读出来是垃圾数值而不是报错，必须挡住
    let item = |size: usize| -> Result<(), String> {
        if body.len() < n * size {
            Err(format!(
                "{:?} 的数据区只有 {} 字节，形状 {:?} 需要 {}",
                path,
                body.len(),
                dims,
                n * size
            ))
        } else {
            Ok(())
        }
    };
    let data: Vec<f32> = match descr.as_str() {
        "<f4" | "=f4" => {
            item(4)?;
            body[..n * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        }
        "<f8" | "=f8" => {
            item(8)?;
            body[..n * 8]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c) as f32)
                .collect()
        }
        "|b1" | "|u1" => {
            item(1)?;
            body[..n].iter().map(|&b| b as f32).collect()
        }
        "<i4" | "=i4" => {
            item(4)?;
            body[..n * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c) as f32)
                .collect()
        }
        "<i8" | "=i8" => {
            item(8)?;
            body[..n * 8]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c) as f32)
                .collect()
        }
        other => {
            return Err(format!(
                "{:?} 的 dtype {:?} 不受支持（只认小端 f4/f8/i4/i8 与 b1/u1）",
                path, other
            ))
        }
    };
    Ok((data, h, w))
}

fn between(text: &str, start: &str, end: &str) -> Option<String> {
    let i = text.find(start)? + start.len();
    let j = text[i..].find(end)? + i;
    Some(text[i..j].to_string())
}

// ================================================================ 超参白名单

/// checkpoint 里能被读回来的超参**白名单**。对应 `surrogate_predict.DEFAULT_HP` 的键集。
///
/// 不在这里的键会被丢弃。`t_geometry` / `no_mask` 曾经就漏在外面，导致推理拿默认
/// T 型几何去裁剪一个车站模型的输出——覆盖率 45% vs 25%，而且一句警告都没有。
/// `tests/test_transport.py::test_checkpoint_hp_whitelist` 在 Python 侧钉住了这一条，
/// 本模块的单元测试在 Rust 侧钉同一条。
pub const HP_KEYS: &[&str] = &[
    "blocks",
    "growth_rate",
    "init_features",
    "drop_rate",
    "bn_size",
    "bottleneck",
    "num_heads",
    "attn_max_tokens",
    "no_positional_encoding",
    "channel_order",
    "task",
    "diffusivity",
    "t_geometry",
    "no_mask",
];

/// 一组推理超参，默认值与 `DEFAULT_HP` 逐项一致。
///
/// `channel_order` / `task` / `t_geometry` 存的是**原始字符串**，与 Python 一致：
/// 它们要原样写回输出文件的属性里，几何还要留到拿到分辨率时才解析。
/// 合法性在 [`merge_ckpt_args`] 时就校验，不拖到用的时候。
#[derive(Debug, Clone, PartialEq)]
pub struct Hyperparams {
    pub blocks: Vec<usize>,
    pub growth_rate: usize,
    pub init_features: usize,
    pub drop_rate: f64,
    pub bn_size: usize,
    pub bottleneck: bool,
    pub num_heads: usize,
    pub attn_max_tokens: usize,
    pub no_positional_encoding: bool,
    pub channel_order: String,
    pub task: String,
    pub diffusivity: f64,
    pub t_geometry: Option<String>,
    pub no_mask: bool,
}

impl Default for Hyperparams {
    fn default() -> Self {
        Self {
            blocks: vec![3, 6, 3],
            growth_rate: 16,
            init_features: 48,
            drop_rate: 0.0,
            bn_size: 8,
            bottleneck: false,
            num_heads: 4,
            attn_max_tokens: 8192,
            no_positional_encoding: false,
            channel_order: "pvv".into(),
            task: "flow".into(),
            diffusivity: 1e-5,
            t_geometry: None,
            no_mask: false,
        }
    }
}

impl Hyperparams {
    /// 2D 网络配置，对应 Python 里那一长串 `DenseED(...)` 关键字参数。
    pub fn model_config_2d(
        &self,
        in_channels: usize,
        out_channels: usize,
        output_size: [usize; 2],
    ) -> DenseEdConfig {
        let mut cfg = DenseEdConfig::new(in_channels, out_channels, self.blocks.clone());
        cfg.growth_rate = self.growth_rate;
        cfg.init_features = self.init_features;
        cfg.drop_rate = self.drop_rate;
        cfg.bn_size = self.bn_size;
        cfg.bottleneck = self.bottleneck;
        cfg.out_activation = None;
        cfg.output_size = Some(output_size);
        cfg.num_heads = self.num_heads;
        // Python 传的是 `use_positional_encoding=not hp['no_positional_encoding']`，
        // 少一个取反就等于悄悄关掉位置编码
        cfg.use_positional_encoding = !self.no_positional_encoding;
        cfg.attn_max_tokens = self.attn_max_tokens;
        cfg
    }

    /// 3D 网络配置，与 2D 同构，只是 `output_size` 多一根深度轴。
    pub fn model_config_3d(
        &self,
        in_channels: usize,
        out_channels: usize,
        output_size: [usize; 3],
    ) -> DenseEd3dConfig {
        let mut cfg = DenseEd3dConfig::new(in_channels, out_channels, self.blocks.clone());
        cfg.growth_rate = self.growth_rate;
        cfg.init_features = self.init_features;
        cfg.drop_rate = self.drop_rate;
        cfg.bn_size = self.bn_size;
        cfg.bottleneck = self.bottleneck;
        cfg.out_activation = None;
        cfg.output_size = Some(output_size);
        cfg.num_heads = self.num_heads;
        cfg.use_positional_encoding = !self.no_positional_encoding;
        cfg.attn_max_tokens = self.attn_max_tokens;
        cfg
    }

    /// 序列化成 `f.attrs['hyperparams']` 里那一份 JSON。
    pub fn to_json_string(&self) -> String {
        let blocks = self
            .blocks
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let geom = match &self.t_geometry {
            None => "null".to_string(),
            Some(s) => escape_json(s),
        };
        format!(
            concat!(
                "{{\"blocks\": [{}], \"growth_rate\": {}, \"init_features\": {}, ",
                "\"drop_rate\": {}, \"bn_size\": {}, \"bottleneck\": {}, \"num_heads\": {}, ",
                "\"attn_max_tokens\": {}, \"no_positional_encoding\": {}, ",
                "\"channel_order\": {}, \"task\": {}, \"diffusivity\": {}, ",
                "\"t_geometry\": {}, \"no_mask\": {}}}"
            ),
            blocks,
            self.growth_rate,
            self.init_features,
            self.drop_rate,
            self.bn_size,
            self.bottleneck,
            self.num_heads,
            self.attn_max_tokens,
            self.no_positional_encoding,
            escape_json(&self.channel_order),
            escape_json(&self.task),
            self.diffusivity,
            geom,
            self.no_mask,
        )
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `_parse_blocks`：接受 `[3, 6, 3]` 数组、`"[3,6,3]"` 或 `"3,6,3"` 字符串。
pub fn parse_blocks(v: &Json) -> Result<Vec<usize>, String> {
    match v {
        Json::Array(items) => items
            .iter()
            .map(|e| {
                e.as_f64()
                    .filter(|x| *x >= 0.0)
                    .map(|x| x as usize)
                    .ok_or_else(|| format!("blocks 里出现了非法项 {:?}", e))
            })
            .collect(),
        Json::Str(s) => s
            .trim()
            .trim_matches(|c| c == '[' || c == ']' || c == '(' || c == ')')
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(|t| {
                t.parse::<usize>()
                    .map_err(|e| format!("blocks 项 {:?} 解析失败: {}", t, e))
            })
            .collect(),
        other => Err(format!("blocks 必须是数组或字符串，实得 {:?}", other)),
    }
}

/// 把 checkpoint 里的 `args` 合并进超参，返回**被丢弃的键**。
///
/// 对应 `hp.update({k: v for k, v in ckpt_args.items() if k in hp and v is not None})`：
/// 不在 [`HP_KEYS`] 里的键不会生效。Python 版对此完全静默，本移植版把它们返回给
/// 调用方——那个 45% vs 25% 的事故就是「静默丢弃」造成的。
pub fn merge_ckpt_args(hp: &mut Hyperparams, args: &Json) -> Result<Vec<String>, String> {
    let Json::Object(kv) = args else {
        return Err("checkpoint 的 args 必须是 JSON 对象".into());
    };
    let mut dropped = Vec::new();
    for (key, value) in kv {
        if !HP_KEYS.contains(&key.as_str()) {
            dropped.push(key.clone());
            continue;
        }
        if value.is_null() {
            // Python 侧 `v is not None` 的分支：没信息就保持默认值
            continue;
        }
        let num = |v: &Json| -> Result<f64, String> {
            v.as_f64()
                .ok_or_else(|| format!("超参 {} 期望数字，实得 {:?}", key, v))
        };
        let boolean = |v: &Json| -> Result<bool, String> {
            v.as_bool()
                .or_else(|| v.as_f64().map(|x| x != 0.0))
                .ok_or_else(|| format!("超参 {} 期望布尔值，实得 {:?}", key, v))
        };
        let text = |v: &Json| -> Result<String, String> {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("超参 {} 期望字符串，实得 {:?}", key, v))
        };
        match key.as_str() {
            "blocks" => hp.blocks = parse_blocks(value)?,
            "growth_rate" => hp.growth_rate = num(value)? as usize,
            "init_features" => hp.init_features = num(value)? as usize,
            "drop_rate" => hp.drop_rate = num(value)?,
            "bn_size" => hp.bn_size = num(value)? as usize,
            "bottleneck" => hp.bottleneck = boolean(value)?,
            "num_heads" => hp.num_heads = num(value)? as usize,
            "attn_max_tokens" => hp.attn_max_tokens = num(value)? as usize,
            "no_positional_encoding" => hp.no_positional_encoding = boolean(value)?,
            "channel_order" => {
                let s = text(value)?;
                // 立刻校验：写错的通道顺序会原样写进输出文件的属性，下游据此解读通道
                ChannelOrder::parse(&s)?;
                hp.channel_order = s;
            }
            "task" => {
                let s = text(value)?;
                if s != "flow" && s != "dispersion" {
                    return Err(format!("未知的 task={:?}，可选 flow / dispersion", s));
                }
                hp.task = s;
            }
            "diffusivity" => hp.diffusivity = num(value)?,
            "t_geometry" => hp.t_geometry = Some(text(value)?),
            "no_mask" => hp.no_mask = boolean(value)?,
            other => return Err(format!("白名单里有 {:?} 但没有对应的赋值分支", other)),
        }
    }
    Ok(dropped)
}

/// 从「扁平字符串表」形式的 checkpoint 元信息合并超参。
///
/// P5 的 `train::CheckpointMeta::hyper` 就是 `Vec<(String, String)>`，这里刻意
/// **不 import 那个模块**——推理侧只依赖数据形状，不依赖训练侧的类型，两边各自
/// 演进不会互相拽着走。
///
/// 值按 JSON 字面量解析（`false` / `0.002` / `[1, 1, 1]`），解析不了就当字符串
/// （`pvv` / `3,6,3` / 路径）。字面量 `null` / `None` 视同没这个键。
pub fn merge_ckpt_hp_strings(
    hp: &mut Hyperparams,
    entries: &[(String, String)],
) -> Result<Vec<String>, String> {
    let mut object = Vec::with_capacity(entries.len());
    for (k, v) in entries {
        let value = match v.trim() {
            "null" | "None" | "" => Json::Null,
            text => Json::parse(text).unwrap_or_else(|_| Json::Str(v.clone())),
        };
        object.push((k.clone(), value));
    }
    merge_ckpt_args(hp, &Json::Object(object))
}

// ================================================================ 元信息推断

/// 从权重形状反推输入/输出通道数。对应 `infer_channels`。
///
/// checkpoint 的 `args` 里有 `nic` / `noc` 时优先用，缺了才查权重。
/// 查权重时会把对应的键标成「已用」——那两个键随后本来就会被 `load_state` 取走，
/// 不影响 [`StateDict::ensure_fully_used`]。
pub fn infer_channels<B: Backend>(
    sd: &StateDict,
    ndim: usize,
    ckpt_nic: Option<usize>,
    ckpt_noc: Option<usize>,
) -> Result<(usize, usize), String> {
    let (in_key, out_key) = ("in_conv.weight", "last_trans_up.conv2.weight");
    let dims: Option<Vec<usize>> = match ndim {
        2 => {
            let a = sd.contains(in_key).then(|| sd.tensor::<B, 4>(in_key));
            let b = sd.contains(out_key).then(|| sd.tensor::<B, 4>(out_key));
            match (a, b) {
                (Some(a), Some(b)) => Some(vec![a?.dims()[1], b?.dims()[0]]),
                _ => None,
            }
        }
        3 => {
            let a = sd.contains(in_key).then(|| sd.tensor::<B, 5>(in_key));
            let b = sd.contains(out_key).then(|| sd.tensor::<B, 5>(out_key));
            match (a, b) {
                (Some(a), Some(b)) => Some(vec![a?.dims()[1], b?.dims()[0]]),
                _ => None,
            }
        }
        other => return Err(format!("ndim 只能是 2 或 3，实得 {}", other)),
    };

    let nic = ckpt_nic.or_else(|| dims.as_ref().map(|d| d[0]));
    let noc = ckpt_noc.or_else(|| dims.as_ref().map(|d| d[1]));
    match (nic, noc) {
        (Some(a), Some(b)) => Ok((a, b)),
        _ => Err(format!(
            "无法从权重表推断通道数（缺 {:?} 或 {:?}），请检查 checkpoint 是否完整",
            in_key, out_key
        )),
    }
}

/// 判断数据是 2D 还是 3D。对应 `detect_ndim`：命令行 > checkpoint > 数据文件。
pub fn detect_ndim(
    data_file: &Path,
    ckpt_ndim: Option<usize>,
    override_ndim: Option<usize>,
) -> Result<usize, String> {
    if let Some(v) = override_ndim {
        return Ok(v);
    }
    // Python 是 `if ckpt_args.get('ndim'):` —— 0 与 None 一样被当成「没这个信息」
    if let Some(v) = ckpt_ndim.filter(|v| *v != 0) {
        return Ok(v);
    }
    let file = H5File::open(data_file).map_err(|e| format!("打不开 {:?}: {}", data_file, e))?;
    let ds = file
        .dataset("output")
        .map_err(|e| format!("{:?} 里没有 output 数据集: {}", data_file, e))?;
    Ok(if ds.shape().len() == 5 { 3 } else { 2 })
}

// ================================================================ 掩码来源

/// 掩码从哪儿来。Python 版把这个信息只打进日志，这里做成返回值，好让调用方
/// （和测试）能断言。
#[derive(Debug, Clone, PartialEq)]
pub enum MaskChoice {
    /// 推理侧显式 `--no-mask`
    Disabled,
    /// 训练时就是 `--no-mask`，跳过以保持一致
    SkippedTrainedWithoutMask,
    Applied {
        spec: GeometrySpec,
        source: MaskSource,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskSource {
    CommandLine,
    Checkpoint,
}

impl MaskSource {
    /// 与 Python 日志里的中文来源标签一致。
    pub fn label(self) -> &'static str {
        match self {
            Self::CommandLine => "命令行",
            Self::Checkpoint => "checkpoint",
        }
    }
}

/// 决定用哪一份几何。对应 `main()` 里 `if args.apply_mask:` 那一段。
///
/// **缺省一律沿用 checkpoint 里训练时的那份**，命令行只作为显式覆盖；命令行给了
/// 几何就压过 `no_mask`（用户明确要裁）。
pub fn resolve_mask(
    hp: &Hyperparams,
    cli_geometry: Option<&str>,
    apply_mask: bool,
) -> Result<MaskChoice, String> {
    if !apply_mask {
        return Ok(MaskChoice::Disabled);
    }
    if let Some(spec) = cli_geometry {
        return Ok(MaskChoice::Applied {
            spec: GeometrySpec::parse(Some(spec))?,
            source: MaskSource::CommandLine,
        });
    }
    if hp.no_mask {
        return Ok(MaskChoice::SkippedTrainedWithoutMask);
    }
    Ok(MaskChoice::Applied {
        spec: GeometrySpec::parse(hp.t_geometry.as_deref())?,
        source: MaskSource::Checkpoint,
    })
}

// ================================================================ 推理

/// 推理选项，对应 `surrogate_predict.py` 的命令行开关。
#[derive(Debug, Clone)]
pub struct PredictOptions {
    pub batch_size: usize,
    /// 只推理前 N 个样本
    pub limit: Option<usize>,
    /// 输出有量纲物理场（默认开）
    pub denormalize: bool,
    /// 把计算域外置零（默认开）
    pub apply_mask: bool,
    /// 显式覆盖几何；缺省沿用 checkpoint 里的那份
    pub t_geometry: Option<String>,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            batch_size: 64,
            limit: None,
            denormalize: true,
            apply_mask: true,
            t_geometry: None,
        }
    }
}

/// 一次推理的全部产物。
#[derive(Debug, Clone)]
pub struct Prediction {
    /// 行优先展平的预测场
    pub data: Vec<f32>,
    /// `(N, C, ...)`
    pub dims: Vec<usize>,
    pub y_min: Vec<f32>,
    pub y_max: Vec<f32>,
    pub denormalized: bool,
    pub mask: MaskChoice,
    /// 掩码覆盖率，没用掩码时是 `None`
    pub coverage: Option<f64>,
}

impl Prediction {
    pub fn n_samples(&self) -> usize {
        self.dims.first().copied().unwrap_or(0)
    }

    pub fn n_channels(&self) -> usize {
        self.dims.get(1).copied().unwrap_or(0)
    }
}

/// 2D 离线推理。对应 `main()` 里「建模 -> 逐 batch 前向 -> 反归一化 -> 掩码」那一段。
///
/// 模型要用**非 autodiff** 后端（如 `NdArray`）建：BatchNorm 的 train/eval 由后端
/// 决定，推理正需要 eval 那条路径。
pub fn predict_2d<B: Backend>(
    model: &DenseEd<B>,
    dataset: &CfdDataset,
    hp: &Hyperparams,
    opts: &PredictOptions,
    device: &B::Device,
) -> Result<Prediction, String> {
    let n_total = limit_of(dataset.len(), opts.limit);
    let mut chunks: Vec<Tensor<B, 4>> = Vec::new();
    let mut written = 0usize;
    for sel in crate::data::batch_indices(dataset.len(), opts.batch_size, None, false) {
        if written >= n_total {
            break;
        }
        let (x, _) = dataset.batch4::<B>(&sel, device)?;
        let y = model.forward(x);
        written += y.dims()[0];
        chunks.push(y);
    }
    if chunks.is_empty() {
        return Err("数据集为空，没有可推理的样本".into());
    }
    let y = Tensor::cat(chunks, 0).narrow(0, 0, n_total);
    finish(y, &dataset.stats, hp, opts)
}

/// 3D 离线推理。3D 加载器与 Python 版一样一次性把数据读进内存。
pub fn predict_3d<B: Backend>(
    model: &DenseEd3d<B>,
    x_all: Tensor<B, 5>,
    stats: &OutputStats,
    hp: &Hyperparams,
    opts: &PredictOptions,
) -> Result<Prediction, String> {
    let n = x_all.dims()[0];
    let n_total = limit_of(n, opts.limit);
    let mut chunks: Vec<Tensor<B, 5>> = Vec::new();
    let mut written = 0usize;
    for sel in crate::data::batch_indices(n, opts.batch_size, None, false) {
        if written >= n_total {
            break;
        }
        let x = x_all.clone().narrow(0, sel[0], sel.len());
        let y = model.forward(x);
        written += y.dims()[0];
        chunks.push(y);
    }
    if chunks.is_empty() {
        return Err("数据集为空，没有可推理的样本".into());
    }
    let y = Tensor::cat(chunks, 0).narrow(0, 0, n_total);
    finish(y, stats, hp, opts)
}

fn limit_of(n: usize, limit: Option<usize>) -> usize {
    match limit {
        Some(k) => n.min(k),
        None => n,
    }
}

/// 反归一化 + 掩码，2D / 3D 共用。**顺序不能反**：Python 先还原量纲再置零，
/// 反过来的话域外那圈会被 `y_min` 顶成非零值。
fn finish<B: Backend, const D: usize>(
    y: Tensor<B, D>,
    stats: &OutputStats,
    hp: &Hyperparams,
    opts: &PredictOptions,
) -> Result<Prediction, String> {
    let dims = y.dims().to_vec();
    if stats.y_min.len() != dims[1] {
        return Err(format!(
            "统计量有 {} 个通道，预测场有 {} 个——多半是另一个文件的统计量",
            stats.y_min.len(),
            dims[1]
        ));
    }
    let y = if opts.denormalize {
        denormalize(y, &stats.y_min, &stats.y_max)
    } else {
        y
    };

    let choice = resolve_mask(hp, opts.t_geometry.as_deref(), opts.apply_mask)?;
    let (y, cov) = match &choice {
        MaskChoice::Applied { spec, .. } => {
            let (h, w) = (dims[D - 2], dims[D - 1]);
            let mask = spec.mask(h, w)?;
            let cov = coverage(&mask);
            (apply_mask(y, &mask)?, Some(cov))
        }
        _ => (y, None),
    };

    Ok(Prediction {
        data: y
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| format!("取回预测数据失败: {:?}", e))?,
        dims,
        y_min: stats.y_min.clone(),
        y_max: stats.y_max.clone(),
        denormalized: opts.denormalize,
        mask: choice,
        coverage: cov,
    })
}

/// 把计算域外置零。对应 `preds[..., ~mask] = 0.0`。
///
/// 用 `mask_fill` 而不是「乘以 0/1 掩码」：乘法遇到 NaN/Inf 会留下 NaN，与 Python
/// 的赋值语义不同——预测场真出了 NaN 时，两个实现会给出不一样的文件。
fn apply_mask<B: Backend, const D: usize>(
    y: Tensor<B, D>,
    mask: &[f32],
) -> Result<Tensor<B, D>, String> {
    let dims = y.dims();
    let (h, w) = (dims[D - 2], dims[D - 1]);
    if mask.len() != h * w {
        return Err(format!(
            "掩码有 {} 个元素，与网格 {}x{} 不符",
            mask.len(),
            h,
            w
        ));
    }
    let mut shape = [1usize; D];
    shape[D - 2] = h;
    shape[D - 1] = w;
    let device = y.device();
    let m = Tensor::<B, 1>::from_data(TensorData::from(mask), &device).reshape(shape);
    let outside: Tensor<B, D, Bool> = m.equal_elem(0.0).expand(dims);
    Ok(y.mask_fill(outside, 0.0))
}

// ================================================================ 写盘

/// 输出文件里那一组属性，对应 `f.attrs[...]`。
#[derive(Debug, Clone)]
pub struct OutputMeta {
    pub source_data: String,
    pub ckpt_file: String,
    pub ndim: usize,
    pub nic: usize,
    pub noc: usize,
    pub channel_order: String,
    /// `hp` 的 JSON 序列化，见 [`Hyperparams::to_json_string`]
    pub hyperparams: String,
}

impl OutputMeta {
    /// 从超参与推断出的通道数装配一份属性表。
    pub fn new(
        hp: &Hyperparams,
        source_data: &str,
        ckpt_file: &str,
        ndim: usize,
        nic: usize,
        noc: usize,
    ) -> Self {
        Self {
            source_data: source_data.to_string(),
            ckpt_file: ckpt_file.to_string(),
            ndim,
            nic,
            noc,
            channel_order: hp.channel_order.clone(),
            hyperparams: hp.to_json_string(),
        }
    }
}

/// 写出预测文件。对应 `with h5py.File(out_file, 'w') as f: ...`。
///
/// **还原量纲所需的一切都存进去**（`y_min` / `y_max` 与是否已反归一化），下游不必再猜。
pub fn write_prediction_hdf5(
    path: &Path,
    pred: &Prediction,
    meta: &OutputMeta,
) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| format!("建不了目录 {:?}: {}", dir, e))?;
        }
    }
    let file = H5File::create(path).map_err(|e| format!("建不了 {:?}: {}", path, e))?;

    // gzip 需要分块存储，按「一个样本一块」切
    let mut chunk = pred.dims.clone();
    chunk[0] = 1;
    let ds = file
        .new_dataset::<f32>()
        .shape(&pred.dims)
        .chunk(&chunk)
        .deflate(4)
        .create("predictions")
        .map_err(|e| format!("建不了 predictions 数据集: {}", e))?;
    ds.write_raw(&pred.data)
        .map_err(|e| format!("写不进 predictions: {}", e))?;

    for (name, values) in [("y_min", &pred.y_min), ("y_max", &pred.y_max)] {
        let d = file
            .new_dataset::<f32>()
            .shape([values.len()])
            .create(name)
            .map_err(|e| format!("建不了 {} 数据集: {}", name, e))?;
        d.write_raw(values)
            .map_err(|e| format!("写不进 {}: {}", name, e))?;
    }

    let masked = matches!(pred.mask, MaskChoice::Applied { .. });
    let attr_str = |k: &str, v: &str| -> Result<(), String> {
        file.set_attr_string(k, v)
            .map_err(|e| format!("写不了属性 {}: {}", k, e))
    };
    let attr_int = |k: &str, v: i64| -> Result<(), String> {
        file.set_attr_numeric(k, &v)
            .map_err(|e| format!("写不了属性 {}: {}", k, e))
    };
    attr_str("source_data", &meta.source_data)?;
    attr_str("ckpt_file", &meta.ckpt_file)?;
    attr_int("ndim", meta.ndim as i64)?;
    attr_int("nic", meta.nic as i64)?;
    attr_int("noc", meta.noc as i64)?;
    attr_int("denormalized", i64::from(pred.denormalized))?;
    attr_int("masked", i64::from(masked))?;
    attr_str("channel_order", &meta.channel_order)?;
    attr_str("hyperparams", &meta.hyperparams)?;

    file.close()
        .map_err(|e| format!("关不掉 {:?}: {}", path, e))
}

/// 缺省输出路径：`<数据文件同目录>/<basename>.pred.hdf5`。
pub fn default_out_file(data_file: &Path) -> std::path::PathBuf {
    let stem = data_file
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "prediction".into());
    let dir = data_file.parent().unwrap_or_else(|| Path::new("."));
    dir.join(format!("{}.pred.hdf5", stem))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------ 白名单

    /// 与 `tests/test_transport.py::test_checkpoint_hp_whitelist` 同一条约束：
    /// 推理侧曾把 `t_geometry` 丢在白名单外，导致车站模型被 T 型几何裁剪。
    #[test]
    fn hp_whitelist_covers_the_five_critical_keys() {
        for key in [
            "t_geometry",
            "no_mask",
            "task",
            "diffusivity",
            "channel_order",
        ] {
            assert!(HP_KEYS.contains(&key), "checkpoint 超参白名单漏了 {}", key);
        }
    }

    #[test]
    fn every_whitelisted_key_has_an_assignment_branch() {
        // 白名单里有键却没有赋值分支 = 静默丢弃，与漏在白名单外一样有害
        for key in HP_KEYS {
            let args = Json::parse(&format!("{{{:?}: null}}", key)).unwrap();
            let mut hp = Hyperparams::default();
            let dropped = merge_ckpt_args(&mut hp, &args).unwrap();
            assert!(dropped.is_empty(), "{} 不该被丢弃", key);
        }
    }

    #[test]
    fn ckpt_geometry_survives_the_merge() {
        let args = Json::parse(
            r#"{"t_geometry": "station.npy", "no_mask": false, "channel_order": "c",
                "task": "dispersion", "diffusivity": 0.002, "blocks": "[1,1,1]",
                "epochs": 200, "lr": 0.001}"#,
        )
        .unwrap();
        let mut hp = Hyperparams::default();
        let dropped = merge_ckpt_args(&mut hp, &args).unwrap();

        assert_eq!(hp.t_geometry.as_deref(), Some("station.npy"));
        assert_eq!(hp.channel_order, "c");
        assert_eq!(hp.task, "dispersion");
        assert_eq!(hp.diffusivity, 0.002);
        assert_eq!(hp.blocks, vec![1, 1, 1]);
        // 不在白名单里的键要被**报出来**，而不是静默吞掉
        assert_eq!(dropped, vec!["epochs".to_string(), "lr".to_string()]);
    }

    /// P5 的 checkpoint 元信息是扁平字符串表，推理侧必须能从那儿把超参捡回来。
    #[test]
    fn flat_string_checkpoint_meta_merges() {
        let entries: Vec<(String, String)> = [
            ("channel_order", "c"),
            ("task", "dispersion"),
            ("diffusivity", "0.002"),
            ("no_mask", "false"),
            ("blocks", "3,6,3"),
            ("growth_rate", "24"),
            ("t_geometry", r"dataset\station_mask.npy"),
            ("no_positional_encoding", "true"),
            ("epochs", "200"),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

        let mut hp = Hyperparams::default();
        let dropped = merge_ckpt_hp_strings(&mut hp, &entries).unwrap();
        assert_eq!(hp.channel_order, "c");
        assert_eq!(hp.task, "dispersion");
        assert_eq!(hp.diffusivity, 0.002);
        assert!(!hp.no_mask);
        assert_eq!(hp.blocks, vec![3, 6, 3]);
        assert_eq!(hp.growth_rate, 24);
        assert_eq!(hp.t_geometry.as_deref(), Some(r"dataset\station_mask.npy"));
        assert!(hp.no_positional_encoding);
        assert_eq!(dropped, vec!["epochs".to_string()]);

        // 「没这个键」的几种写法都不该覆盖默认值
        for text in ["null", "None", ""] {
            let mut hp = Hyperparams::default();
            merge_ckpt_hp_strings(&mut hp, &[("t_geometry".into(), text.into())]).unwrap();
            assert_eq!(hp.t_geometry, None, "{:?} 应当被当成没设过", text);
        }
    }

    #[test]
    fn null_ckpt_value_keeps_the_default() {
        let args = Json::parse(r#"{"t_geometry": null, "growth_rate": null}"#).unwrap();
        let mut hp = Hyperparams::default();
        merge_ckpt_args(&mut hp, &args).unwrap();
        assert_eq!(hp.t_geometry, None);
        assert_eq!(hp.growth_rate, 16);
    }

    #[test]
    fn bad_channel_order_is_rejected_at_merge_time() {
        let args = Json::parse(r#"{"channel_order": "pvx"}"#).unwrap();
        let mut hp = Hyperparams::default();
        assert!(
            merge_ckpt_args(&mut hp, &args).is_err(),
            "写错的通道顺序会原样写进输出属性，必须当场报错"
        );
        let args = Json::parse(r#"{"task": "diffusion"}"#).unwrap();
        assert!(merge_ckpt_args(&mut Hyperparams::default(), &args).is_err());
    }

    #[test]
    fn defaults_match_python_default_hp() {
        let hp = Hyperparams::default();
        assert_eq!(hp.blocks, vec![3, 6, 3]);
        assert_eq!(hp.growth_rate, 16);
        assert_eq!(hp.init_features, 48);
        assert_eq!(hp.drop_rate, 0.0);
        assert_eq!(hp.bn_size, 8);
        assert!(!hp.bottleneck);
        assert_eq!(hp.num_heads, 4);
        assert_eq!(hp.attn_max_tokens, 8192);
        assert!(!hp.no_positional_encoding);
        assert_eq!(hp.channel_order, "pvv");
        assert_eq!(hp.task, "flow");
        assert_eq!(hp.diffusivity, 1e-5);
        assert_eq!(hp.t_geometry, None);
        assert!(!hp.no_mask);
    }

    #[test]
    fn positional_encoding_flag_is_negated_for_the_model() {
        let mut hp = Hyperparams::default();
        assert!(hp.model_config_2d(1, 3, [32, 32]).use_positional_encoding);
        hp.no_positional_encoding = true;
        assert!(!hp.model_config_2d(1, 3, [32, 32]).use_positional_encoding);
    }

    // ------------------------------------------------------------ 掩码来源

    #[test]
    fn mask_defaults_to_the_checkpoint_geometry() {
        let hp = Hyperparams {
            t_geometry: Some(r#"{"a": [0.0, 0.5, 0.0, 1.0]}"#.into()),
            ..Default::default()
        };
        let choice = resolve_mask(&hp, None, true).unwrap();
        match choice {
            MaskChoice::Applied { spec, source } => {
                assert_eq!(source, MaskSource::Checkpoint);
                assert_eq!(
                    spec,
                    GeometrySpec::Segments(vec![Segment::new(0.0, 0.5, 0.0, 1.0)])
                );
            }
            other => panic!("期望用 checkpoint 的几何，实得 {:?}", other),
        }
    }

    #[test]
    fn command_line_geometry_overrides_the_checkpoint() {
        let hp = Hyperparams {
            t_geometry: Some(r#"{"a": [0.0, 0.5, 0.0, 1.0]}"#.into()),
            ..Default::default()
        };
        let choice = resolve_mask(&hp, Some(r#"{"b": [0.5, 1.0, 0.0, 1.0]}"#), true).unwrap();
        match choice {
            MaskChoice::Applied { spec, source } => {
                assert_eq!(source, MaskSource::CommandLine);
                assert_eq!(
                    spec,
                    GeometrySpec::Segments(vec![Segment::new(0.5, 1.0, 0.0, 1.0)])
                );
            }
            other => panic!("命令行几何必须压过 checkpoint，实得 {:?}", other),
        }
    }

    #[test]
    fn trained_without_mask_is_honored() {
        let hp = Hyperparams {
            no_mask: true,
            ..Default::default()
        };
        assert_eq!(
            resolve_mask(&hp, None, true).unwrap(),
            MaskChoice::SkippedTrainedWithoutMask
        );
        // 但命令行显式给了几何就照裁——用户明确要求
        assert!(matches!(
            resolve_mask(&hp, Some(r#"{"a": [0.0, 1.0, 0.0, 1.0]}"#), true).unwrap(),
            MaskChoice::Applied { .. }
        ));
        assert_eq!(
            resolve_mask(&hp, None, false).unwrap(),
            MaskChoice::Disabled
        );
    }

    #[test]
    fn default_geometry_matches_the_t_pipe() {
        // 137x137 是设计分辨率，覆盖率应当就是方案里那个 44.9%
        let mask = GeometrySpec::Default.mask(137, 137).unwrap();
        let cov = coverage(&mask);
        assert!((cov - 0.449).abs() < 0.002, "T 型几何覆盖率 {:.4}", cov);
    }

    // ------------------------------------------------------------ JSON

    #[test]
    fn json_parses_windows_paths_and_python_specials() {
        let v = Json::parse(
            r#"{"train_file": "F:\\data\\x.hdf5", "best_rmse": NaN, "final_r2": null,
                "domain_size": [1.0, 1.0], "ok": true, "n": -1.5e-3}"#,
        )
        .unwrap();
        assert_eq!(
            v.get("train_file").unwrap().as_str().unwrap(),
            r"F:\data\x.hdf5"
        );
        assert!(v.get("best_rmse").unwrap().as_f64().unwrap().is_nan());
        assert!(v.get("final_r2").unwrap().is_null());
        assert_eq!(v.get("domain_size").unwrap().as_array().unwrap().len(), 2);
        assert!(v.get("ok").unwrap().as_bool().unwrap());
        assert_eq!(v.get("n").unwrap().as_f64().unwrap(), -1.5e-3);
    }

    #[test]
    fn json_unicode_escapes_round_trip() {
        let v = Json::parse(r#"{"s": "\u4e2d\u6587\t\u00e9"}"#).unwrap();
        assert_eq!(v.get("s").unwrap().as_str().unwrap(), "中文\té");
    }

    /// 参考值由 `python -c "print(repr(v))"` 逐个取回来的——诊断报告要和 Python
    /// 逐行对上，浮点打印差一个字符就对不上了。
    #[test]
    fn float_repr_matches_python() {
        use super::json::fmt_py_float;
        for (v, want) in [
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (1e-4, "0.0001"),
            (0.001, "0.001"),
            (1.0, "1.0"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123456.789, "123456.789"),
            (-2.5e-7, "-2.5e-07"),
            (3.0, "3.0"),
            (0.2, "0.2"),
            (2e20, "2e+20"),
            (1.0 / 3.0, "0.3333333333333333"),
        ] {
            assert_eq!(fmt_py_float(v), want, "{} 的 Python repr 是 {}", v, want);
        }
        assert_eq!(fmt_py_float(f64::NAN), "nan");
        assert_eq!(fmt_py_float(f64::INFINITY), "inf");
    }

    /// JSON 里的 `200` 与 `200.0` 在 Python 侧还原成 int / float，打印不一样。
    #[test]
    fn integer_and_float_literals_print_differently() {
        let v = Json::parse(r#"{"epochs": 200, "lr": 200.0, "d": 1e-05}"#).unwrap();
        assert_eq!(v.get("epochs").unwrap().py_repr(), "200");
        assert_eq!(v.get("lr").unwrap().py_repr(), "200.0");
        assert_eq!(v.get("d").unwrap().py_repr(), "1e-05");
        assert_eq!(
            Json::parse(r#"[1, 2.5, "a", true, null]"#)
                .unwrap()
                .py_repr(),
            "[1, 2.5, 'a', True, None]"
        );
    }

    #[test]
    fn json_rejects_trailing_garbage() {
        assert!(Json::parse(r#"{"a": 1} oops"#).is_err());
        assert!(Json::parse(r#"{"a": }"#).is_err());
    }

    #[test]
    fn hyperparams_json_round_trips() {
        let hp = Hyperparams {
            t_geometry: Some(r"dataset\station_mask.npy".into()),
            blocks: vec![1, 2, 1],
            ..Default::default()
        };
        let text = hp.to_json_string();
        let back = Json::parse(&text).unwrap_or_else(|e| panic!("{} -> {}", text, e));
        let mut restored = Hyperparams::default();
        merge_ckpt_args(&mut restored, &back).unwrap();
        assert_eq!(restored, hp);
    }

    // ------------------------------------------------------------ .npy

    #[test]
    fn npy_reader_rejects_non_npy_and_bad_dtype() {
        let dir = std::env::temp_dir().join(format!("eser_p6_npy_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("not.npy");
        std::fs::write(&bad, b"just some text, definitely not numpy").unwrap();
        assert!(read_npy_2d(&bad).is_err());

        // 手工拼一个合法的 2x3 f4 v1 头，验证正常路径
        let good = dir.join("ok.npy");
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
        let mut padded = header.to_string();
        while !(10 + padded.len() + 1).is_multiple_of(64) {
            padded.push(' ');
        }
        padded.push('\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend((padded.len() as u16).to_le_bytes());
        bytes.extend(padded.as_bytes());
        for v in [0.0f32, 1.0, 0.0, 1.0, 1.0, 0.0] {
            bytes.extend(v.to_le_bytes());
        }
        std::fs::write(&good, &bytes).unwrap();
        let (data, h, w) = read_npy_2d(&good).unwrap();
        assert_eq!((h, w), (2, 3));
        assert_eq!(data, vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_out_file_mirrors_python() {
        let p = default_out_file(Path::new("dataset/p_v_256.hdf5"));
        assert_eq!(p, Path::new("dataset").join("p_v_256.pred.hdf5"));
    }
}
