//! 合成 CFD 数据集生成器，移植自 `tests/make_fake_data.py`。
//!
//! ```text
//! eser-makedata [--out-dir DIR]     # 默认 ./dataset
//! ```
//!
//! **验收标准是和 h5py 那版逐位相同**，不是「统计上接近」：`tests/data_golden.rs`
//! 里的 `stats_on_h5py_file_match_python` 把 `fake2d.hdf5` 的 min/max/std 写死了。
//! 因此这里的每一处都在刻意复刻 numpy 的算术细节：
//!
//! * 中间量一律走 f64，**只在写进数组时才截成 f32**——numpy 的 `out[i, 0] = <f64 表达式>`
//!   就是这个语义，提前降精度会让低位全错。
//! * 运算顺序照抄 Python 的结合律（`100.0 * a * sin * cos` 是 `((100*a)*sin)*cos`），
//!   浮点乘法不满足结合律，换个括号位置就差最后一位。
//! * `np.linspace` 不是「等差累加」，见 [`linspace01`]。

use std::path::{Path, PathBuf};

use eser::data::Mt19937;
use rust_hdf5::H5File;

const PI: f64 = std::f64::consts::PI;

// ================================================================ 网格与落盘

/// 对应 `np.linspace(0.0, 1.0, n)`。
///
/// numpy 的实现是 `i * step` 再把**末元素显式钉成 stop**，不是逐步累加。
///
/// 钉这一下不是多余：`(n-1) * (1/(n-1))` 只是「大多数时候」等于 1.0，2..4000 之间
/// 有 483 个 n 不等于，最小的是 50——`linspace(0, 1, 50)[49]` 不钉就是
/// 0.9999999999999999。本项目当前用的 65/96/128/137 恰好都落在相等的那一边，所以
/// 这行是防御性的：换个网格尺寸就会真的用上，而差一个 ulp 不报错、只让统计量悄悄飘。
fn linspace01(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![0.0; n];
    }
    let step = 1.0 / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
    v[n - 1] = 1.0;
    v
}

/// 写出 `input` / `output` 两个 f32 数据集，对应 `h5py.File(path, 'w')` 那几行。
///
/// 不加分块、不加压缩，和 h5py `create_dataset(data=...)` 的默认行为一致。
fn write_h5(
    path: &Path,
    in_dims: &[usize],
    inp: &[f32],
    out_dims: &[usize],
    out: &[f32],
) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| format!("建不了目录 {:?}: {}", dir, e))?;
        }
    }
    let file = H5File::create(path).map_err(|e| format!("建不了 {:?}: {}", path, e))?;
    for (name, dims, data) in [("input", in_dims, inp), ("output", out_dims, out)] {
        file.new_dataset::<f32>()
            .shape(dims)
            .create(name)
            .map_err(|e| format!("建不了 {} 数据集: {}", name, e))?
            .write_raw(data)
            .map_err(|e| format!("写不进 {}: {}", name, e))?;
    }
    Ok(())
}

/// 写出 2D f32 的 `.npy`，对应 `np.save`。读取端见 `predict::read_npy_2d`。
///
/// 格式版本 1.0：魔数 6 字节 + 版本 2 字节 + 头长度 2 字节。头是一行 Python 字典
/// 字面量，**键按字母序**，末尾用空格补到「总头部长度是 64 的整数倍」再加换行——
/// 对齐是为了让数据区起始地址可以直接 mmap，少补几个空格 numpy 自己也读得回来，
/// 但那就不是逐字节相同的文件了。
fn write_npy_2d(path: &Path, h: usize, w: usize, data: &[f32]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| format!("建不了目录 {:?}: {}", dir, e))?;
        }
    }
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        h, w
    );
    let header_len = dict.len() + 1; // +1 是结尾的换行
    let pad = 64 - ((8 + 2 + header_len) % 64); // 整除时补满 64，numpy 也是这么算的
    let mut buf = Vec::with_capacity(10 + header_len + pad + data.len() * 4);
    buf.extend_from_slice(b"\x93NUMPY\x01\x00");
    buf.extend_from_slice(&((header_len + pad) as u16).to_le_bytes());
    buf.extend_from_slice(dict.as_bytes());
    buf.resize(buf.len() + pad, b' ');
    buf.push(b'\n');
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, buf).map_err(|e| format!("写不了 {:?}: {}", path, e))
}

// ================================================================ 各数据集

/// 对应 `make_2d`：工况标量 a -> (P, Vx, Vy)。
///
/// 三个通道的量级刻意拉开到 100 : 2 : 0.3，用来验证「逐通道归一化 + 反归一化」
/// 真的生效——三个通道共用一套 min/max 的话，后两个通道会被压成 0。
fn make_2d(path: &Path, n: usize, c: usize, h: usize, w: usize, seed: u32) -> Result<(), String> {
    let mut rng = Mt19937::new(seed);
    let (ly, lx) = (linspace01(h), linspace01(w));
    // sin/cos 只依赖单个坐标轴，提到循环外算；提取本身不改数值，只是省时间
    let sin2x: Vec<f64> = lx.iter().map(|x| (2.0 * PI * x).sin()).collect();
    let sin3x: Vec<f64> = lx.iter().map(|x| (3.0 * PI * x).sin()).collect();
    let cos_y: Vec<f64> = ly.iter().map(|y| (PI * y).cos()).collect();
    let cos2y: Vec<f64> = ly.iter().map(|y| (2.0 * PI * y).cos()).collect();

    let plane = h * w;
    let mut out = vec![0f32; n * c * plane];
    let mut inp = vec![0f32; n * plane];
    for i in 0..n {
        let a = 0.5 + rng.rand();
        let base = i * c * plane;
        for r in 0..h {
            for j in 0..w {
                let k = base + r * w + j;
                out[k] = (100.0 * a * sin2x[j] * cos_y[r]) as f32; // P
                if c > 1 {
                    out[k + plane] = (2.0 * a * cos2y[r]) as f32; // Vx
                }
                if c > 2 {
                    out[k + 2 * plane] = (0.3 * a * sin3x[j]) as f32; // Vy
                }
            }
        }
        inp[i * plane..(i + 1) * plane].fill(a as f32);
    }
    write_h5(path, &[n, 1, h, w], &inp, &[n, c, h, w], &out)
}

/// 对应 `make_3d`：多一个深度轴，P 沿深度线性增强。
fn make_3d(
    path: &Path,
    n: usize,
    c: usize,
    d: usize,
    h: usize,
    w: usize,
    seed: u32,
) -> Result<(), String> {
    let mut rng = Mt19937::new(seed);
    let (lz, ly, lx) = (linspace01(d), linspace01(h), linspace01(w));
    let sin2x: Vec<f64> = lx.iter().map(|x| (2.0 * PI * x).sin()).collect();
    let sin3x: Vec<f64> = lx.iter().map(|x| (3.0 * PI * x).sin()).collect();
    let cos2y: Vec<f64> = ly.iter().map(|y| (2.0 * PI * y).cos()).collect();

    let vol = d * h * w;
    let mut out = vec![0f32; n * c * vol];
    let mut inp = vec![0f32; n * vol];
    for i in 0..n {
        let a = 0.5 + rng.rand();
        let base = i * c * vol;
        for (zi, z) in lz.iter().enumerate() {
            for (r, c2y) in cos2y.iter().enumerate() {
                for j in 0..w {
                    let k = base + (zi * h + r) * w + j;
                    out[k] = (50.0 * a * sin2x[j] * (1.0 + z)) as f32;
                    if c > 1 {
                        out[k + vol] = (2.0 * a * c2y) as f32;
                    }
                    if c > 2 {
                        out[k + 2 * vol] = (0.3 * a * sin3x[j]) as f32;
                    }
                }
            }
        }
        inp[i * vol..(i + 1) * vol].fill(a as f32);
    }
    write_h5(path, &[n, 1, d, h, w], &inp, &[n, c, d, h, w], &out)
}

/// 对应 `make_flow_stage1`：地铁站阶段一，工况 -> (P, Vx, Vy, T)，即 `--channel-order pvvt`。
fn make_flow_stage1(path: &Path, n: usize, h: usize, w: usize, seed: u32) -> Result<(), String> {
    let mut rng = Mt19937::new(seed);
    let (ly, lx) = (linspace01(h), linspace01(w));
    let sin2x: Vec<f64> = lx.iter().map(|x| (2.0 * PI * x).sin()).collect();
    let sin3x: Vec<f64> = lx.iter().map(|x| (3.0 * PI * x).sin()).collect();
    let cos_y: Vec<f64> = ly.iter().map(|y| (PI * y).cos()).collect();
    let cos2y: Vec<f64> = ly.iter().map(|y| (2.0 * PI * y).cos()).collect();
    // T 的热羽：exp(-(x-0.5)^2 * 8)，只沿 x 变化
    let plume: Vec<f64> = lx
        .iter()
        .map(|x| {
            let dx = x - 0.5;
            (-(dx * dx) * 8.0).exp()
        })
        .collect();

    let plane = h * w;
    let mut out = vec![0f32; n * 4 * plane];
    let mut inp = vec![0f32; n * 3 * plane];
    // door / train 对应 randint(2) / randint(4)：numpy 在上界不超过 2^32 时走 32 位
    // 掩码拒绝采样，与 interval(n-1) 等价——抽取次数一致，序列才不会分岔
    for i in 0..n {
        let vent = 0.5 + rng.rand(); // 通风量
        let door = rng.interval(1) as f64; // 屏蔽门开闭
        let train = rng.interval(3) as f64; // 列车工况
        let base = i * 4 * plane;
        for r in 0..h {
            for j in 0..w {
                let k = base + r * w + j;
                out[k] = (120.0 * vent * sin2x[j] * cos_y[r]) as f32; // P
                out[k + plane] = (2.0 * vent * cos2y[r] + 0.3 * door) as f32; // Vx
                out[k + 2 * plane] = (0.4 * vent * sin3x[j]) as f32; // Vy
                out[k + 3 * plane] = (293.0 + 6.0 * vent * plume[j]) as f32; // T
            }
        }
        let ibase = i * 3 * plane;
        for (ch, v) in [vent, door, train].iter().enumerate() {
            inp[ibase + ch * plane..ibase + (ch + 1) * plane].fill(*v as f32);
        }
    }
    write_h5(path, &[n, 3, h, w], &inp, &[n, 4, h, w], &out)
}

/// 对应 `make_dispersion_stage2`：阶段二，(Vx, Vy, 源位置图, t) -> C，即 `--channel-order c`。
///
/// 浓度是随风漂移的高斯烟团解析解，量级和形态接近真实泄漏扩散，冒烟测试才有意义。
fn make_dispersion_stage2(
    path: &Path,
    n: usize,
    h: usize,
    w: usize,
    seed: u32,
) -> Result<(), String> {
    let mut rng = Mt19937::new(seed);
    let (ly, lx) = (linspace01(h), linspace01(w));
    let sites = [(0.25f64, 0.2f64), (0.5, 0.5), (0.75, 0.8)]; // (sy, sx) 固定关心点位
    let times = [5.0f64, 10.0, 20.0, 40.0];
    let diffusivity = 2e-3f64;

    let plane = h * w;
    let mut out = vec![0f32; n * plane];
    let mut inp = vec![0f32; n * 4 * plane];
    for i in 0..n {
        // Python 的元组赋值从左到右求值：先 rand() 再 randn()，顺序反了整条序列就分岔
        let u = 0.05 + 0.05 * rng.rand();
        let v = 0.02 * rng.randn();
        let (sy, sx) = sites[i % sites.len()];
        let t = times[i % 4];
        let denom = 4.0 * diffusivity * t;
        let (ushift, vshift) = (u * t * 0.01, v * t * 0.01);
        let obase = i * plane;
        let ibase = i * 4 * plane;
        for (r, y) in ly.iter().enumerate() {
            let dy = y - sy - vshift;
            let qy = (y - sy) * (y - sy);
            for (j, x) in lx.iter().enumerate() {
                let dx = x - sx - ushift;
                let k = r * w + j;
                let r2 = dx * dx + dy * dy;
                out[obase + k] = ((-r2 / denom).exp() / (PI * denom)) as f32;
                let qx = (x - sx) * (x - sx);
                inp[ibase + 2 * plane + k] = (-(qx + qy) / 2e-3).exp() as f32; // 源位置图
            }
        }
        inp[ibase..ibase + plane].fill(u as f32); // Vx
        inp[ibase + plane..ibase + 2 * plane].fill(v as f32); // Vy
        inp[ibase + 3 * plane..ibase + 4 * plane].fill(t as f32); // 时刻
    }
    write_h5(path, &[n, 4, h, w], &inp, &[n, 1, h, w], &out)
}

/// 对应 `make_station_mask`：非矩形的车站平面掩码，验证 `.npy` 栅格掩码通路。
///
/// 边界全部用 `(len * 系数) as usize` 现算，不写死数字——`96 * 0.6` 在 f64 下是
/// 57.599999999999994，截断得 57 而不是 58，硬编码「正确」值反而会对不上 Python。
fn make_station_mask(path: &Path, h: usize, w: usize) -> Result<(), String> {
    let mut mask = vec![0f32; h * w];
    let hi = |f: f64| (h as f64 * f) as usize;
    let wi = |f: f64| (w as f64 * f) as usize;

    for r in hi(0.40)..hi(0.60) {
        mask[r * w..(r + 1) * w].fill(1.0); // 站台层，横贯整幅
    }
    for r in hi(0.15)..hi(0.40) {
        mask[r * w + wi(0.20)..r * w + wi(0.30)].fill(1.0); // 楼梯 1
        mask[r * w + wi(0.70)..r * w + wi(0.80)].fill(1.0); // 楼梯 2
    }
    for i in 0..hi(0.2) {
        // 斜向通道：每下两行右移一格，宽 4 格
        let (r, c0) = (hi(0.60) + i / 2, wi(0.45) + i);
        if r < h && c0 < w {
            mask[r * w + c0..r * w + (c0 + 4).min(w)].fill(1.0);
        }
    }
    write_npy_2d(path, h, w, &mask)
}

// ================================================================ 入口

const USAGE: &str = "用法：eser-makedata [--out-dir DIR]（默认 dataset）";

fn main() {
    let mut out_dir = PathBuf::from("dataset");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let value = match a.as_str() {
            "-h" | "--help" => {
                println!("{}", USAGE);
                return;
            }
            "--out-dir" => args.next(),
            // argparse 同时接受 `--out-dir=X` 写法，这里也认
            other => other.strip_prefix("--out-dir=").map(str::to_string),
        };
        match value {
            Some(v) => out_dir = PathBuf::from(v),
            // 和 argparse 一样，参数出错退出码 2；但要分清「不认识」和「少了值」
            None => {
                if a == "--out-dir" {
                    eprintln!("--out-dir 后面缺少目录\n{}", USAGE);
                } else {
                    eprintln!("无法识别的参数 {:?}\n{}", a, USAGE);
                }
                std::process::exit(2);
            }
        }
    }

    // 生成一个产物：失败就退出码 1，成功打印路径——和 Python 版的 print 一一对应
    let step = |name: &str, build: &dyn Fn(&Path) -> Result<(), String>| {
        let path = out_dir.join(name);
        if let Err(e) = build(&path) {
            eprintln!("生成 {:?} 失败：{}", path, e);
            std::process::exit(1);
        }
        println!("{}", path.display());
    };
    // 形状与 seed 与 make_fake_data.py 的 `__main__` 一一对应，改任何一个都会让黄金数据失效
    step("fake2d.hdf5", &|p| make_2d(p, 24, 3, 137, 137, 0));
    step("fake2d_even.hdf5", &|p| make_2d(p, 24, 3, 128, 128, 7));
    step("fake2d_1ch.hdf5", &|p| make_2d(p, 24, 1, 65, 65, 3));
    step("fake3d.hdf5", &|p| make_3d(p, 12, 3, 2, 65, 65, 1));
    step("fake_stage1.hdf5", &|p| make_flow_stage1(p, 24, 96, 96, 11));
    step("fake_stage2.hdf5", &|p| {
        make_dispersion_stage2(p, 32, 96, 96, 13)
    });
    step("station_mask.npy", &|p| make_station_mask(p, 96, 96));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_endpoints_are_exact() {
        // 端点必须是精确的 0.0 / 1.0。50 是刻意放进来的：它是最小的、光靠 i*step
        // 凑不出 1.0 的尺寸，去掉 linspace01 里那行「钉住末元素」就只有它会红
        for n in [2usize, 50, 65, 96, 128, 137] {
            let v = linspace01(n);
            assert_eq!(v.len(), n);
            assert_eq!(v[0], 0.0);
            assert_eq!(v[n - 1], 1.0);
            assert_eq!(v[1], 1.0 / (n - 1) as f64);
        }
    }

    #[test]
    fn npy_header_is_64_byte_aligned() {
        let dir = std::env::temp_dir().join("eser_makedata_npy_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.npy");
        make_station_mask(&path, 96, 96).unwrap();
        let raw = std::fs::read(&path).unwrap();

        assert_eq!(&raw[..8], b"\x93NUMPY\x01\x00");
        let hlen = u16::from_le_bytes([raw[8], raw[9]]) as usize;
        assert_eq!((10 + hlen) % 64, 0, "数据区没有对齐到 64 字节");
        assert_eq!(raw[10 + hlen - 1], b'\n');
        assert_eq!(raw.len(), 10 + hlen + 96 * 96 * 4);
        // 站台层横贯整幅：第 40 行（96*0.40=38.4 -> 38 起）应当全是 1
        let body = &raw[10 + hlen..];
        let at = |r: usize, c: usize| {
            f32::from_le_bytes(body[(r * 96 + c) * 4..][..4].try_into().unwrap())
        };
        assert_eq!(at(38, 0), 1.0);
        assert_eq!(at(37, 0), 0.0);
        assert_eq!(at(57, 0), 0.0, "站台层上界是开区间 hi(0.60)=57");
        std::fs::remove_dir_all(&dir).ok();
    }
}
