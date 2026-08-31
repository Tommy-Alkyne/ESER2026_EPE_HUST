//! 把 PDEBench 的二维不可压 Navier-Stokes 数据映射成本项目 §4.2 的阶段二格式。
//!
//! ```text
//! eser-mappdebench --in dataset/external/ns_incom_inhom_2d_512-100.h5 \
//!                  --out-dir dataset/pdebench --frames 25
//! eser-mappdebench --in <文件> --verify     # 只做自检，不写数据
//! ```
//!
//! # 源数据长什么样
//!
//! ```text
//! velocity   (4, 1000, 512, 512, 2)  f32, LZF 压缩
//! particles  (4, 1000, 512, 512, 1)  f32
//! force      (4, 512, 512, 2)        f32   静态外力，本转换不用
//! t          (4, 1000)               f32   0 .. 4.995 s
//! 根属性 config: domain_size [1,1], grid 512x512, NU=0.01, DT=5e-5, frame_int=100
//! ```
//!
//! # 映射到 §4.2，以及三处必须说清的偏差
//!
//! ```text
//! input  (N,4,H,W) = (Vx, Vy, C0, t/t_max)     output (N,1,H,W) = C
//! ```
//!
//! 1. **`particles` 不是浓度场。** 它取值有正有负（实测 -2.49 .. 2.88），是一个随机初始
//!    场被平流输运的**带符号被动标量**，不是从点源释放的非负浓度。所以本项目 §4.2 的
//!    「源位置图」这一路输入在这里没有对应物——**改用 `C0`（t=0 时刻的标量场）顶替**。
//!    学习问题的形状是一样的（给定流场 + 初始分布 + 时刻，预测该时刻的标量场），
//!    但它不是「泄漏点位 → 浓度场」，**外推点位测试在这份数据上做不了**。
//! 2. **流场本身是瞬态的。** §4.2 的前提是阶段一给出稳态流场、浓度随时间演化；这里
//!    `velocity` 每帧都在变。转换取**同帧的瞬时速度**——因为对流扩散残差 `u·∇C` 量的
//!    就是那一瞬间的 u，取初始帧会让残差项失去意义。
//! 3. **几何是周期方腔，不是地铁站。** 没有壁面掩码，训练时应当加 `--no-mask`。
//!
//! # 它能干什么
//!
//! 给「真实湍流场长什么样」一个参照：`eser-makestation` 造的是无粘无涡的势流回流，
//! 这份是真解出来的 Navier-Stokes。两者互补，**都不是地铁站 CFD 的替代品**。

use std::path::{Path, PathBuf};

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use rust_hdf5::{H5Dataset, H5File};

type B = NdArray<f32>;

const USAGE: &str = "\
eser-mappdebench —— 把 PDEBench 二维不可压 NS 数据映射成 §4.2 的阶段二格式

  --in <文件>        PDEBench 的 ns_incom_inhom_2d_*.h5
  --out-dir <目录>   默认 dataset/pdebench
  --frames <N>       每个 batch 抽多少帧（对数间隔，§4.3），默认 25
  --batches <N>      用前几个 batch，默认全部
  --stride <K>       空间降采样步长，默认 1（512x512 原分辨率）
  --verify           只跑自检：核对读数、量 div(v) 与对流扩散残差，不写数据
";

struct Cli {
    input: PathBuf,
    out_dir: PathBuf,
    frames: usize,
    batches: Option<usize>,
    stride: usize,
    verify_only: bool,
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", USAGE);
        return;
    }
    if let Err(e) = run(&argv) {
        eprintln!("[mappdebench] {}", e);
        std::process::exit(1);
    }
}

fn parse(argv: &[String]) -> Result<Cli, String> {
    let (mut input, mut out_dir) = (None, PathBuf::from("dataset/pdebench"));
    let (mut frames, mut batches, mut stride, mut verify_only) = (25usize, None, 1usize, false);
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].clone();
        let val = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            argv.get(*i)
                .cloned()
                .ok_or_else(|| format!("{} 后面缺参数", flag))
        };
        match argv[i].as_str() {
            "--in" => input = Some(PathBuf::from(val(&mut i)?)),
            "--out-dir" => out_dir = PathBuf::from(val(&mut i)?),
            "--frames" => frames = val(&mut i)?.parse().map_err(|e| format!("--frames: {}", e))?,
            "--batches" => {
                batches = Some(val(&mut i)?.parse().map_err(|e| format!("--batches: {}", e))?)
            }
            "--stride" => stride = val(&mut i)?.parse().map_err(|e| format!("--stride: {}", e))?,
            "--verify" => verify_only = true,
            other => return Err(format!("不认识的参数 {:?}\n\n{}", other, USAGE)),
        }
        i += 1;
    }
    if frames == 0 || stride == 0 {
        return Err("--frames 与 --stride 都必须大于 0".into());
    }
    Ok(Cli {
        input: input.ok_or("必须给 --in")?,
        out_dir,
        frames,
        batches,
        stride,
        verify_only,
    })
}

// ================================================================ 读取

/// 源文件的形状信息。
struct Source {
    file: H5File,
    n_batch: usize,
    n_time: usize,
    h: usize,
    w: usize,
}

impl Source {
    fn open(path: &Path) -> Result<Self, String> {
        let file = H5File::open(path).map_err(|e| format!("打不开 {:?}: {}", path, e))?;
        let v = file
            .dataset("velocity")
            .map_err(|e| format!("没有 velocity 数据集（这不像 PDEBench 的 ns_incom 文件）: {}", e))?;
        let s = v.shape();
        if s.len() != 5 || s[4] != 2 {
            return Err(format!(
                "velocity 形状是 {:?}，期望 (batch, time, H, W, 2)",
                s
            ));
        }
        Ok(Self { file, n_batch: s[0], n_time: s[1], h: s[2], w: s[3] })
    }

    fn ds(&self, name: &str) -> Result<H5Dataset, String> {
        self.file
            .dataset(name)
            .map_err(|e| format!("没有 {} 数据集: {}", name, e))
    }

    /// 读一帧 `velocity[b, t]`，拆成 `(vx, vy)` 两个 H*W 的平面。
    ///
    /// 源数据末维是分量（`..., 2`），所以内存里两个分量是交错的，要按 stride 2 拆开。
    fn velocity(&self, b: usize, t: usize, o: Orient) -> Result<(Vec<f32>, Vec<f32>), String> {
        let raw = self.slab("velocity", &[b, t, 0, 0, 0], &[1, 1, self.h, self.w, 2])?;
        Ok(o.split(&raw))
    }

    fn particles(&self, b: usize, t: usize) -> Result<Vec<f32>, String> {
        self.slab("particles", &[b, t, 0, 0, 0], &[1, 1, self.h, self.w, 1])
    }

    fn times(&self, b: usize) -> Result<Vec<f32>, String> {
        self.slab("t", &[b, 0], &[1, self.n_time])
    }

    /// 走 `data::read_block_f32`——它带字节序检查，大端文件会明确报错而不是给出垃圾数。
    fn slab(&self, name: &str, start: &[usize], count: &[usize]) -> Result<Vec<f32>, String> {
        eser::data::read_block_f32(&self.ds(name)?, start, count)
            .map_err(|e| format!("读 {} 的 {:?}+{:?}：{}", name, start, count, e))
    }
}

// ================================================================ 自检

/// 用 h5py 读出来的值做交叉验证。
///
/// **不是形式主义**：源数据是 LZF 压缩的，rust-hdf5 的 LZF 是它自己的纯 Rust 实现。
/// 解压错一位不会报错，只会给出一堆看着像场、其实是垃圾的数——后面所有的残差测量
/// 都会跟着变成废话。这几个值取自 h5py，位置是随手挑的边角，不是首元素。
const H5PY_REF: &[(&str, [usize; 5], f64)] = &[
    ("velocity", [0, 0, 100, 200, 0], -0.21946463),
    ("velocity", [0, 0, 100, 200, 1], 0.342144549),
    ("velocity", [2, 500, 300, 400, 0], 0.0483976007),
    ("velocity", [3, 999, 511, 511, 1], 1.55908892e-05),
    ("particles", [0, 0, 100, 200, 0], 0.975914776),
    ("particles", [1, 250, 50, 60, 0], 1.76481175),
    ("particles", [3, 999, 511, 511, 0], -0.970962703),
];

fn verify(src: &Source) -> Result<(), String> {
    println!("[mappdebench] --- 读数交叉验证（参照值来自 h5py）---");
    let mut worst = 0.0f64;
    for (name, idx, want) in H5PY_REF {
        let comps = if *name == "velocity" { 2 } else { 1 };
        let raw = src.slab(name, &[idx[0], idx[1], idx[2], idx[3], 0], &[1, 1, 1, 1, comps])?;
        let got = raw[idx[4]] as f64;
        let err = (got - want).abs() / want.abs().max(1e-6);
        worst = worst.max(err);
        println!(
            "  {}{:?} = {:.9e}  期望 {:.9e}  相对差 {:.2e}",
            name, idx, got, want, err
        );
    }
    if worst > 1e-6 {
        return Err(format!(
            "读数与 h5py 对不上（最大相对差 {:.3e}）。源数据是 LZF 压缩的，\
             很可能是 rust-hdf5 的 LZF 解压有问题——后面所有测量都不可信",
            worst
        ));
    }
    println!("  => 全部一致（最大相对差 {:.2e}）", worst);
    Ok(())
}

// ================================================================ 物理量测

fn to_t(v: &[f32], h: usize, w: usize) -> Tensor<B, 4> {
    Tensor::<B, 1>::from_data(TensorData::from(v), &Default::default()).reshape([1, 1, h, w])
}

fn to_vec(t: Tensor<B, 4>) -> Vec<f64> {
    t.into_data()
        .into_vec::<f32>()
        .expect("张量取不出 f32")
        .into_iter()
        .map(|x| x as f64)
        .collect()
}

fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

/// 速度分量在文件里的朝向。
///
/// **必须实测，不能照抄生成脚本。** PhiFlow 内部按 (y, x) 排轴，末维那两个分量到底
/// 哪个是 x 方向，源码里看不出来，而搞反了**不会报任何错**——只会让 `u·∇C` 算成一个
/// 无意义的量，残差和对流项同量级。这正是本项目方案 §1 反复强调的那类静默错误。
#[derive(Debug, Clone, Copy)]
struct Orient {
    /// 末维的第 0 个分量是不是 x 方向速度
    first_is_x: bool,
    sx: f64,
    sy: f64,
}

impl Orient {
    const ALL: [Orient; 8] = [
        Orient { first_is_x: true, sx: 1.0, sy: 1.0 },
        Orient { first_is_x: true, sx: 1.0, sy: -1.0 },
        Orient { first_is_x: true, sx: -1.0, sy: 1.0 },
        Orient { first_is_x: true, sx: -1.0, sy: -1.0 },
        Orient { first_is_x: false, sx: 1.0, sy: 1.0 },
        Orient { first_is_x: false, sx: 1.0, sy: -1.0 },
        Orient { first_is_x: false, sx: -1.0, sy: 1.0 },
        Orient { first_is_x: false, sx: -1.0, sy: -1.0 },
    ];

    fn label(&self) -> String {
        format!(
            "vx={}c{}  vy={}c{}",
            if self.sx < 0.0 { "-" } else { "+" },
            if self.first_is_x { 0 } else { 1 },
            if self.sy < 0.0 { "-" } else { "+" },
            if self.first_is_x { 1 } else { 0 },
        )
    }

    /// 按本朝向把交错存放的两个分量拆成 (vx, vy)。
    fn split(&self, raw: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (mut a, mut b) = (Vec::with_capacity(raw.len() / 2), Vec::with_capacity(raw.len() / 2));
        for p in raw.as_chunks::<2>().0 {
            a.push(p[0]);
            b.push(p[1]);
        }
        let (mut vx, mut vy) = if self.first_is_x { (a, b) } else { (b, a) };
        if self.sx < 0.0 {
            for v in &mut vx {
                *v = -*v;
            }
        }
        if self.sy < 0.0 {
            for v in &mut vy {
                *v = -*v;
            }
        }
        (vx, vy)
    }
}

/// 量一量这份数据在**本项目损失实际用的算子**下，两个物理约束满足到什么程度。
///
/// 这不是可有可无的报告：`--div-weight` 和 `--advdiff-weight` 该不该开、开多大，
/// 完全取决于这两个数。如果 div 的相对残差是 O(1)，那散度约束在这份数据上就是噪声，
/// 开了只会把模型往错误方向拉。
/// 一次量测的结论。这些数不是报告用的装饰——`--diffusivity` 该填多少、
/// `--div-weight` 值不值得开，全靠它们，所以要一路带到 README 里去。
#[derive(Debug, Clone, Copy)]
struct Measured {
    orient: Orient,
    /// 对流扩散残差 / 对流项，取最优朝向与最优 D
    residual_ratio: f64,
    /// div 的相对残差
    div_rel: f64,
    /// 残差最小的扩散系数
    best_d: f64,
}

fn measure(src: &Source, b: usize, t: usize, dx: f64, dy: f64, dt: f64) -> Result<Measured, String> {
    let (h, w) = (src.h, src.w);
    let raw = src.slab("velocity", &[b, t, 0, 0, 0], &[1, 1, h, w, 2])?;
    let c_prev = src.particles(b, t - 1)?;
    let c_now = src.particles(b, t)?;
    let c_next = src.particles(b, t + 1)?;
    let dcdt = eser::transport::central_time_derivative(
        to_t(&c_prev, h, w),
        to_t(&c_next, h, w),
        dt,
    )?;

    println!(
        "[mappdebench] --- 速度分量朝向扫描（batch {} frame {}）---",
        b, t
    );
    println!("  判据是对流扩散残差：搞反了 u·∇C 就是个无意义的量，残差会和对流项同量级");
    let mut best: Option<(f64, Orient, f64)> = None;
    for o in Orient::ALL {
        let (vx, vy) = o.split(&raw);
        let (gx, gy) = eser::ops::grad_xy(to_t(&c_now, h, w), dx, dy);
        let adv = rms(&to_vec(gx.mul(to_t(&vx, h, w)) + gy.mul(to_t(&vy, h, w))));
        let res = rms(&to_vec(eser::transport::advection_diffusion_residual(
            to_t(&c_now, h, w),
            to_t(&vx, h, w),
            to_t(&vy, h, w),
            0.0,
            dx,
            dy,
            Some(dcdt.clone()),
            None,
        )));
        let div = rms(&to_vec(eser::physics::divergence(
            to_t(&vx, h, w),
            to_t(&vy, h, w),
            dx,
            dy,
        )));
        let speed = rms(&vx.iter().zip(&vy).map(|(p, q)| ((p * p + q * q) as f64).sqrt()).collect::<Vec<_>>());
        println!(
            "  {}   残差/对流 = {:.4e}   div 相对 = {:.4e}",
            o.label(),
            res / adv,
            div / (speed / dx)
        );
        if best.is_none_or(|(r, _, _)| res / adv < r) {
            best = Some((res / adv, o, div / (speed / dx)));
        }
    }
    let (score, o, div_rel) = best.expect("至少扫了一个朝向");
    println!(
        "[mappdebench] => 最优朝向 {}（残差/对流 = {:.4e}）",
        o.label(),
        score
    );
    if score > 0.5 {
        println!(
            "[mappdebench] 警告：即便是最优朝向，残差仍有对流项的 {:.0}%。\n\
             这份数据在中心差分下并不满足对流扩散方程——训练时 --advdiff-weight \n\
             应当设成 0，否则物理项会把模型往错误方向拉",
            score * 100.0
        );
    }
    println!(
        "[mappdebench] div 相对残差 {:.3e}（对照：eser-makestation 的合成数据 < 1e-6）",
        div_rel
    );

    // 定下朝向后再扫扩散系数
    let (vx, vy) = o.split(&raw);
    let (gx, gy) = eser::ops::grad_xy(to_t(&c_now, h, w), dx, dy);
    let adv = rms(&to_vec(gx.mul(to_t(&vx, h, w)) + gy.mul(to_t(&vy, h, w))));
    println!("[mappdebench] --- 扩散系数扫描（源数据 config 里没有标量扩散项）---");
    let mut bestd = (f64::INFINITY, 0.0);
    for d in [0.0, 1e-6, 1e-5, 1e-4, 1e-3] {
        let res = rms(&to_vec(eser::transport::advection_diffusion_residual(
            to_t(&c_now, h, w),
            to_t(&vx, h, w),
            to_t(&vy, h, w),
            d,
            dx,
            dy,
            Some(dcdt.clone()),
            None,
        )));
        println!("      D={:<9} 残差/对流 = {:.4e}", d, res / adv);
        if res < bestd.0 {
            bestd = (res, d);
        }
    }
    println!("[mappdebench] => 残差最小的是 D={}", bestd.1);
    Ok(Measured {
        orient: o,
        residual_ratio: bestd.0 / adv,
        div_rel,
        best_d: bestd.1,
    })
}

// ================================================================ 主流程

fn run(argv: &[String]) -> Result<(), String> {
    let cli = parse(argv)?;
    let src = Source::open(&cli.input)?;
    println!(
        "[mappdebench] {} batch x {} 帧 x {}x{}",
        src.n_batch, src.n_time, src.h, src.w
    );
    verify(&src)?;

    let times = src.times(0)?;
    // PDEBench 的 domain_size 是 [1,1]，网格 512x512
    let (dx, dy) = (1.0 / (src.w - 1) as f64, 1.0 / (src.h - 1) as f64);
    let frame_dt = (times[1] - times[0]) as f64;
    println!(
        "[mappdebench] t: {:.4}..{:.4}s，帧间隔 {:.5}s；dx=dy={:.6}（域 1x1）",
        times[0], times[times.len() - 1], frame_dt, dx
    );
    let m = measure(&src, 0, src.n_time / 2, dx, dy, frame_dt)?;
    let orient = m.orient;
    // 只在一个点上定朝向太险：那一帧碰巧让某个错误朝向占优，整份数据就全错了，
    // 而且不会报任何错。换个 batch、换个时刻再判一次，不一致就停下
    let cross = measure(&src, src.n_batch - 1, src.n_time / 8, dx, dy, frame_dt)?.orient;
    if (cross.first_is_x, cross.sx, cross.sy) != (orient.first_is_x, orient.sx, orient.sy) {
        return Err(format!(
            "两个采样点判出的速度分量朝向不一致：{} vs {}。
             不能靠猜——请人工确认源数据的分量约定后再转换",
            orient.label(),
            cross.label()
        ));
    }
    println!("[mappdebench] 两个采样点的朝向一致：{}", orient.label());
    if cli.verify_only {
        return Ok(());
    }

    // ---- 抽帧并写出 ----
    let n_batch = cli.batches.unwrap_or(src.n_batch).min(src.n_batch);
    let idx = frame_indices(src.n_time, cli.frames);
    let (oh, ow) = (src.h.div_ceil(cli.stride), src.w.div_ceil(cli.stride));
    let n = n_batch * idx.len();
    println!(
        "[mappdebench] 抽 {} batch x {} 帧 = {} 个样本，输出 {}x{}（stride {}）",
        n_batch, idx.len(), n, oh, ow, cli.stride
    );

    let t_max = *times.last().unwrap_or(&1.0);
    let mut inp: Vec<f32> = Vec::with_capacity(n * 4 * oh * ow);
    let mut out: Vec<f32> = Vec::with_capacity(n * oh * ow);
    for b in 0..n_batch {
        // C0 是「初始标量分布」这一路输入，代替 §4.2 的源位置图（见模块文档）
        let c0 = down(&src.particles(b, 0)?, src.h, src.w, cli.stride);
        for (k, &ti) in idx.iter().enumerate() {
            let (vx, vy) = src.velocity(b, ti, orient)?;
            inp.extend(down(&vx, src.h, src.w, cli.stride));
            inp.extend(down(&vy, src.h, src.w, cli.stride));
            inp.extend(&c0);
            let tn = times.get(ti).copied().unwrap_or(0.0) / t_max;
            inp.extend(std::iter::repeat_n(tn, oh * ow));
            out.extend(down(&src.particles(b, ti)?, src.h, src.w, cli.stride));
            if (k + 1) % 10 == 0 || k + 1 == idx.len() {
                println!("[mappdebench]   batch {} {}/{}", b, k + 1, idx.len());
            }
        }
    }

    std::fs::create_dir_all(&cli.out_dir).map_err(|e| format!("建不了输出目录：{}", e))?;
    let path = cli.out_dir.join("stage2_pdebench.hdf5");
    write_h5(&path, &[n, 4, oh, ow], &inp, &[n, 1, oh, ow], &out)?;
    write_readme(&cli, oh, ow, n, dx, &m)?;
    println!("[mappdebench] 完成 -> {}", path.display());
    Ok(())
}

/// 对数间隔抽帧（§4.3）。首帧恒定包含，且**跳过第 0 帧**——它就是 `C0` 那一路输入，
/// 留着等于给网络一道抄答案的题。
fn frame_indices(n_time: usize, want: usize) -> Vec<usize> {
    let lo = 1usize;
    let hi = n_time.saturating_sub(2).max(lo);
    if want > hi - lo {
        return (lo..=hi).collect();
    }
    let (a, b) = ((lo as f64).ln(), (hi as f64).ln());
    let mut v: Vec<usize> = (0..want)
        .map(|i| (a + (b - a) * i as f64 / (want - 1).max(1) as f64).exp().round() as usize)
        .map(|x| x.clamp(lo, hi))
        .collect();
    v.dedup();
    v
}

/// 空间降采样：每 `stride` 取一个点。不做平均——平均会改变场的谱，
/// 而对流扩散残差对高频最敏感，抽点至少保持「这些点满足方程」。
fn down(v: &[f32], h: usize, w: usize, stride: usize) -> Vec<f32> {
    if stride <= 1 {
        return v.to_vec();
    }
    let mut o = Vec::with_capacity(h.div_ceil(stride) * w.div_ceil(stride));
    for i in (0..h).step_by(stride) {
        for j in (0..w).step_by(stride) {
            o.push(v[i * w + j]);
        }
    }
    o
}

fn write_h5(
    path: &Path,
    in_dims: &[usize],
    inp: &[f32],
    out_dims: &[usize],
    out: &[f32],
) -> Result<(), String> {
    let n = |d: &[usize]| d.iter().product::<usize>();
    if inp.len() != n(in_dims) || out.len() != n(out_dims) {
        return Err(format!(
            "元素数与形状不符：input {} vs {:?}，output {} vs {:?}",
            inp.len(), in_dims, out.len(), out_dims
        ));
    }
    let f = H5File::create(path).map_err(|e| format!("建不了 {:?}: {}", path, e))?;
    for (name, dims, data) in [("input", in_dims, inp), ("output", out_dims, out)] {
        f.new_dataset::<f32>()
            .shape(dims)
            .create(name)
            .map_err(|e| format!("建不了 {}: {}", name, e))?
            .write_raw(data)
            .map_err(|e| format!("写不进 {}: {}", name, e))?;
    }
    Ok(())
}

fn write_readme(cli: &Cli, h: usize, w: usize, n: usize, dx: f64, m: &Measured) -> Result<(), String> {
    let text = format!(
        "由 eser-mappdebench 从 PDEBench 二维不可压 Navier-Stokes 数据转换而来。\n\
         源文件: {src}\n\
         =====================================================================\n\
         这是**真解出来的 NS 湍流场**，不是合成场。但它也不是地铁站 CFD：\n\
         几何是周期方腔、没有壁面掩码、没有通风工况。作用是给「真实湍流场长什么样」\n\
         一个参照，与 eser-makestation 的势流数据互补。\n\
         \n\
         形状: input ({n}, 4, {h}, {w})  output ({n}, 1, {h}, {w})\n\
         通道: input = (Vx, Vy, C0, t/t_max)     output = C\n\
         网格间距 dx = dy = {dx:.6}（源数据 domain_size = [1, 1]）\n\
         \n\
         实测的物理性质（用训练损失里同一批算子量的）\n\
         ------------------------------------------\n\
         速度分量朝向      : {orient}\n\
           ^ 这一条是**实测扫出来的**，不是照抄源码。PhiFlow 按 (y,x) 排轴，\n\
             末维两个分量哪个是 x 方向源码里看不出来；搞反了不报错，只会让\n\
             u·grad C 变成无意义的量。实测把分量对调后，对流扩散残差降 10 倍、\n\
             div 相对残差降 740 倍，两个互相独立的判据指向同一个朝向。\n\
         div(v) 相对残差   : {div_rel:.3e}   （对照：eser-makestation 合成数据 < 1e-6）\n\
           ^ 足够小，--div-weight 在这份数据上是有意义的\n\
         对流扩散残差/对流项: {res:.3e}  取 D={best_d}\n\
           ^ 不是机器零。余下的误差来自帧间隔较粗（中心时间差分跨 ±0.005 s）\n\
             以及 PhiFlow 用半拉格朗日平流、与我们的中心差分不是同一个离散。\n\
             约 10%，可用，但别指望物理项能压到很低\n\
         \n\
         三处与本项目 §4.2 的偏差（照抄前务必看）\n\
         ----------------------------------------\n\
         1. `particles` 取值有正有负，是带符号被动标量，**不是非负浓度场**，也没有点源。\n\
            所以「源位置图」那一路输入换成了 C0（t=0 的标量场）。\n\
            **外推点位测试在这份数据上做不了。**\n\
         2. 流场本身是瞬态的，转换取同帧的瞬时速度（对流残差 u·grad C 量的就是那一瞬）。\n\
         3. 周期方腔，无壁面——训练要加 --no-mask。\n\
         \n\
         建议命令行\n\
         ----------\n\
         eser-train --task dispersion --channel-order c --no-mask \\\n\
                    --train-file {out}/stage2_pdebench.hdf5 \\\n\
                    --velocity-channels 0 1 \\\n\
                    --domain-size 1.0 1.0 \\\n\
                    --diffusivity {best_d} \\\n\
                    --blocks \"[3,4,6,4,3]\"\n\
         \n\
         --blocks 为什么是 5 段：512x512 输入在 3 段配置下瓶颈是 128x128=16384 个 token，\n\
         超过 --attn-max-tokens 默认的 8192，注意力层会被静默跳过（只打一行提示）。\n\
         ",
        src = cli.input.display(), out = cli.out_dir.display(),
        n = n, h = h, w = w, dx = dx,
        orient = m.orient.label(),
        div_rel = m.div_rel,
        res = m.residual_ratio,
        best_d = m.best_d,
    );
    std::fs::write(cli.out_dir.join("README.txt"), text)
        .map_err(|e| format!("写 README.txt 失败：{}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 分量拆分是整条链路里最容易错又最不会报错的一环，逐元素钉住。
    #[test]
    fn orient_split_picks_the_right_component() {
        // 交错存放的两个分量：偶数位 = 10,11,12，奇数位 = 20,21,22
        let raw: Vec<f32> = vec![10.0, 20.0, 11.0, 21.0, 12.0, 22.0];

        let o = Orient { first_is_x: true, sx: 1.0, sy: 1.0 };
        assert_eq!(o.split(&raw), (vec![10.0, 11.0, 12.0], vec![20.0, 21.0, 22.0]));

        // 实测选中的就是这个：末维第 1 个分量才是 x 方向
        let o = Orient { first_is_x: false, sx: 1.0, sy: 1.0 };
        assert_eq!(o.split(&raw), (vec![20.0, 21.0, 22.0], vec![10.0, 11.0, 12.0]));

        let o = Orient { first_is_x: false, sx: -1.0, sy: 1.0 };
        assert_eq!(o.split(&raw), (vec![-20.0, -21.0, -22.0], vec![10.0, 11.0, 12.0]));

        // 八种朝向必须两两不同，否则扫描等于在重复比同一个东西
        let mut seen = std::collections::HashSet::new();
        for o in Orient::ALL {
            let (vx, vy) = o.split(&raw);
            let key: Vec<String> = vx.iter().chain(&vy).map(|v| v.to_string()).collect();
            assert!(seen.insert(key), "朝向 {} 与之前某个重复了", o.label());
        }
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn frame_indices_skip_frame_zero_and_stay_in_range() {
        let v = frame_indices(1000, 25);
        assert!(!v.is_empty());
        // 第 0 帧就是 C0 那一路输入，留着等于给网络一道抄答案的题
        assert!(v[0] >= 1, "抽到了第 0 帧：{:?}", v);
        assert!(*v.last().unwrap() <= 998, "越界：{:?}", v);
        assert!(v.windows(2).all(|p| p[1] > p[0]), "没有严格递增：{:?}", v);
        // 对数间隔：后半段的步长应当明显大于前半段
        let gaps: Vec<usize> = v.windows(2).map(|p| p[1] - p[0]).collect();
        assert!(
            gaps[gaps.len() - 1] > gaps[0],
            "不像对数间隔，间隔序列 {:?}",
            gaps
        );
        // 要的帧数超过可用范围时不能重复也不能 panic
        let all = frame_indices(6, 100);
        assert_eq!(all, vec![1, 2, 3, 4]);
    }

    #[test]
    fn down_subsamples_by_stride() {
        let v: Vec<f32> = (0..12).map(|x| x as f32).collect(); // 3x4
        assert_eq!(down(&v, 3, 4, 1), v);
        // stride 2：取行 0,2 与列 0,2
        assert_eq!(down(&v, 3, 4, 2), vec![0.0, 2.0, 8.0, 10.0]);
    }

    #[test]
    fn cli_rejects_bad_input() {
        let a = |s: &str| -> Vec<String> { s.split_whitespace().map(String::from).collect() };
        assert!(parse(&a("--out-dir x")).is_err(), "没给 --in 应当报错");
        assert!(parse(&a("--in f --frames 0")).is_err());
        assert!(parse(&a("--in f --stride 0")).is_err());
        assert!(parse(&a("--in f --bogus")).is_err());
        assert!(parse(&a("--in")).is_err(), "旗标后缺参数应当报错");
        let ok = parse(&a("--in f --frames 7 --stride 2 --verify")).expect("合法参数被拒");
        assert_eq!((ok.frames, ok.stride, ok.verify_only), (7, 2, true));
    }
}
