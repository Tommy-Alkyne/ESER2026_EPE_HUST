//! 按《地铁站污染物扩散代理模型·实施方案》§4 生成两阶段训练数据。
//!
//! ```text
//! eser-makestation --out-dir dataset/station --grid 128
//! ```
//!
//! # 这份数据是什么，不是什么
//!
//! **不是 CFD。** 没有湍流模型、没有浮力、没有真实站体几何，流动是内部涡量驱动的
//! 回流而不是贯穿流。它唯一的长处是：产出的场**在损失实际用的那个离散算子下**满足
//! §3.5 保留的连续性方程与 §3.2 的对流扩散方程（`tests/synth_physics.rs` 用
//! `physics::divergence` / `transport::advection_diffusion_residual` 实测钉住）。
//!
//! 所以它能干什么：打通管线、让物理残差项在训练中真的起作用、给一个「数据长这样时
//! 模型能学成什么样」的预演。不能干什么：当 CFD 的替代品，或当任何校核依据。
//!
//! # 产物
//!
//! ```text
//! <out-dir>/
//!   station_mask.npy    §4.5 的 0/1 栅格
//!   stage1.hdf5         §4.1  input (N,3,H,W) / output (N,4,H,W) = (P,Vx,Vy,T)
//!   stage2.hdf5         §4.2  input (N,4,H,W) = (Vx,Vy,源位置图,t) / output (N,1,H,W) = C
//!   README.txt          物性参数与建议的训练命令行（填错了残差就没有参考意义）
//! ```

use std::path::{Path, PathBuf};

use eser::geometry;
use eser::synth::{
    advance, log_times, solve_flow, source_map, source_positions, stable_dt, station_mask,
    FlowCase, Scenario, N_TRAIN_SOURCES,
};
use rust_hdf5::H5File;

const USAGE: &str = "\
eser-makestation —— 按方案 §4 生成两阶段合成数据（不是 CFD，见 --help 说明）

  --out-dir <目录>     默认 dataset/station
  --grid <H> <W>       网格，默认 128 256（长条形站体，与 60m x 12m 的长宽比相称）
  --vent-levels <N>    通风量档数，默认 5（§4.1 是 5）
  --sources <N>        源位置个数，默认 10（前 7 个训练，后 3 个留作外推测试）
  --frames <N>         每个瞬态算例抽多少帧，默认 25（§4.3 建议 20~30，对数间隔）
  --t-end <秒>         瞬态终止时刻，默认 320
  --diffusivity <D>    湍流扩散系数 m²/s，默认 0.5（**不是**分子扩散的 1e-5，见文档）
  --domain-size <Ly> <Lx>  物理尺寸（米），默认 12 60
  --quick              小规模冒烟：网格 64x128、通风量 2 档、源 3 个、每例 6 帧
";

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", USAGE);
        return;
    }
    if let Err(e) = run(&argv) {
        eprintln!("[eser-makestation] {}", e);
        std::process::exit(1);
    }
}

struct Cfg {
    out_dir: PathBuf,
    h: usize,
    w: usize,
    vent_levels: usize,
    sources: usize,
    frames: usize,
    t_end: f64,
    sc: Scenario,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("dataset/station"),
            h: 128,
            w: 256,
            vent_levels: 5,
            sources: 10,
            frames: 25,
            t_end: 320.0,
            sc: Scenario::default(),
        }
    }
}

fn parse(argv: &[String]) -> Result<Cfg, String> {
    let mut c = Cfg::default();
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].clone();
        let next = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            argv.get(*i)
                .cloned()
                .ok_or_else(|| format!("{} 后面缺参数", flag))
        };
        match argv[i].as_str() {
            "--out-dir" => c.out_dir = PathBuf::from(next(&mut i)?),
            "--grid" => {
                c.h = num(&next(&mut i)?, "--grid H")?;
                c.w = num(&next(&mut i)?, "--grid W")?;
            }
            "--vent-levels" => c.vent_levels = num(&next(&mut i)?, "--vent-levels")?,
            "--sources" => c.sources = num(&next(&mut i)?, "--sources")?,
            "--frames" => c.frames = num(&next(&mut i)?, "--frames")?,
            "--t-end" => c.t_end = fnum(&next(&mut i)?, "--t-end")?,
            "--diffusivity" => {
                c.sc.diffusivity = fnum(&next(&mut i)?, "--diffusivity")?;
                c.sc.thermal_diffusivity = c.sc.diffusivity;
            }
            "--domain-size" => {
                let ly = fnum(&next(&mut i)?, "--domain-size Ly")?;
                let lx = fnum(&next(&mut i)?, "--domain-size Lx")?;
                c.sc.domain_size = (ly, lx);
            }
            "--quick" => {
                c.h = 64;
                c.w = 128;
                c.vent_levels = 2;
                c.sources = 3;
                c.frames = 6;
            }
            other => return Err(format!("不认识的参数 {:?}\n\n{}", other, USAGE)),
        }
        i += 1;
    }
    if c.sources > source_positions().len() {
        return Err(format!(
            "--sources 最多 {}（点位表里就这么多），实得 {}",
            source_positions().len(),
            c.sources
        ));
    }
    for (name, v) in [("--grid H", c.h), ("--grid W", c.w)] {
        if v < 16 {
            return Err(format!("{} 至少 16，实得 {}", name, v));
        }
    }
    if c.vent_levels == 0 || c.sources == 0 || c.frames == 0 {
        return Err("--vent-levels / --sources / --frames 都必须大于 0".into());
    }
    if c.t_end <= 0.0 {
        return Err(format!("--t-end 必须为正，实得 {}", c.t_end));
    }
    Ok(c)
}

fn num(s: &str, what: &str) -> Result<usize, String> {
    s.parse().map_err(|e| format!("{} 不是正整数：{}", what, e))
}
fn fnum(s: &str, what: &str) -> Result<f64, String> {
    s.parse().map_err(|e| format!("{} 不是数字：{}", what, e))
}

// ================================================================ 主流程

fn run(argv: &[String]) -> Result<(), String> {
    let cfg = parse(argv)?;
    let (h, w) = (cfg.h, cfg.w);
    let (dx, dy) = cfg.sc.spacing(h, w);
    std::fs::create_dir_all(&cfg.out_dir).map_err(|e| format!("建不了输出目录：{}", e))?;

    let mask = station_mask(h, w)?;
    let cov = geometry::coverage(&mask);
    println!(
        "[makestation] 网格 {}x{}  域 {:.1}m x {:.1}m  dx={:.4} dy={:.4}  覆盖率 {:.1}%",
        h, w, cfg.sc.domain_size.0, cfg.sc.domain_size.1, dx, dy, cov * 100.0
    );
    write_npy_mask(&cfg.out_dir.join("station_mask.npy"), h, w, &mask)?;

    // ---- 阶段一：工况 -> (P, Vx, Vy, T) ----
    let cases = flow_cases(cfg.vent_levels);
    println!("[makestation] 阶段一：{} 个稳态工况", cases.len());
    let mut s1_in = Vec::with_capacity(cases.len() * 3 * h * w);
    let mut s1_out = Vec::with_capacity(cases.len() * 4 * h * w);
    let mut flows = Vec::with_capacity(cases.len());
    let mut worst_resid = 0.0f64;
    for (n, case) in cases.iter().enumerate() {
        let sol = solve_flow(&mask, h, w, case, &cfg.sc, 8000)?;
        worst_resid = worst_resid.max(sol.residual);
        for v in case.as_channels() {
            s1_in.extend(std::iter::repeat_n(v, h * w));
        }
        for ch in [&sol.p, &sol.vx, &sol.vy, &sol.t] {
            s1_out.extend(ch.iter().map(|&x| x as f32));
        }
        flows.push(sol);
        if (n + 1) % 10 == 0 || n + 1 == cases.len() {
            println!("[makestation]   {}/{}", n + 1, cases.len());
        }
    }
    // SOR 没收敛的话流场本身就不是 ∇²ψ=-ω 的解，涡量结构会被抹平；div=0 仍然成立
    // （那是构造性质），所以这个问题不会被物理残差抓到，必须在这里说
    println!("[makestation] 阶段一 SOR 最大残差 {:.2e}", worst_resid);
    if worst_resid > 1e-6 {
        println!(
            "[makestation] 警告：SOR 残差偏大，流场可能没收敛到 ∇²ψ=-ω 的解。\n\
             注意 div(v)=0 是流函数的构造性质，即便没收敛也成立——物理残差抓不到这个问题"
        );
    }
    write_h5(
        &cfg.out_dir.join("stage1.hdf5"),
        &[cases.len(), 3, h, w],
        &s1_in,
        &[cases.len(), 4, h, w],
        &s1_out,
    )?;
    println!(
        "[makestation] stage1.hdf5  input {:?}  output {:?} = (P, Vx, Vy, T)",
        [cases.len(), 3, h, w],
        [cases.len(), 4, h, w]
    );

    // ---- 阶段二：(Vx, Vy, 源位置图, t) -> C ----
    let positions = source_positions();
    let used = &positions[..cfg.sources];
    let times = log_times(cfg.t_end / 64.0, cfg.t_end, cfg.frames);
    let n2 = cases.len() * cfg.sources * cfg.frames;
    println!(
        "[makestation] 阶段二：{} 工况 x {} 源位置 x {} 帧 = {} 帧；时刻 {:.1}..{:.1}s",
        cases.len(), cfg.sources, cfg.frames, n2, times[0], times[times.len() - 1]
    );
    let mut s2_in: Vec<f32> = Vec::with_capacity(n2 * 4 * h * w);
    let mut s2_out: Vec<f32> = Vec::with_capacity(n2 * h * w);
    // 时间通道归一化到 [0,1]：直接喂 320 会和 [0,1] 量级的其它通道差两个数量级
    let t_norm = |t: f64| (t / cfg.t_end) as f32;

    for (ci, sol) in flows.iter().enumerate() {
        let dt = stable_dt(&sol.vx, &sol.vy, dx, dy, cfg.sc.diffusivity);
        for (si, &(cy, cx)) in used.iter().enumerate() {
            let src = source_map(h, w, cy, cx, 0.035, &mask);
            let mut c = vec![0.0f64; h * w];
            let mut t_prev = 0.0;
            for &t in &times {
                let (next, _) = advance(
                    &c, &sol.vx, &sol.vy, &mask, &src, h, w, dx, dy,
                    cfg.sc.diffusivity, dt, t - t_prev,
                );
                c = next;
                t_prev = t;
                s2_in.extend(sol.vx.iter().map(|&x| x as f32));
                s2_in.extend(sol.vy.iter().map(|&x| x as f32));
                s2_in.extend(src.iter().map(|&x| x as f32));
                s2_in.extend(std::iter::repeat_n(t_norm(t), h * w));
                s2_out.extend(c.iter().map(|&x| x as f32));
            }
            let done = ci * cfg.sources + si + 1;
            if done % 5 == 0 || done == cases.len() * cfg.sources {
                println!("[makestation]   {}/{} 个瞬态算例", done, cases.len() * cfg.sources);
            }
        }
    }
    write_h5(
        &cfg.out_dir.join("stage2.hdf5"),
        &[n2, 4, h, w],
        &s2_in,
        &[n2, 1, h, w],
        &s2_out,
    )?;
    println!(
        "[makestation] stage2.hdf5  input {:?} = (Vx, Vy, 源位置图, t)  output {:?} = C",
        [n2, 4, h, w],
        [n2, 1, h, w]
    );

    write_readme(&cfg, cov, worst_resid)?;
    println!("[makestation] 完成 -> {}", cfg.out_dir.display());
    Ok(())
}

/// §4.1 的工况组合：通风量 N 档 × 屏蔽门 2 态 × 列车工况 4 种。
fn flow_cases(vent_levels: usize) -> Vec<FlowCase> {
    let mut v = Vec::new();
    for i in 0..vent_levels {
        // 从 0.3 到 1.0，避开 0（全静止的场没有可学的结构，还会让 R² 的分母塌掉）
        let vent = if vent_levels == 1 {
            1.0
        } else {
            0.3 + 0.7 * i as f64 / (vent_levels - 1) as f64
        };
        for psd_open in [true, false] {
            for train_phase in 0..4u8 {
                v.push(FlowCase { vent, psd_open, train_phase });
            }
        }
    }
    v
}

// ================================================================ 落盘

fn write_h5(
    path: &Path,
    in_dims: &[usize],
    inp: &[f32],
    out_dims: &[usize],
    out: &[f32],
) -> Result<(), String> {
    let expect = |d: &[usize]| d.iter().product::<usize>();
    if inp.len() != expect(in_dims) || out.len() != expect(out_dims) {
        return Err(format!(
            "元素数与形状不符：input {} vs {:?}，output {} vs {:?}",
            inp.len(), in_dims, out.len(), out_dims
        ));
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

/// 写 §4.5 要的 0/1 栅格。格式与 `predict::read_npy_2d` 读的一致（`.npy` v1.0）。
fn write_npy_mask(path: &Path, h: usize, w: usize, mask: &[f32]) -> Result<(), String> {
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        h, w
    );
    let header_len = dict.len() + 1;
    let pad = 64 - ((8 + 2 + header_len) % 64);
    let mut buf = Vec::with_capacity(10 + header_len + pad + mask.len() * 4);
    buf.extend_from_slice(b"\x93NUMPY\x01\x00");
    buf.extend_from_slice(&((header_len + pad) as u16).to_le_bytes());
    buf.extend_from_slice(dict.as_bytes());
    buf.resize(buf.len() + pad, b' ');
    buf.push(b'\n');
    for v in mask {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, buf).map_err(|e| format!("写不了 {:?}: {}", path, e))
}

/// 把「填错了残差就没意义」的那几个参数连同建议命令行一起落盘。
///
/// §4.5 说得很清楚：ρ / ν / D 和计算域物理尺寸没法从数据里推断。数据和参数分开放，
/// 过两周就对不上了。
fn write_readme(cfg: &Cfg, coverage: f64, residual: f64) -> Result<(), String> {
    let (dx, dy) = cfg.sc.spacing(cfg.h, cfg.w);
    let (ly, lx) = cfg.sc.domain_size;
    let text = format!(
        "本目录由 eser-makestation 生成，**不是 CFD 数据**。\n\
         =====================================================================\n\
         没有湍流模型、没有浮力、没有真实站体几何；流动是内部涡量驱动的回流，\n\
         不是贯穿流。唯一的长处：产出的场在训练损失实际用的那个离散算子下满足\n\
         连续性方程 div(v)=0 与对流扩散方程。可用于打通管线、验证物理残差项、\n\
         预演模型能学成什么样；不可用作 CFD 替代品或任何校核依据。\n\
         \n\
         网格          : {h} x {w}\n\
         计算域        : Ly={ly} m, Lx={lx} m  (dx={dx:.5}, dy={dy:.5})\n\
         计算域覆盖率  : {cov:.2}%\n\
         空气密度 rho  : {rho}\n\
         扩散系数 D    : {d}   <- 湍流量级，不是分子扩散的 1e-5\n\
         参考压力      : {p} Pa\n\
         参考温度      : {t} C\n\
         阶段一 SOR 最大残差: {res:.3e}\n\
         \n\
         通道语义\n\
         --------\n\
         stage1  input  (N,3,H,W) = (通风量, 屏蔽门开=1, 列车工况/3)  各自广播成常数图\n\
         stage1  output (N,4,H,W) = (P, Vx, Vy, T)      -> --channel-order pvvt\n\
         stage2  input  (N,4,H,W) = (Vx, Vy, 源位置图, t/t_end)\n\
         stage2  output (N,1,H,W) = C                   -> --channel-order c\n\
         \n\
         源位置：前 {ntrain} 个用于训练，其余留作 §7.1 的外推点位测试。\n\
         源强恒为 1（§3.3）：方程对 C 线性，用的时候乘系数即可。\n\
         \n\
         建议的训练命令行（物性参数填错，物理残差的绝对数值就没有参考意义）\n\
         ----------------------------------------------------------------\n\
         # 阶段一：只保留连续性方程，去掉 N-S 动量项（§3.5）\n\
         eser-train --task flow --channel-order pvvt --ns-weight 0 \\\n\
                    --train-file {dir}/stage1.hdf5 \\\n\
                    --domain-mask {dir}/station_mask.npy \\\n\
                    --domain-size {ly} {lx} --rho {rho} \\\n\
                    --blocks \"[3,4,6,4,3]\"\n\
         \n\
         # 阶段二：对流扩散\n\
         eser-train --task dispersion --channel-order c \\\n\
                    --train-file {dir}/stage2.hdf5 \\\n\
                    --diffusivity {d} --velocity-channels 0 1 \\\n\
                    --domain-mask {dir}/station_mask.npy \\\n\
                    --domain-size {ly} {lx} \\\n\
                    --blocks \"[3,4,6,4,3]\"\n\
         \n\
         --blocks 为什么是 5 段：编码器要多降一次采样，否则 512x512 输入在瓶颈处是\n\
         128x128=16384 个 token，超过 --attn-max-tokens 默认的 8192，注意力层会被\n\
         静默跳过（只打一行提示）。见方案 §5。\n",
        h = cfg.h, w = cfg.w, ly = ly, lx = lx, dx = dx, dy = dy,
        cov = coverage * 100.0, rho = cfg.sc.rho, d = cfg.sc.diffusivity,
        p = cfg.sc.p_ref, t = cfg.sc.t_ref, res = residual,
        ntrain = N_TRAIN_SOURCES.min(cfg.sources),
        dir = cfg.out_dir.display(),
    );
    std::fs::write(cfg.out_dir.join("README.txt"), text)
        .map_err(|e| format!("写 README.txt 失败：{}", e))
}
