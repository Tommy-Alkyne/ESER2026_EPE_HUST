//! Rust 版离线推理入口，对应 Python 的 `surrogate_predict.py`。
//!
//! ```text
//! eser-predict --ckpt-file experiments/run01/checkpoints/best \
//!              --data-file dataset/new_case.hdf5 \
//!              --out-file  experiments/run01/pred.hdf5
//! ```
//!
//! # 三处「不这么做就会静默出错」的地方
//!
//! 1. **归一化统计量必须来自 checkpoint，不能拿新数据重算。** 反归一化用的是训练时的
//!    逐通道极值；拿待推理数据自己的极值去反归一化，输出会被映射到另一套标度，图看着
//!    正常、数全错。所以这里把 `CheckpointMeta` 里的 `stats` 注入 `DatasetOptions`。
//! 2. **几何来自 checkpoint。** Python 版曾经因为超参白名单漏收 `t_geometry`，推理时
//!    拿默认 T 型几何（覆盖 45%）去裁剪车站模型（25%）的输出——这是方案文档记下的三个
//!    历史缺陷之一。`--domain-mask` 只在显式给出时才覆盖。
//! 3. **网络结构来自 checkpoint。** `blocks` / `growth_rate` / `num_heads` 等对不上时，
//!    权重加载会直接失败而不是凑合跑——这正是要的行为。

use std::path::{Path, PathBuf};

use burn::backend::NdArray;

use eser::data::{load_3d, CfdDataset, DatasetOptions, Options3d};
use eser::model::DenseEd;
use eser::model3d::DenseEd3d;
use eser::predict::{
    self, default_out_file, merge_ckpt_hp_strings, resolve_mask, Hyperparams, OutputMeta,
    PredictOptions,
};
use eser::train::{self, CheckpointMeta};

/// 推理走**非 autodiff** 后端：BatchNorm 与 Dropout 由后端决定走哪条分支，
/// 这里正需要 eval 那条。
type B = NdArray<f32>;

const USAGE: &str = "\
eser-predict —— 离线推理（对应 surrogate_predict.py）

必给：
  --ckpt-file <路径>     checkpoint 前缀，如 .../checkpoints/best
                         （实际读的是 <前缀>.mpk 与 <前缀>.meta）
  --data-file <路径>     待推理的 HDF5

可选：
  --out-file <路径>      默认 <data-file 同目录>/<名字>.pred.hdf5
  --batch-size <N>       默认 64
  --limit <N>            只推前 N 个样本
  --no-denorm            输出归一化空间的值，不还原量纲
  --no-mask              不裁剪计算域
  --domain-mask <路径>   覆盖 checkpoint 里的几何（.npy 栅格 / .json 矩形段 / 内联 JSON）
  --ndim <2|3>           覆盖自动判定
";

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", USAGE);
        return;
    }
    if let Err(e) = run(&argv) {
        eprintln!("[eser-predict] {}", e);
        std::process::exit(1);
    }
}

/// 命令行。手写而不是引 clap：整个项目的解析风格统一在 `cli.rs` 那套，
/// 为一个二进制多一个依赖不划算。
struct Cli {
    ckpt_file: PathBuf,
    data_file: PathBuf,
    out_file: Option<PathBuf>,
    ndim: Option<usize>,
    domain_mask: Option<String>,
    opts: PredictOptions,
}

fn parse(argv: &[String]) -> Result<Cli, String> {
    let mut ckpt_file: Option<PathBuf> = None;
    let mut data_file: Option<PathBuf> = None;
    let mut out_file = None;
    let mut ndim = None;
    let mut domain_mask = None;
    let mut opts = PredictOptions::default();

    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        // 取下一个位置的值；缺了就报清楚是哪个旗标缺参数
        let mut val = || -> Result<String, String> {
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| format!("{} 后面缺参数", flag))
        };
        match flag {
            "--ckpt-file" => ckpt_file = Some(PathBuf::from(val()?)),
            "--data-file" => data_file = Some(PathBuf::from(val()?)),
            "--out-file" => out_file = Some(PathBuf::from(val()?)),
            "--domain-mask" | "--t-geometry" => domain_mask = Some(val()?),
            "--batch-size" => {
                opts.batch_size = val()?
                    .parse()
                    .map_err(|e| format!("--batch-size 不是正整数：{}", e))?;
                if opts.batch_size == 0 {
                    return Err("--batch-size 必须大于 0".into());
                }
            }
            "--limit" => {
                opts.limit = Some(
                    val()?
                        .parse()
                        .map_err(|e| format!("--limit 不是正整数：{}", e))?,
                )
            }
            "--ndim" => {
                let v: usize = val()?
                    .parse()
                    .map_err(|e| format!("--ndim 不是数字：{}", e))?;
                if v != 2 && v != 3 {
                    return Err(format!("--ndim 只能是 2 或 3，实得 {}", v));
                }
                ndim = Some(v);
            }
            "--no-denorm" => opts.denormalize = false,
            "--no-mask" => opts.apply_mask = false,
            // 未知旗标一律报错：拼错的旗标被静默忽略，等于按默认值跑了一遍还不知道
            other => return Err(format!("不认识的参数 {:?}\n\n{}", other, USAGE)),
        }
        i += 1;
    }
    opts.t_geometry = domain_mask.clone();
    Ok(Cli {
        ckpt_file: ckpt_file.ok_or("必须给 --ckpt-file")?,
        data_file: data_file.ok_or("必须给 --data-file")?,
        out_file,
        ndim,
        domain_mask,
        opts,
    })
}

fn run(argv: &[String]) -> Result<(), String> {
    let cli = parse(argv)?;
    let device = Default::default();

    // ---- checkpoint 的元信息（权重晚一步读，先要知道该建多大的壳）----
    let meta = read_meta(&cli.ckpt_file)?;
    let mut hp = Hyperparams::default();
    let dropped = merge_ckpt_hp_strings(&mut hp, &meta.hyper)?;
    // nic / noc / ndim 不属于 Hyperparams，但**下面就要用**（建壳的通道数、走 2D 还是 3D），
    // 不能跟着 dropped 一起报成「未参与建模」——那会把一条正常的元信息说成疑似缺陷
    let consumed = ["nic", "noc", "ndim"];
    let unused: Vec<&String> = dropped
        .iter()
        .filter(|k| !consumed.contains(&k.as_str()))
        .collect();
    if !unused.is_empty() {
        // 真没人用的键要说出来：将来加了超参却忘了收录，这行是唯一线索
        println!("[eser-predict] checkpoint 里有超参未参与建模：{:?}", unused);
    }
    let (nic, noc) = channels_from_meta(&meta)?;
    let ckpt_ndim = meta.hp("ndim").and_then(|v| v.parse::<usize>().ok());
    let ndim = predict::detect_ndim(&cli.data_file, ckpt_ndim, cli.ndim)?;

    let stats = meta.stats().ok_or(
        "checkpoint 里没有完整的归一化统计量，无法还原量纲。\
         这个 checkpoint 可能来自更早的格式版本，需要重新训练",
    )?;
    let mask = resolve_mask(&hp, cli.domain_mask.as_deref(), cli.opts.apply_mask)?;
    println!(
        "[eser-predict] ndim={} 通道 {}->{} channel_order={} 掩码={:?}",
        ndim, nic, noc, hp.channel_order, mask
    );

    let pred = if ndim == 3 {
        let opts3 = Options3d {
            stats: Some(stats),
            ..Default::default()
        };
        let (x, y, ds_stats) = load_3d::<B>(&cli.data_file, &opts3, &device)?;
        let dims = y.dims();
        let shell: DenseEd3d<B> = hp
            .model_config_3d(nic, noc, [dims[2], dims[3], dims[4]])
            .init(&device)?;
        let (model, _) = train::load_checkpoint::<B, _>(shell, &cli.ckpt_file)?;
        predict::predict_3d(&model, x, &ds_stats, &hp, &cli.opts)?
    } else {
        let dopts = DatasetOptions {
            stats: Some(stats),
            ..Default::default()
        };
        let ds = CfdDataset::open(&cli.data_file, &dopts)?;
        let ys = ds.y_shape();
        let shell: DenseEd<B> = hp
            .model_config_2d(nic, noc, [ys[2], ys[3]])
            .init(&device)?;
        let (model, _) = train::load_checkpoint::<B, _>(shell, &cli.ckpt_file)?;
        predict::predict_2d(&model, &ds, &hp, &cli.opts, &device)?
    };

    let out = cli
        .out_file
        .unwrap_or_else(|| default_out_file(&cli.data_file));
    let ometa = OutputMeta::new(
        &hp,
        &cli.data_file.to_string_lossy(),
        &cli.ckpt_file.to_string_lossy(),
        ndim,
        nic,
        noc,
    );
    predict::write_prediction_hdf5(&out, &pred, &ometa)?;

    println!(
        "[eser-predict] {} 个样本 × {} 通道 -> {}（{}）",
        pred.n_samples(),
        pred.n_channels(),
        out.display(),
        if pred.denormalized {
            "有量纲"
        } else {
            "归一化空间"
        }
    );
    if let Some(c) = pred.coverage {
        println!("[eser-predict] 计算域覆盖率 {:.2}%", c * 100.0);
    }
    Ok(())
}

/// 读 `<前缀>.meta`。
///
/// 单独读一遍而不是走 `load_checkpoint`，是因为**得先知道通道数和 ndim 才能建出
/// 形状正确的空壳模型**，而 `load_checkpoint` 需要那个空壳当参数。
fn read_meta(prefix: &Path) -> Result<CheckpointMeta, String> {
    let name = prefix
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .ok_or_else(|| format!("{:?} 不像一个 checkpoint 前缀", prefix))?;
    let meta_path = prefix.with_file_name(format!("{}.meta", name));
    let mpk = prefix.with_file_name(format!("{}.mpk", name));
    if !mpk.is_file() {
        return Err(format!(
            "找不到权重 {}。--ckpt-file 要给的是**前缀**（如 .../checkpoints/best），不带扩展名",
            mpk.display()
        ));
    }
    let text = std::fs::read_to_string(&meta_path)
        .map_err(|e| format!("读不了 {}：{}", meta_path.display(), e))?;
    CheckpointMeta::from_text(&text)
}

/// 从元信息里取输入/输出通道数。
///
/// 这两个是 `fit` 从数据形状推出来后写进 checkpoint 的，不是用户给的，所以缺了就是
/// checkpoint 有问题——**不猜**：猜错了模型建出来结构不同，权重加载会报一堆形状错，
/// 排查起来比这里直接说「元信息缺 nic」远远费事。
fn channels_from_meta(meta: &CheckpointMeta) -> Result<(usize, usize), String> {
    let get = |k: &str| -> Result<usize, String> {
        meta.hp(k)
            .ok_or_else(|| format!("checkpoint 元信息里没有 {}，无法重建同构的网络", k))?
            .parse::<usize>()
            .map_err(|e| format!("checkpoint 里的 {} 不是整数：{}", k, e))
    };
    Ok((get("nic")?, get("noc")?))
}
