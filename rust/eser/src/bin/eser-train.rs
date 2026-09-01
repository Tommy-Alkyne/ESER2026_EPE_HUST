//! Rust 版训练入口，对应 Python 的 `train.py`。
//!
//! ```text
//! eser-train --train-file dataset/p_v_256.hdf5 --epochs 200 --batch-size 16
//! ```
//!
//! **旗标名与 `args.py` 逐字一致**，所以 GUI 的命令拼装不用改，只换可执行文件。
//!
//! # 与 Python 版行为一致的几处
//!
//! * 训练集与测试集是同一个文件时，按 `--val-split` 切出互不重叠的验证集；
//!   测试集**复用训练集的归一化统计量**，两边标度一致，R² 才有可比性
//! * 物理损失在**有量纲**场上算（`--physics-in-physical-units`）
//! * 每轮把指标落成 `.txt`——`diagnose` 只认这些文件，不写的话诊断报告整列打 `--`
//! * `<run_dir>/args.txt` 落一份参数快照，诊断侧据此做配置体检

use std::path::{Path, PathBuf};

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

use eser::boundary::{Boundary, BoundaryZone};
use eser::cli::{self, Args};
use eser::data::{load_train_val, CfdDataset, DatasetOptions};
use eser::geometry;
use eser::model::{DenseEd, DenseEdConfig};
use eser::model3d::{DenseEd3d, DenseEd3dConfig};
use eser::physics::{ChannelOrder, PhysicsLoss, Scales, Task};
use eser::predict::GeometrySpec;
use eser::train::{self, FitConfig, ReduceLrOnPlateau};

/// CPU + autodiff。CUDA 后端由 `--features cuda` 打开，那是另一条编译路径。
type B = Autodiff<NdArray<f32>>;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", USAGE);
        return;
    }
    if let Err(e) = run(&argv) {
        // 出错走 stderr 并给非零退出码：GUI 靠退出码判断这一轮成没成
        eprintln!("[eser-train] {}", e);
        std::process::exit(1);
    }
}

const USAGE: &str = "\
eser-train —— Rust 版训练入口（旗标与 args.py 一致）

常用：
  --train-file <路径>       训练数据 HDF5
  --test-file <路径>        独立测试集；不给就按 --val-split 从训练集切
  --epochs / --batch-size / --lr / --weight-decay / --grad-clip / --seed
  --task flow|dispersion    方程组
  --channel-order pvv|vvp|pvvt|c
  --domain-size <Ly> <Lx>   物理域尺寸，决定 dx/dy
  --no-mask                 不用计算域掩码
  --domain-mask <路径>      .npy 栅格或 .json 矩形段（= --t-geometry）
                            栅格支持分区标签：0 域外 1 流体 2 壁 3 入口 4 出口
  --bc-weight <w>           边界条件残差权重（缺省 0.05）
  --inlet-speed <q>         送风口风速 m/s，正值 = 送进域内。掩码里有入口就必须给
  --outlet-speed <q>        排风口风速 m/s，正值 = 从域内抽出。掩码里有出口就必须给
  --ndim 3                  走 3D 模型
  --domain-height <Lz>      三维物理层高，决定 dz（缺省 1.0）
  --velocity-channel-z <i>  dispersion 三维时竖向速度所在的输入通道
                            channel-order 用 pvvv / pvvvt 才有真三维流场物理
  --exp-dir / --exp-name    产物落到 <exp-dir>/<exp-name>
  --post                    只评估不训练：读 checkpoint 跑一次，落测试侧指标
  --ckpt-epoch <N>          配合 --post 指定 epoch<N>，缺省用 best
";

fn run(argv: &[String]) -> Result<(), String> {
    let args = cli::parse(argv)?;
    let device = Default::default();

    let train_path = resolve_train_file(&args)?;
    println!("[eser-train] 数据: {}", train_path.display());

    let run_dir = args.run_dir();
    std::fs::create_dir_all(&run_dir).map_err(|e| format!("建实验目录失败：{}", e))?;
    write_args_snapshot(&run_dir, &args, &train_path)?;

    // ---- 数据 ----
    let opts = DatasetOptions {
        input_key: args.input_key.clone(),
        prefer_sensor_input: args.prefer_sensor_input,
        ..Default::default()
    };
    let (train, val) = match &args.test_file {
        // 有独立测试集：**测试集复用训练集的统计量**，否则两边被映射到不同标度，R² 不可比
        Some(t) => {
            let train = CfdDataset::open(&train_path, &opts)?;
            let mut vopts = opts.clone();
            vopts.stats = Some(train.stats().clone());
            let vp = resolve_under(&args.data_dir, t);
            (train, Some(CfdDataset::open(&vp, &vopts)?))
        }
        None => load_train_val(&train_path, args.val_split, args.seed, &opts)?,
    };
    let stats = train.stats().clone();
    let y_shape = train.y_shape().to_vec();
    println!(
        "[eser-train] 训练 {} 样本，验证 {} 样本，输出形状 {:?}",
        train.len(),
        val.as_ref().map(|v| v.len()).unwrap_or(0),
        y_shape
    );
    for c in stats.degenerate_channels() {
        println!(
            "[eser-train] 警告：第 {} 个输出通道是常数场，归一化后恒为 0",
            c
        );
    }

    // ---- 网格与几何 ----
    let (grid_h, grid_w) = (y_shape[y_shape.len() - 2], y_shape[y_shape.len() - 1]);
    let (dx, dy) = args.spacing(grid_h, grid_w);
    println!(
        "[eser-train] 网格间距 dx={:.6} dy={:.6}（domain_size={:?}）",
        dx, dy, args.domain_size
    );

    // 三维的深度层数：数据形状是 (N, C, D, H, W)
    let grid_d = if args.ndim == 3 { y_shape[2] } else { 1 };
    let dz = args.spacing_z(grid_d);
    if args.ndim == 3 {
        println!(
            "[eser-train] 层数 {}，层间距 dz={:.6}（domain_height={}）",
            grid_d, dz, args.domain_height
        );
    }

    let (mask, boundary) = build_mask(&args, grid_d, grid_h, grid_w, &device)?;

    // ---- 损失 ----
    let order = ChannelOrder::parse(&args.channel_order)?;
    let task = if args.task == "dispersion" {
        Task::Dispersion
    } else {
        Task::Flow
    };
    // 物理项默认在有量纲场上算；关掉的话残差在归一化空间求，只有两个速度分量
    // 动态范围恰好相等时才等价
    let (y_min, y_max) = if args.physics_in_physical_units {
        (Some(stats.y_min.clone()), Some(stats.y_max.clone()))
    } else {
        (None, None)
    };
    let loss = PhysicsLoss::<B> {
        div_weight: args.div_weight,
        ns_weight: args.ns_weight,
        smooth_weight: args.smooth_weight,
        advdiff_weight: args.advdiff_weight,
        channel_order: order,
        task,
        rho: args.rho,
        nu: args.nu,
        diffusivity: args.diffusivity,
        dx,
        dy,
        dz,
        physics_warmup: args.physics_warmup,
        scales: Scales::default(),
        mask,
        y_min,
        y_max,
        boundary,
        bc_weight: args.bc_weight,
        inlet_speed: args.inlet_speed.unwrap_or(0.0),
        outlet_speed: args.outlet_speed.unwrap_or(0.0),
        ..Default::default()
    };

    let fit_cfg = FitConfig {
        epochs: args.epochs,
        batch_size: args.batch_size,
        test_batch_size: Some(args.test_batch_size),
        lr: args.lr,
        weight_decay: args.weight_decay as f32,
        grad_clip: args.grad_clip,
        scheduler: ReduceLrOnPlateau::new(args.lr_patience),
        seed: args.seed,
        ckpt_freq: args.ckpt_freq,
        ckpt_dir: Some(args.ckpt_dir()),
        keep_best: true,
        verbose: true,
        physics_scaling: args.physics_loss_scaling == "ema",
        velocity_channels: args.velocity_channels,
        velocity_channel_z: args.velocity_channel_z,
        hyper: checkpoint_hyper(&args),
    };

    let n_in = train.x_shape()[1];
    let n_out = y_shape[1];

    // ---- 仅评估（--post）----
    // 在建模之前分叉：post 走的是「读 checkpoint -> 评估一次 -> 落测试侧指标」，
    // 与训练循环没有共用的部分，混在一起只会让两条路互相牵制
    if args.post {
        return run_post(&args, &train, val.as_ref(), &loss, &fit_cfg, n_in, n_out, &y_shape,
                        grid_h, grid_w, &device);
    }

    if args.ndim == 3 {
        if order.has_vz() || args.velocity_channel_z.is_some() {
            println!("[eser-train] 三维物理已开：div / N-S / 对流扩散都含 ∂/∂z 项");
        } else {
            println!(
                "[eser-train] 注意：通道口径 {:?} 没有第三个速度分量，\n\
                 物理约束仍是**逐层二维**（div 只有 ∂u/∂x+∂v/∂y，没有 ∂w/∂z）。\n\
                 要真三维请用 --channel-order pvvv，或 dispersion 下给 --velocity-channel-z",
                args.channel_order
            );
        }
    }

    let logs = if args.ndim == 3 {
        let depth = y_shape[2];
        let mut mcfg = DenseEd3dConfig::new(n_in, n_out, args.blocks.clone());
        apply_model_args(
            &args,
            &mut mcfg.growth_rate,
            &mut mcfg.init_features,
            &mut mcfg.bn_size,
            &mut mcfg.drop_rate,
            &mut mcfg.bottleneck,
            &mut mcfg.num_heads,
            &mut mcfg.use_positional_encoding,
            &mut mcfg.attn_max_tokens,
        );
        mcfg.output_size = Some([depth, grid_h, grid_w]);
        let model: DenseEd3d<B> = mcfg.init(&device)?;
        report_model("DenseED3D", &args, n_in, n_out);
        let (_, logs) = train::fit_3d(model, &train, val.as_ref(), &loss, &fit_cfg, &device)?;
        logs
    } else {
        let mut mcfg = DenseEdConfig::new(n_in, n_out, args.blocks.clone());
        apply_model_args(
            &args,
            &mut mcfg.growth_rate,
            &mut mcfg.init_features,
            &mut mcfg.bn_size,
            &mut mcfg.drop_rate,
            &mut mcfg.bottleneck,
            &mut mcfg.num_heads,
            &mut mcfg.use_positional_encoding,
            &mut mcfg.attn_max_tokens,
        );
        mcfg.output_size = Some([grid_h, grid_w]);
        let model: DenseEd<B> = mcfg.init(&device)?;
        report_model("DenseED", &args, n_in, n_out);
        let (_, logs) = train::fit(
            model,
            &train,
            val.as_ref(),
            &loss,
            &fit_cfg,
            &device,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )?;
        logs
    };

    // ---- 产物 ----
    // **必须写进 <run_dir>/training/**：诊断侧就是从这个子目录读 .txt 的
    // （diagnose.rs 的 exp_dir.join("training")）。写在 run_dir 根下的话，
    // 文件明明生成了，诊断却报「没有找到 r2_test.txt，训练可能未正常结束」——
    // 这条路径不一致端到端跑一次就撞上了，单测各自绿着也照样漏
    let train_dir = run_dir.join("training");
    train::save_stats(&train_dir, &logs)?;
    train::plot_history(&train_dir, &logs)?;
    println!(
        "[eser-train] 完成。指标与曲线已写到 {}，checkpoint 在 {}",
        train_dir.display(),
        args.ckpt_dir().display()
    );
    Ok(())
}

/// `--post`：只载入 checkpoint 跑一次评估，不做任何参数更新。
///
/// 与 Python `train.py` 的 `--post` 分支对应，但**有一处刻意不一样**：Python 找不到
/// checkpoint 时只打一行警告，然后拿**随机初始化的权重**去评估并把 R² 写进
/// `r2_test.txt`。那是一份看起来正常、实则毫无意义的指标文件，而且事后没法从文件本身
/// 分辨。这里改成直接报错退出。
#[allow(clippy::too_many_arguments)]
fn run_post(
    args: &Args,
    train: &CfdDataset,
    val: Option<&CfdDataset>,
    loss: &PhysicsLoss<B>,
    fit_cfg: &FitConfig,
    n_in: usize,
    n_out: usize,
    y_shape: &[usize],
    grid_h: usize,
    grid_w: usize,
    device: &<B as burn::tensor::backend::Backend>::Device,
) -> Result<(), String> {
    // 评估要在「没参与训练」的那份数据上做。没有验证集就没有可评估的对象——
    // 拿训练集顶会给出一个偏乐观、且与训练日志不可比的数
    let eval_set = val.ok_or_else(|| {
        format!(
            "--post 需要一份验证/测试集：给 --test-file，或让 --val-split 大于 0（当前 {}）",
            args.val_split
        )
    })?;

    let ckpt = post_checkpoint_path(args)?;
    println!("[post] 载入 {}", ckpt.display());

    let train_dir = args.run_dir().join("training");
    let rep = if args.ndim == 3 {
        let depth = y_shape[2];
        let mut mcfg = DenseEd3dConfig::new(n_in, n_out, args.blocks.clone());
        apply_model_args(
            args, &mut mcfg.growth_rate, &mut mcfg.init_features, &mut mcfg.bn_size,
            &mut mcfg.drop_rate, &mut mcfg.bottleneck, &mut mcfg.num_heads,
            &mut mcfg.use_positional_encoding, &mut mcfg.attn_max_tokens,
        );
        mcfg.output_size = Some([depth, grid_h, grid_w]);
        let shell: DenseEd3d<B> = mcfg.init(device)?;
        let (model, meta) = train::load_checkpoint::<B, _>(shell, &ckpt)?;
        report_model("DenseED3D", args, n_in, n_out);
        train::evaluate_once_3d::<B, _>(
            &model,
            eval_set,
            fit_cfg.test_batch_size.unwrap_or(fit_cfg.batch_size),
            n_out,
            meta.epoch,
            device,
        )?
    } else {
        let mut mcfg = DenseEdConfig::new(n_in, n_out, args.blocks.clone());
        apply_model_args(
            args, &mut mcfg.growth_rate, &mut mcfg.init_features, &mut mcfg.bn_size,
            &mut mcfg.drop_rate, &mut mcfg.bottleneck, &mut mcfg.num_heads,
            &mut mcfg.use_positional_encoding, &mut mcfg.attn_max_tokens,
        );
        mcfg.output_size = Some([grid_h, grid_w]);
        let shell: DenseEd<B> = mcfg.init(device)?;
        let (model, meta) = train::load_checkpoint::<B, _>(shell, &ckpt)?;
        report_model("DenseED", args, n_in, n_out);
        train::evaluate_once(
            &model, eval_set, loss, fit_cfg.test_batch_size.unwrap_or(fit_cfg.batch_size),
            n_out, meta.epoch, fit_cfg.velocity_channels, device,
            |m, x| m.forward(x),
        )?
    };
    let _ = train;

    let per_ch: Vec<String> = rep
        .r2_per_channel
        .iter()
        .map(|v| format!("{:.4}", v))
        .collect();
    println!(
        "[post] epoch {}｜R²={:.6}  RMSE={:.6}  分通道 R²=[{}]",
        rep.epoch,
        rep.r2,
        rep.rmse,
        per_ch.join(", ")
    );
    for (name, v) in [
        ("div_rms", rep.physics.div_rms),
        ("ns_rms", rep.physics.ns_rms),
        ("advdiff_rms", rep.physics.advdiff_rms),
    ] {
        if let Some(v) = v {
            println!("[post] {} = {:.6e}", name, v);
        }
    }

    train::save_eval_stats(&train_dir, &rep)?;
    println!("[post] 指标已写到 {}", train_dir.display());
    Ok(())
}

/// `--post` 要评估哪个 checkpoint。
///
/// 给了 `--ckpt-epoch` 就用 `epoch<N>`，否则用训练时留下的 `best`。找不到就报错，
/// 并把目录里实际有什么列出来——「文件名猜错了」和「压根没训练过」是两回事。
fn post_checkpoint_path(args: &Args) -> Result<PathBuf, String> {
    let dir = args.ckpt_dir();
    let name = match args.ckpt_epoch {
        Some(e) => format!("epoch{}", e),
        None => "best".to_string(),
    };
    let path = dir.join(&name);
    // Burn 的记录器落的是 <name>.mpk，元信息是 <name>.meta，两个都得在
    let mpk = dir.join(format!("{}.mpk", name));
    let meta = dir.join(format!("{}.meta", name));
    if mpk.is_file() && meta.is_file() {
        return Ok(path);
    }
    let mut have: Vec<String> = std::fs::read_dir(&dir)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    have.sort();
    Err(format!(
        "找不到 checkpoint {:?}（需要 {}.mpk 与 {}.meta）。目录 {} 里现有：{}",
        name,
        name,
        name,
        dir.display(),
        if have.is_empty() {
            "（空）".to_string()
        } else {
            have.join(", ")
        }
    ))
}

/// 逐 epoch 打印。GUI 靠 stdout 实时滚日志，所以这里每轮都要冲一行出去。
fn report_model(kind: &str, args: &Args, n_in: usize, n_out: usize) {
    println!(
        "[eser-train] {} in={} out={} blocks={:?} growth={} init={} heads={}",
        kind, n_in, n_out, args.blocks, args.growth_rate, args.init_features, args.num_heads
    );
}

#[allow(clippy::too_many_arguments)]
fn apply_model_args(
    args: &Args,
    growth_rate: &mut usize,
    init_features: &mut usize,
    bn_size: &mut usize,
    drop_rate: &mut f64,
    bottleneck: &mut bool,
    num_heads: &mut usize,
    use_pe: &mut bool,
    attn_max_tokens: &mut usize,
) {
    *growth_rate = args.growth_rate;
    *init_features = args.init_features;
    *bn_size = args.bn_size;
    *drop_rate = args.drop_rate;
    *bottleneck = args.bottleneck;
    *num_heads = args.num_heads;
    // **注意取反**：命令行给的是「关掉」，配置项问的是「开着吗」
    *use_pe = !args.no_positional_encoding;
    *attn_max_tokens = args.attn_max_tokens;
}

/// 写进 checkpoint 的超参。**推理侧靠这些重建同构的网络**，少一样就只能靠猜。
fn checkpoint_hyper(args: &Args) -> Vec<(String, String)> {
    vec![
        ("channel_order".into(), args.channel_order.clone()),
        ("task".into(), args.task.clone()),
        ("blocks".into(), join_usize(&args.blocks)),
        ("growth_rate".into(), args.growth_rate.to_string()),
        ("init_features".into(), args.init_features.to_string()),
        ("bn_size".into(), args.bn_size.to_string()),
        ("num_heads".into(), args.num_heads.to_string()),
        ("attn_max_tokens".into(), args.attn_max_tokens.to_string()),
        (
            "no_positional_encoding".into(),
            args.no_positional_encoding.to_string(),
        ),
        ("bottleneck".into(), args.bottleneck.to_string()),
        ("diffusivity".into(), args.diffusivity.to_string()),
        ("no_mask".into(), args.no_mask.to_string()),
        (
            "t_geometry".into(),
            args.t_geometry.clone().unwrap_or_default(),
        ),
    ]
}

fn join_usize(v: &[usize]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// 计算域掩码 + 边界分区几何。两者都可能缺席，含义不同：
/// 掩码 `None` = `--no-mask`（损失退化成全场平均）；边界 `None` = 这份几何没有分区标签。
type Domain = (Option<Tensor<B, 4>>, Option<Boundary<B>>);

/// 建计算域掩码与边界分区几何。
fn build_mask(
    args: &Args,
    d: usize,
    h: usize,
    w: usize,
    device: &<B as burn::tensor::backend::Backend>::Device,
) -> Result<Domain, String> {
    if args.no_mask {
        return Ok((None, None));
    }
    // 复用推理侧的几何解析：`.npy` 栅格、`.json` 矩形段、内联 JSON、缺省 T 型
    // 四条路都在那儿，不要在这里再写一份
    let spec = GeometrySpec::parse(args.t_geometry.as_deref())?;
    // 掩码的层数由文件决定：2D 的 .npy 给 1 层，会被广播到所有深度层；
    // 3D 的 (D, H, W) 必须与数据的层数一致，否则第 k 层会配到错误的轮廓
    let mask_d = spec.depth();
    if mask_d != 1 && mask_d != d {
        return Err(format!(
            "掩码是 {} 层，数据是 {} 层，对不上。\n\
             二维掩码（一层）会广播到所有层，三维掩码必须逐层对应",
            mask_d, d
        ));
    }
    let flat = spec.mask(h, w)?;
    let cov = geometry::coverage(&flat);
    println!(
        "[eser-train] 计算域覆盖率 {:.2}%{}",
        cov * 100.0,
        if mask_d > 1 {
            format!("（{} 层逐层不同）", mask_d)
        } else {
            String::new()
        }
    );
    // 第 0 维放深度：`PhysicsLoss::compute_3d` 就是按这个口径复制到 (B*D, C, H, W) 的
    let mask = Tensor::<B, 1>::from_data(burn::tensor::TensorData::from(flat.as_slice()), device)
        .reshape([mask_d, 1, h, w]);

    let (dx, dy) = args.spacing(h, w);
    let boundary = match spec.zones(h, w)? {
        // 只在**真的画了边界标签**时拒绝。纯 0/1 的三维掩码没有边界要处理，照跑。
        // 边界法向是二维的（∇ 只含 x/y），竖向边界（楼板、楼梯井壁）的法向带 z 分量，
        // 这一版算不出来。宁可明说不支持，也不给一个「只约束了水平方向」却看着正常的东西
        Some(z) if mask_d > 1 && geometry::zone_counts(&z)[2..].iter().any(|&n| n > 0) => {
            return Err("三维掩码暂不支持边界分区标签（法向的 z 分量还没实现）；
                        三维请先用只含 0/1 的掩码，或按二维逐层训练"
                .into());
        }
        Some(_) if mask_d > 1 => None,
        None => None,
        Some(z) => {
            let c = geometry::zone_counts(&z);
            println!(
                "[eser-train] 分区格点数：流体 {} / 壁 {} / 入口 {} / 出口 {}",
                c[1], c[2], c[3], c[4]
            );
            Boundary::<B>::build(&z, h, w, dx, dy, device)
        }
    };

    match &boundary {
        None => println!(
            "[eser-train] 这份几何没有边界标签，**不加任何边界条件约束**\n\
             （想加的话把掩码 .npy 里的墙画成 2、送风口 3、排风口 4）"
        ),
        Some(b) => {
            let n = |z: &Option<BoundaryZone<B>>| {
                z.as_ref().map(|v| v.count).unwrap_or(0)
            };
            println!(
                "[eser-train] 贴边流体格点：壁 {} / 入口 {} / 出口 {}（边界权重 {}）",
                n(&b.wall),
                n(&b.inlet),
                n(&b.outlet),
                args.bc_weight
            );
            // 画了风口却不给风速 = 参数漏了。默认成 0 等于把风口当墙，
            // 结果全错却不会报任何错，正是本项目一路在防的那类静默失败
            if b.inlet.is_some() && args.inlet_speed.is_none() {
                return Err("掩码里有送风口（标签 3），必须用 --inlet-speed 给出风速".into());
            }
            if b.outlet.is_some() && args.outlet_speed.is_none() {
                return Err("掩码里有排风口（标签 4），必须用 --outlet-speed 给出风速".into());
            }
            if args.bc_weight <= 0.0 {
                println!("[eser-train] 警告：--bc-weight = 0，边界条件只监控、不约束");
            }
        }
    }
    Ok((Some(mask), boundary))
}

fn resolve_train_file(args: &Args) -> Result<PathBuf, String> {
    let name = args
        .train_file
        .as_ref()
        .ok_or_else(|| "必须给 --train-file".to_string())?;
    let p = resolve_under(&args.data_dir, name);
    if !p.exists() {
        return Err(format!("训练数据不存在：{}", p.display()));
    }
    Ok(p)
}

/// 相对路径按 `--data-dir` 解析；已经带目录的原样用（与 Python 的 `resolve_path` 一致）。
fn resolve_under(data_dir: &str, name: &str) -> PathBuf {
    let p = PathBuf::from(name);
    if p.is_absolute() || p.components().count() > 1 || p.exists() {
        p
    } else {
        PathBuf::from(data_dir).join(name)
    }
}

/// 落一份参数快照。`diagnose` 读它做配置体检——不写的话诊断只能报「找不到配置」。
fn write_args_snapshot(dir: &Path, args: &Args, train_path: &Path) -> Result<(), String> {
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let mut j = String::from("{\n");
    let mut push = |k: &str, v: String| {
        j.push_str(&format!("    \"{}\": {},\n", k, v));
    };
    push(
        "train_file",
        format!("\"{}\"", esc(&train_path.display().to_string())),
    );
    push("task", format!("\"{}\"", esc(&args.task)));
    push("channel_order", format!("\"{}\"", esc(&args.channel_order)));
    push("ndim", args.ndim.to_string());
    push("epochs", args.epochs.to_string());
    push("batch_size", args.batch_size.to_string());
    push("lr", format!("{:?}", args.lr));
    push("weight_decay", format!("{:?}", args.weight_decay));
    push("grad_clip", format!("{:?}", args.grad_clip));
    push("val_split", format!("{:?}", args.val_split));
    push("rho", format!("{:?}", args.rho));
    push("nu", format!("{:?}", args.nu));
    push("div_weight", format!("{:?}", args.div_weight));
    push("ns_weight", format!("{:?}", args.ns_weight));
    push("smooth_weight", format!("{:?}", args.smooth_weight));
    push("advdiff_weight", format!("{:?}", args.advdiff_weight));
    push("bc_weight", format!("{:?}", args.bc_weight));
    push("domain_height", format!("{:?}", args.domain_height));
    if let Some(i) = args.velocity_channel_z {
        push("velocity_channel_z", i.to_string());
    }
    if let Some(q) = args.inlet_speed {
        push("inlet_speed", format!("{:?}", q));
    }
    if let Some(q) = args.outlet_speed {
        push("outlet_speed", format!("{:?}", q));
    }
    push("diffusivity", format!("{:?}", args.diffusivity));
    push("physics_warmup", format!("{:?}", args.physics_warmup));
    push(
        "physics_in_physical_units",
        args.physics_in_physical_units.to_string(),
    );
    push(
        "physics_loss_scaling",
        format!("\"{}\"", esc(&args.physics_loss_scaling)),
    );
    push("no_mask", args.no_mask.to_string());
    push(
        "t_geometry",
        match &args.t_geometry {
            Some(s) => format!("\"{}\"", esc(s)),
            None => "null".into(),
        },
    );
    push(
        "domain_size",
        format!("[{:?}, {:?}]", args.domain_size[0], args.domain_size[1]),
    );
    push(
        "velocity_channels",
        format!(
            "[{}, {}]",
            args.velocity_channels.0, args.velocity_channels.1
        ),
    );
    push("blocks", format!("[{}]", join_usize(&args.blocks)));
    push("growth_rate", args.growth_rate.to_string());
    push("init_features", args.init_features.to_string());
    push("num_heads", args.num_heads.to_string());
    push("seed", args.seed.to_string());
    j.push_str("    \"trainer\": \"eser-train (rust)\"\n}\n");
    std::fs::write(dir.join("args.txt"), j).map_err(|e| format!("写 args.txt 失败：{}", e))
}
