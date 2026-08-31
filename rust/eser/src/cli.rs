//! 命令行参数解析。**旗标名与 `args.py` 逐字一致**。
//!
//! 这样做的好处是 GUI 那边的命令拼装（`gui/src/params.rs::build_args`）一个字都不用
//! 改——从 `python train.py ...` 切到 Rust 二进制只换可执行文件。名字对不上的话，
//! 两处要各改一遍，而漏改一处只会在运行时报「unrecognized argument」。
//!
//! # 为什么手写而不是引 clap
//!
//! 需要解析的是一组**固定的、由 GUI 生成的** `--flag value`，没有子命令、没有简写、
//! 没有位置参数。手写一个查表式解析器约 100 行，还能顺带把 `args.py` 的默认值与
//! 校验规则写在同一处——引一个大依赖换不到什么。P6 出于同样的理由手写了 JSON 解析。

use std::collections::HashMap;
use std::path::PathBuf;

/// 解析后的训练参数。字段名与 `args.py` 一一对应（下划线 ↔ 连字符）。
#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub train_file: Option<String>,
    pub test_file: Option<String>,
    pub data_dir: String,
    pub exp_dir: String,
    pub exp_name: String,
    pub ndim: usize,
    pub task: String,
    pub channel_order: String,
    pub domain_size: [f64; 2],
    pub val_split: f64,
    pub no_mask: bool,
    pub t_geometry: Option<String>,

    pub rho: f64,
    pub nu: f64,
    pub div_weight: f64,
    pub ns_weight: f64,
    pub smooth_weight: f64,
    pub diffusivity: f64,
    pub advdiff_weight: f64,
    pub physics_warmup: f64,
    pub physics_loss_scaling: String,
    pub physics_in_physical_units: bool,
    pub source_scale: f64,
    pub velocity_channels: (usize, usize),

    pub epochs: usize,
    pub batch_size: usize,
    pub test_batch_size: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub grad_clip: f64,
    pub lr_patience: usize,
    pub seed: u32,

    pub blocks: Vec<usize>,
    pub growth_rate: usize,
    pub init_features: usize,
    pub bn_size: usize,
    pub drop_rate: f64,
    pub bottleneck: bool,
    pub num_heads: usize,
    pub no_positional_encoding: bool,
    pub attn_max_tokens: usize,

    pub ckpt_freq: usize,
    pub plot_freq: usize,
    pub log_freq: usize,
    pub input_key: Option<String>,
    pub prefer_sensor_input: bool,
    pub num_workers: Option<usize>,
    pub plot_fn: String,
    pub post: bool,
    pub ckpt_epoch: Option<usize>,
}

impl Default for Args {
    /// 逐条对齐 `args.py` 的 CLI 默认值。
    fn default() -> Self {
        Self {
            train_file: None,
            test_file: None,
            data_dir: "dataset".into(),
            exp_dir: "experiments".into(),
            exp_name: "run01".into(),
            ndim: 2,
            task: "flow".into(),
            channel_order: "pvv".into(),
            domain_size: [1.0, 1.0],
            val_split: 0.2,
            no_mask: false,
            t_geometry: None,

            rho: 1.0,
            nu: 1e-2,
            div_weight: 0.05,
            ns_weight: 0.005,
            smooth_weight: 0.005,
            diffusivity: 1e-5,
            advdiff_weight: 0.05,
            physics_warmup: 0.1,
            physics_loss_scaling: "ema".into(),
            physics_in_physical_units: true,
            source_scale: 1.0,
            velocity_channels: (0, 1),

            epochs: 200,
            batch_size: 16,
            test_batch_size: 100,
            lr: 3e-3,
            weight_decay: 5e-4,
            grad_clip: 1.0,
            lr_patience: 10,
            seed: 1,

            blocks: vec![3, 6, 3],
            growth_rate: 16,
            init_features: 48,
            bn_size: 8,
            drop_rate: 0.0,
            bottleneck: false,
            num_heads: 4,
            no_positional_encoding: false,
            attn_max_tokens: 8192,

            ckpt_freq: 25,
            plot_freq: 50,
            log_freq: 1,
            input_key: None,
            prefer_sensor_input: false,
            num_workers: None,
            plot_fn: "imshow".into(),
            post: false,
            ckpt_epoch: None,
        }
    }
}

impl Args {
    /// `<exp_dir>/<exp_name>`，与 `args.py:175` 一致。
    pub fn run_dir(&self) -> PathBuf {
        PathBuf::from(&self.exp_dir).join(&self.exp_name)
    }

    /// `<run_dir>/checkpoints`。
    pub fn ckpt_dir(&self) -> PathBuf {
        self.run_dir().join("checkpoints")
    }

    /// 网格间距。**分母是 N-1 不是 N**——差一格会让所有物理残差整体偏一个常数因子，
    /// 而这不会报错（与 `train.py:126-127` 一致）。
    pub fn spacing(&self, grid_h: usize, grid_w: usize) -> (f64, f64) {
        let dy = self.domain_size[0] / (grid_h.max(2) - 1) as f64;
        let dx = self.domain_size[1] / (grid_w.max(2) - 1) as f64;
        (dx, dy)
    }
}

/// 需要两个取值的旗标（`--domain-size Ly Lx`、`--velocity-channels vx vy`）。
const PAIRS: &[&str] = &["--domain-size", "--velocity-channels"];

/// 不带取值的开关。
const FLAGS: &[&str] = &[
    "--no-mask",
    "--bottleneck",
    "--no-positional-encoding",
    "--prefer-sensor-input",
    "--post",
];

/// 解析 `--flag value` 形式的参数。
///
/// **不认识的旗标一律报错**，不静默忽略：GUI 与 CLI 的旗标名一旦漂移，静默忽略会让
/// 训练用着默认值跑完，看起来一切正常。
pub fn parse(argv: &[String]) -> Result<Args, String> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    let mut i = 0;
    while i < argv.len() {
        let key = &argv[i];
        if !key.starts_with("--") {
            return Err(format!("不认识的参数 {:?}（期望以 -- 开头）", key));
        }
        if FLAGS.contains(&key.as_str()) {
            map.insert(key.clone(), Vec::new());
            i += 1;
        } else if PAIRS.contains(&key.as_str()) {
            let a = argv.get(i + 1).ok_or_else(|| format!("{} 缺少取值", key))?;
            let b = argv
                .get(i + 2)
                .ok_or_else(|| format!("{} 缺少第二个取值", key))?;
            map.insert(key.clone(), vec![a.clone(), b.clone()]);
            i += 3;
        } else {
            let v = argv.get(i + 1).ok_or_else(|| format!("{} 缺少取值", key))?;
            map.insert(key.clone(), vec![v.clone()]);
            i += 2;
        }
    }

    let mut a = Args::default();
    let mut unknown = Vec::new();

    for (key, vals) in &map {
        let one = || -> Result<&String, String> {
            vals.first().ok_or_else(|| format!("{} 缺少取值", key))
        };
        let num = |s: &str| -> Result<f64, String> {
            s.parse::<f64>()
                .map_err(|e| format!("{} 的取值 {:?} 不是数字：{}", key, s, e))
        };
        let int = |s: &str| -> Result<i64, String> {
            s.parse::<i64>()
                .map_err(|e| format!("{} 的取值 {:?} 不是整数：{}", key, s, e))
        };
        let non_negative = |s: &str| -> Result<usize, String> {
            let v = int(s)?;
            if v < 0 {
                return Err(format!("{} 不能为负，实得 {}", key, v));
            }
            Ok(v as usize)
        };

        match key.as_str() {
            "--train-file" => a.train_file = Some(one()?.clone()),
            "--test-file" => a.test_file = Some(one()?.clone()),
            "--data-dir" => a.data_dir = one()?.clone(),
            "--exp-dir" => a.exp_dir = one()?.clone(),
            "--exp-name" => a.exp_name = one()?.clone(),
            "--ndim" => a.ndim = non_negative(one()?)?,
            "--use-3d" => a.ndim = 3,
            "--task" => a.task = one()?.clone(),
            "--channel-order" => a.channel_order = one()?.clone(),
            "--domain-size" => {
                a.domain_size = [num(&vals[0])?, num(&vals[1])?];
            }
            "--val-split" => a.val_split = num(one()?)?,
            "--no-mask" => a.no_mask = true,
            // `--domain-mask` 是 `--t-geometry` 的别名，两个都要认
            "--t-geometry" | "--domain-mask" => a.t_geometry = Some(one()?.clone()),

            "--rho" => a.rho = num(one()?)?,
            "--nu" => a.nu = num(one()?)?,
            "--div-weight" => a.div_weight = num(one()?)?,
            "--ns-weight" => a.ns_weight = num(one()?)?,
            "--smooth-weight" => a.smooth_weight = num(one()?)?,
            "--diffusivity" => a.diffusivity = num(one()?)?,
            "--advdiff-weight" => a.advdiff_weight = num(one()?)?,
            "--physics-warmup" => a.physics_warmup = num(one()?)?,
            "--physics-loss-scaling" => a.physics_loss_scaling = one()?.clone(),
            "--physics-in-physical-units" => a.physics_in_physical_units = parse_bool(one()?, key)?,
            "--source-scale" => a.source_scale = num(one()?)?,
            "--velocity-channels" => {
                a.velocity_channels = (non_negative(&vals[0])?, non_negative(&vals[1])?)
            }

            "--epochs" => a.epochs = non_negative(one()?)?,
            "--batch-size" => a.batch_size = non_negative(one()?)?,
            "--test-batch-size" => a.test_batch_size = non_negative(one()?)?,
            "--lr" => a.lr = num(one()?)?,
            "--weight-decay" => a.weight_decay = num(one()?)?,
            "--grad-clip" => a.grad_clip = num(one()?)?,
            "--lr-patience" => a.lr_patience = non_negative(one()?)?,
            "--seed" => a.seed = non_negative(one()?)? as u32,

            "--blocks" => a.blocks = parse_blocks(one()?)?,
            "--growth-rate" => a.growth_rate = non_negative(one()?)?,
            "--init-features" => a.init_features = non_negative(one()?)?,
            "--bn-size" => a.bn_size = non_negative(one()?)?,
            "--drop-rate" => a.drop_rate = num(one()?)?,
            "--bottleneck" => a.bottleneck = true,
            "--num-heads" => a.num_heads = non_negative(one()?)?,
            "--no-positional-encoding" => a.no_positional_encoding = true,
            "--attn-max-tokens" => a.attn_max_tokens = non_negative(one()?)?,

            "--ckpt-freq" => a.ckpt_freq = non_negative(one()?)?,
            "--plot-freq" => a.plot_freq = non_negative(one()?)?,
            "--log-freq" => a.log_freq = non_negative(one()?)?,
            "--input-key" => a.input_key = Some(one()?.clone()),
            "--prefer-sensor-input" => a.prefer_sensor_input = true,
            "--num-workers" => a.num_workers = Some(non_negative(one()?)?),
            "--plot-fn" => a.plot_fn = one()?.clone(),
            "--post" => a.post = true,
            "--ckpt-epoch" => a.ckpt_epoch = Some(non_negative(one()?)?),

            other => unknown.push(other.to_string()),
        }
    }

    if !unknown.is_empty() {
        unknown.sort();
        return Err(format!(
            "不认识的参数：{}。\n（旗标名要与 args.py 一致；静默忽略会让训练用着\
             默认值跑完而看起来一切正常）",
            unknown.join("、")
        ));
    }

    a.validate()?;
    Ok(a)
}

fn parse_bool(s: &str, key: &str) -> Result<bool, String> {
    match s.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(format!("{} 的取值 {:?} 不是布尔值", key, other)),
    }
}

/// `"3,6,3"` 或 `"3 6 3"` → `[3, 6, 3]`，与 `args.py` 的 `_parse_blocks` 一致。
pub fn parse_blocks(text: &str) -> Result<Vec<usize>, String> {
    let parts: Vec<&str> = text
        .split([',', ' ', '\t'])
        .filter(|t| !t.is_empty())
        .collect();
    if parts.is_empty() {
        return Err("--blocks 为空".into());
    }
    let mut out = Vec::with_capacity(parts.len());
    for p in parts {
        let v: i64 = p
            .parse()
            .map_err(|e| format!("--blocks 里的 {:?} 不是整数：{}", p, e))?;
        if v <= 0 {
            return Err(format!("--blocks 里的 {} 必须为正", v));
        }
        out.push(v as usize);
    }
    Ok(out)
}

impl Args {
    fn validate(&self) -> Result<(), String> {
        if self.epochs == 0 {
            return Err("--epochs 必须为正".into());
        }
        if self.batch_size == 0 {
            return Err("--batch-size 必须为正".into());
        }
        if self.ckpt_freq == 0 {
            return Err("--ckpt-freq 必须为正".into());
        }
        if !(0.0..1.0).contains(&self.val_split) {
            return Err(format!(
                "--val-split 要落在 [0, 1)，实得 {}",
                self.val_split
            ));
        }
        if self.blocks.len() > 1 && self.blocks.len().is_multiple_of(2) {
            return Err(format!("--blocks 的长度必须是奇数，实得 {:?}", self.blocks));
        }
        if !matches!(self.ndim, 2 | 3) {
            return Err(format!("--ndim 只能是 2 或 3，实得 {}", self.ndim));
        }
        if !matches!(self.task.as_str(), "flow" | "dispersion") {
            return Err(format!(
                "--task 只能是 flow 或 dispersion，实得 {:?}",
                self.task
            ));
        }
        if self.domain_size.iter().any(|v| *v <= 0.0) {
            return Err(format!(
                "--domain-size 必须为正，实得 {:?}",
                self.domain_size
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(|x| x.to_string()).collect()
    }

    #[test]
    fn defaults_match_the_python_cli() {
        // 与 args.py 的 add_argument(default=...) 逐条对照。对不上时训练轨迹会分岔，
        // 而这不报错——只表现为「Rust 版调不出 Python 那个效果」
        let a = Args::default();
        assert_eq!(a.epochs, 200);
        assert_eq!(a.batch_size, 16);
        assert_eq!(a.test_batch_size, 100);
        assert_eq!(a.lr, 3e-3);
        assert_eq!(a.weight_decay, 5e-4);
        assert_eq!(a.grad_clip, 1.0);
        assert_eq!(a.lr_patience, 10);
        assert_eq!(a.ckpt_freq, 25);
        assert_eq!(a.seed, 1);
        assert_eq!(a.growth_rate, 16);
        assert_eq!(a.init_features, 48);
        assert_eq!(a.bn_size, 8);
        assert_eq!(a.num_heads, 4);
        assert_eq!(a.attn_max_tokens, 8192);
        assert_eq!(a.blocks, vec![3, 6, 3]);
        assert_eq!(a.velocity_channels, (0, 1));
        assert_eq!(a.val_split, 0.2);
        assert_eq!(a.exp_dir, "experiments");
        assert_eq!(a.data_dir, "dataset");
    }

    #[test]
    fn parses_the_flags_the_gui_emits() {
        let a = parse(&argv(
            "--train-file dataset/x.hdf5 --task dispersion --channel-order c \
             --domain-size 2.5 1.5 --velocity-channels 1 2 --epochs 7 --no-mask \
             --bottleneck --blocks 2,2,2 --physics-in-physical-units false \
             --num-workers 4 --ndim 3",
        ))
        .unwrap();
        assert_eq!(a.train_file.as_deref(), Some("dataset/x.hdf5"));
        assert_eq!(a.task, "dispersion");
        assert_eq!(a.domain_size, [2.5, 1.5]);
        assert_eq!(a.velocity_channels, (1, 2));
        assert_eq!(a.epochs, 7);
        assert!(a.no_mask && a.bottleneck);
        assert_eq!(a.blocks, vec![2, 2, 2]);
        assert!(!a.physics_in_physical_units);
        assert_eq!(a.num_workers, Some(4));
        assert_eq!(a.ndim, 3);
    }

    #[test]
    fn domain_mask_is_an_alias_for_t_geometry() {
        // 推理侧曾把 t_geometry 丢在白名单外导致车站模型被 T 型几何裁剪，
        // 这条别名不能丢
        let a = parse(&argv("--domain-mask station.npy")).unwrap();
        assert_eq!(a.t_geometry.as_deref(), Some("station.npy"));
    }

    #[test]
    fn unknown_flags_are_rejected_not_ignored() {
        // 静默忽略会让训练用着默认值跑完，看起来一切正常
        let err = parse(&argv("--epochs 3 --typo-flag 5")).unwrap_err();
        assert!(err.contains("--typo-flag"), "报错没点名：{}", err);
    }

    #[test]
    fn invalid_values_are_rejected() {
        assert!(parse(&argv("--epochs 0")).unwrap_err().contains("epochs"));
        assert!(parse(&argv("--val-split 1.5"))
            .unwrap_err()
            .contains("val-split"));
        assert!(parse(&argv("--blocks 2,2")).unwrap_err().contains("奇数"));
        assert!(parse(&argv("--ndim 4")).unwrap_err().contains("ndim"));
        assert!(parse(&argv("--task nonsense"))
            .unwrap_err()
            .contains("task"));
        assert!(parse(&argv("--epochs")).unwrap_err().contains("缺少取值"));
        assert!(parse(&argv("--epochs abc"))
            .unwrap_err()
            .contains("不是整数"));
    }

    #[test]
    fn grid_spacing_divides_by_n_minus_one() {
        // **分母是 N-1 不是 N**：差一格会让所有物理残差整体偏一个常数因子，且不报错
        let a = Args {
            domain_size: [1.0, 2.0],
            ..Default::default()
        };
        let (dx, dy) = a.spacing(11, 21);
        assert!((dy - 0.1).abs() < 1e-12, "dy 实得 {}", dy);
        assert!((dx - 0.1).abs() < 1e-12, "dx 实得 {}", dx);
    }

    #[test]
    fn run_dir_layout_matches_python() {
        let a = Args {
            exp_dir: "exp".into(),
            exp_name: "r1".into(),
            ..Default::default()
        };
        assert!(a.run_dir().ends_with("exp/r1") || a.run_dir().ends_with("exp\\r1"));
        assert!(a.ckpt_dir().ends_with("checkpoints"));
    }

    #[test]
    fn blocks_accepts_both_separators() {
        assert_eq!(parse_blocks("3,6,3").unwrap(), vec![3, 6, 3]);
        assert_eq!(parse_blocks("3 6 3").unwrap(), vec![3, 6, 3]);
        assert!(parse_blocks("3,0,3").is_err(), "0 层应当被拒绝");
        assert!(parse_blocks("").is_err());
    }
}
