//! 训练参数表单的数据模型与命令行拼装。
//!
//! 全部是纯逻辑（不摸 UI、不摸文件系统、不摸子进程），方便脱离 egui 单独做单元测试。
//! `TrainParams::build_args` 是这个文件的核心：把表单状态翻译成 `train.py` 认识的
//! 命令行参数列表，一一对应 `args.py` 里的 `add_argument`。

use std::fmt;

/// 物理任务类型，对应 `--task`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Flow,
    Dispersion,
}

impl Task {
    pub const ALL: [Task; 2] = [Task::Flow, Task::Dispersion];

    fn as_arg(self) -> &'static str {
        match self {
            Task::Flow => "flow",
            Task::Dispersion => "dispersion",
        }
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Task::Flow => write!(f, "flow（流场：连续性 + 动量）"),
            Task::Dispersion => write!(f, "dispersion（污染物输运：对流扩散）"),
        }
    }
}

/// 输出通道语义，对应 `--channel-order`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelOrder {
    Pvv,
    Vvp,
    Pvvt,
    C,
}

impl ChannelOrder {
    pub const ALL: [ChannelOrder; 4] = [
        ChannelOrder::Pvv,
        ChannelOrder::Vvp,
        ChannelOrder::Pvvt,
        ChannelOrder::C,
    ];

    fn as_arg(self) -> &'static str {
        match self {
            ChannelOrder::Pvv => "pvv",
            ChannelOrder::Vvp => "vvp",
            ChannelOrder::Pvvt => "pvvt",
            ChannelOrder::C => "c",
        }
    }
}

impl fmt::Display for ChannelOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChannelOrder::Pvv => write!(f, "pvv = (P, Vx, Vy)"),
            ChannelOrder::Vvp => write!(f, "vvp = (Vx, Vy, P)"),
            ChannelOrder::Pvvt => write!(f, "pvvt = (P, Vx, Vy, T)"),
            ChannelOrder::C => write!(f, "c = (C,)"),
        }
    }
}

/// 物理项量级归一化方式，对应 `--physics-loss-scaling`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsScaling {
    Ema,
    None,
}

impl PhysicsScaling {
    pub const ALL: [PhysicsScaling; 2] = [PhysicsScaling::Ema, PhysicsScaling::None];

    fn as_arg(self) -> &'static str {
        match self {
            PhysicsScaling::Ema => "ema",
            PhysicsScaling::None => "none",
        }
    }
}

impl fmt::Display for PhysicsScaling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhysicsScaling::Ema => write!(f, "ema（滑动估计拉回 O(1)，推荐）"),
            PhysicsScaling::None => write!(f, "none（原始量级，仅供对比）"),
        }
    }
}

/// 预测图渲染方式，对应 `--plot-fn`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotFn {
    Imshow,
    Contourf,
}

impl PlotFn {
    pub const ALL: [PlotFn; 2] = [PlotFn::Imshow, PlotFn::Contourf];

    fn as_arg(self) -> &'static str {
        match self {
            PlotFn::Imshow => "imshow",
            PlotFn::Contourf => "contourf",
        }
    }
}

impl fmt::Display for PlotFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlotFn::Imshow => write!(f, "imshow（真值/预测严格同色阶）"),
            PlotFn::Contourf => write!(f, "contourf（会忽略 vmin/vmax）"),
        }
    }
}

/// 训练参数表单的完整状态。字段名与默认值对齐 `args.py`。
#[derive(Debug, Clone, PartialEq)]
pub struct TrainParams {
    // ---- 数据 ----
    pub train_file: String,
    pub test_file: String,
    pub val_split: f32,
    pub ndim3: bool,

    // ---- 物理 ----
    pub task: Task,
    pub channel_order: ChannelOrder,
    pub domain_size_ly: f32,
    pub domain_size_lx: f32,
    pub t_geometry: String,
    pub no_mask: bool,
    pub rho: f32,
    pub nu: f32,
    pub div_weight: f32,
    pub ns_weight: f32,
    pub diffusivity: f32,
    pub advdiff_weight: f32,
    pub smooth_weight: f32,
    pub physics_warmup: f32,

    // ---- 网络结构 ----
    pub blocks: String,
    pub growth_rate: i32,
    pub init_features: i32,
    pub drop_rate: f32,
    pub bn_size: i32,
    pub bottleneck: bool,
    pub num_heads: i32,
    pub no_positional_encoding: bool,
    pub attn_max_tokens: i32,

    // ---- 训练超参 ----
    pub epochs: i32,
    pub batch_size: i32,
    pub lr: f32,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub seed: i32,
    pub lr_patience: i32,
    pub test_batch_size: i32,
    pub log_freq: i32,

    // ---- 实验管理 ----
    pub exp_name: String,
    pub exp_dir: String,
    pub ckpt_freq: i32,
    pub plot_freq: i32,
    pub plot_fn: PlotFn,

    // ---- 高级：数据加载 ----
    pub data_dir: String,
    pub num_workers: Option<i32>,
    pub input_key: String,
    pub prefer_sensor_input: bool,

    // ---- 高级：物理细节 ----
    pub physics_loss_scaling: PhysicsScaling,
    pub physics_in_physical_units: bool,
    pub source_scale: f32,
    pub velocity_vx: i32,
    pub velocity_vy: i32,

    // ---- 高级：仅评估模式 ----
    pub post: bool,
    pub ckpt_epoch: Option<i32>,
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            train_file: "dataset/p_v_256.hdf5".to_string(),
            test_file: String::new(),
            val_split: 0.2,
            ndim3: false,

            task: Task::Flow,
            channel_order: ChannelOrder::Pvv,
            domain_size_ly: 1.0,
            domain_size_lx: 1.0,
            t_geometry: String::new(),
            no_mask: false,
            rho: 1.0,
            nu: 0.01,
            div_weight: 0.05,
            ns_weight: 0.005,
            diffusivity: 1e-5,
            advdiff_weight: 0.05,
            smooth_weight: 0.005,
            physics_warmup: 0.1,

            blocks: "3,6,3".to_string(),
            growth_rate: 16,
            init_features: 48,
            drop_rate: 0.0,
            bn_size: 8,
            bottleneck: false,
            num_heads: 4,
            no_positional_encoding: false,
            attn_max_tokens: 8192,

            epochs: 200,
            batch_size: 16,
            lr: 0.003,
            weight_decay: 0.0005,
            grad_clip: 1.0,
            seed: 1,
            lr_patience: 10,
            test_batch_size: 100,
            log_freq: 2,

            exp_name: "run01".to_string(),
            exp_dir: "./experiments".to_string(),
            ckpt_freq: 50,
            plot_freq: 50,
            plot_fn: PlotFn::Imshow,

            data_dir: "./dataset".to_string(),
            num_workers: None,
            input_key: String::new(),
            prefer_sensor_input: false,

            physics_loss_scaling: PhysicsScaling::Ema,
            physics_in_physical_units: true,
            source_scale: 1.0,
            velocity_vx: 0,
            velocity_vy: 1,

            post: false,
            ckpt_epoch: None,
        }
    }
}

/// 解析 `--blocks`：接受逗号或空白分隔的整数列表，长度必须是奇数（DenseED 的
/// dense block / transition block 交替结构决定的），且每层至少 1 个 dense layer。
/// 只做校验，不改写用户输入——校验通过后原始字符串原样传给 `train.py`，
/// 它自己的 `_blocks()` 解析器认得同样的格式。
pub fn parse_blocks(text: &str) -> Result<Vec<i32>, String> {
    let sep_on_comma = text.contains(',');
    let tokens: Vec<&str> = text
        .split(|c: char| {
            if sep_on_comma {
                c == ','
            } else {
                c.is_whitespace()
            }
        })
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if tokens.is_empty() {
        return Err("--blocks 不能为空".to_string());
    }
    let mut values = Vec::with_capacity(tokens.len());
    for tok in &tokens {
        let v: i32 = tok
            .parse()
            .map_err(|_| format!("--blocks 里的 \"{tok}\" 不是整数"))?;
        if v < 1 {
            return Err(format!("--blocks 里的 {v} 必须 >= 1"));
        }
        values.push(v);
    }
    if values.len() % 2 == 0 {
        return Err(format!(
            "--blocks 长度必须是奇数（当前 {} 项：{:?}）",
            values.len(),
            values
        ));
    }
    Ok(values)
}

/// 往命令行参数列表里追加一个 `--flag value` 对。写成自由函数而不是闭包，是因为
/// 闭包捕获 `cmd` 后，借用检查器会认为整个函数体内 `cmd` 一直被闭包独占借用，
/// 导致中间穿插的 `cmd.push(...)` 直接调用全部编译不过（试过，`E0499`）。
fn kv(cmd: &mut Vec<String>, flag: &str, value: String) {
    cmd.push(flag.to_string());
    cmd.push(value);
}

impl TrainParams {
    /// 把表单状态翻译成 `train.py` 的命令行参数（不含解释器路径与脚本名）。
    /// 校验失败时返回中文错误信息，调用方应直接展示给用户、不要发起子进程。
    pub fn build_args(&self) -> Result<Vec<String>, String> {
        parse_blocks(&self.blocks)?;
        if self.ckpt_freq <= 0 {
            return Err("存档频率 ckpt-freq 必须为正".to_string());
        }
        if self.epochs <= 0 {
            return Err("总轮数 epochs 必须为正".to_string());
        }
        if self.batch_size <= 0 {
            return Err("批大小 batch-size 必须为正".to_string());
        }

        let mut cmd = Vec::new();

        if !self.train_file.trim().is_empty() {
            kv(&mut cmd, "--train-file", self.train_file.trim().to_string());
        }
        if !self.test_file.trim().is_empty() {
            kv(&mut cmd, "--test-file", self.test_file.trim().to_string());
        }
        if self.ndim3 {
            kv(&mut cmd, "--ndim", "3".to_string());
        }
        if self.no_mask {
            cmd.push("--no-mask".to_string());
        }

        kv(&mut cmd, "--task", self.task.as_arg().to_string());
        kv(
            &mut cmd,
            "--channel-order",
            self.channel_order.as_arg().to_string(),
        );
        cmd.push("--domain-size".to_string());
        cmd.push(self.domain_size_ly.to_string());
        cmd.push(self.domain_size_lx.to_string());

        kv(&mut cmd, "--val-split", self.val_split.to_string());
        kv(&mut cmd, "--rho", self.rho.to_string());
        kv(&mut cmd, "--nu", self.nu.to_string());
        kv(&mut cmd, "--div-weight", self.div_weight.to_string());
        kv(&mut cmd, "--ns-weight", self.ns_weight.to_string());
        kv(&mut cmd, "--smooth-weight", self.smooth_weight.to_string());
        kv(&mut cmd, "--diffusivity", self.diffusivity.to_string());
        kv(
            &mut cmd,
            "--advdiff-weight",
            self.advdiff_weight.to_string(),
        );
        if !self.t_geometry.trim().is_empty() {
            kv(
                &mut cmd,
                "--domain-mask",
                self.t_geometry.trim().to_string(),
            );
        }
        kv(
            &mut cmd,
            "--physics-warmup",
            self.physics_warmup.to_string(),
        );

        kv(&mut cmd, "--epochs", self.epochs.to_string());
        kv(&mut cmd, "--batch-size", self.batch_size.to_string());
        kv(&mut cmd, "--lr", self.lr.to_string());
        kv(&mut cmd, "--growth-rate", self.growth_rate.to_string());
        kv(&mut cmd, "--init-features", self.init_features.to_string());
        kv(&mut cmd, "--weight-decay", self.weight_decay.to_string());
        kv(&mut cmd, "--grad-clip", self.grad_clip.to_string());
        kv(&mut cmd, "--seed", self.seed.to_string());
        kv(&mut cmd, "--exp-name", self.exp_name.trim().to_string());
        kv(&mut cmd, "--ckpt-freq", self.ckpt_freq.to_string());
        kv(&mut cmd, "--plot-freq", self.plot_freq.to_string());

        // ---- 高级参数 ----
        kv(&mut cmd, "--blocks", self.blocks.trim().to_string());
        kv(&mut cmd, "--drop-rate", self.drop_rate.to_string());
        kv(&mut cmd, "--bn-size", self.bn_size.to_string());
        if self.bottleneck {
            cmd.push("--bottleneck".to_string());
        }
        kv(&mut cmd, "--num-heads", self.num_heads.to_string());
        if self.no_positional_encoding {
            cmd.push("--no-positional-encoding".to_string());
        }
        kv(
            &mut cmd,
            "--attn-max-tokens",
            self.attn_max_tokens.to_string(),
        );
        kv(&mut cmd, "--lr-patience", self.lr_patience.to_string());
        kv(
            &mut cmd,
            "--test-batch-size",
            self.test_batch_size.to_string(),
        );
        kv(&mut cmd, "--log-freq", self.log_freq.to_string());
        kv(&mut cmd, "--exp-dir", self.exp_dir.trim().to_string());
        kv(&mut cmd, "--data-dir", self.data_dir.trim().to_string());
        if let Some(n) = self.num_workers {
            kv(&mut cmd, "--num-workers", n.to_string());
        }
        if !self.input_key.trim().is_empty() {
            kv(&mut cmd, "--input-key", self.input_key.trim().to_string());
        }
        if self.prefer_sensor_input {
            cmd.push("--prefer-sensor-input".to_string());
        }
        kv(
            &mut cmd,
            "--physics-loss-scaling",
            self.physics_loss_scaling.as_arg().to_string(),
        );
        kv(
            &mut cmd,
            "--physics-in-physical-units",
            if self.physics_in_physical_units {
                "true"
            } else {
                "false"
            }
            .to_string(),
        );
        kv(&mut cmd, "--source-scale", self.source_scale.to_string());
        cmd.push("--velocity-channels".to_string());
        cmd.push(self.velocity_vx.to_string());
        cmd.push(self.velocity_vy.to_string());
        kv(&mut cmd, "--plot-fn", self.plot_fn.as_arg().to_string());
        if self.post {
            cmd.push("--post".to_string());
            if let Some(e) = self.ckpt_epoch {
                kv(&mut cmd, "--ckpt-epoch", e.to_string());
            }
        }

        Ok(cmd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.iter()
            .position(|a| a == flag)
            .map(|i| args[i + 1].as_str())
    }

    fn has_flag(args: &[String], flag: &str) -> bool {
        args.iter().any(|a| a == flag)
    }

    // -------------------------------------------------------------- parse_blocks

    #[test]
    fn parse_blocks_accepts_comma_separated() {
        assert_eq!(parse_blocks("3,6,3").unwrap(), vec![3, 6, 3]);
    }

    #[test]
    fn parse_blocks_accepts_space_separated() {
        assert_eq!(parse_blocks("3 6 3").unwrap(), vec![3, 6, 3]);
    }

    #[test]
    fn parse_blocks_rejects_even_length() {
        assert!(parse_blocks("3,6").is_err());
    }

    #[test]
    fn parse_blocks_rejects_non_integer() {
        assert!(parse_blocks("3,x,3").is_err());
    }

    #[test]
    fn parse_blocks_rejects_empty() {
        assert!(parse_blocks("").is_err());
        assert!(parse_blocks("   ").is_err());
    }

    #[test]
    fn parse_blocks_rejects_zero_or_negative() {
        assert!(parse_blocks("3,0,3").is_err());
        assert!(parse_blocks("3,-1,3").is_err());
    }

    // -------------------------------------------------------------- build_args 基本拼装

    #[test]
    fn build_args_includes_required_flags_with_defaults() {
        let p = TrainParams::default();
        let args = p.build_args().unwrap();
        assert_eq!(
            flag_value(&args, "--train-file"),
            Some("dataset/p_v_256.hdf5")
        );
        assert_eq!(flag_value(&args, "--task"), Some("flow"));
        assert_eq!(flag_value(&args, "--channel-order"), Some("pvv"));
        assert_eq!(flag_value(&args, "--epochs"), Some("200"));
        assert_eq!(flag_value(&args, "--blocks"), Some("3,6,3"));
    }

    #[test]
    fn build_args_omits_empty_optional_paths() {
        let p = TrainParams::default();
        let args = p.build_args().unwrap();
        assert!(!has_flag(&args, "--test-file"));
        assert!(!has_flag(&args, "--domain-mask"));
        assert!(!has_flag(&args, "--input-key"));
    }

    #[test]
    fn build_args_includes_optional_paths_when_set() {
        let p = TrainParams {
            test_file: "dataset/test.hdf5".to_string(),
            t_geometry: "dataset/mask.npy".to_string(),
            ..Default::default()
        };
        let args = p.build_args().unwrap();
        assert_eq!(flag_value(&args, "--test-file"), Some("dataset/test.hdf5"));
        assert_eq!(flag_value(&args, "--domain-mask"), Some("dataset/mask.npy"));
    }

    #[test]
    fn build_args_ndim3_and_no_mask_are_bare_flags() {
        let p = TrainParams {
            ndim3: true,
            no_mask: true,
            ..Default::default()
        };
        let args = p.build_args().unwrap();
        assert_eq!(flag_value(&args, "--ndim"), Some("3"));
        assert!(has_flag(&args, "--no-mask"));

        let p2 = TrainParams::default();
        let args2 = p2.build_args().unwrap();
        assert!(!has_flag(&args2, "--ndim"));
        assert!(!has_flag(&args2, "--no-mask"));
    }

    #[test]
    fn build_args_domain_size_emits_two_values_in_order() {
        let p = TrainParams {
            domain_size_ly: 2.5,
            domain_size_lx: 1.25,
            ..Default::default()
        };
        let args = p.build_args().unwrap();
        let i = args.iter().position(|a| a == "--domain-size").unwrap();
        assert_eq!(args[i + 1], "2.5");
        assert_eq!(args[i + 2], "1.25");
    }

    #[test]
    fn build_args_velocity_channels_emits_two_ints_in_order() {
        let p = TrainParams {
            velocity_vx: 2,
            velocity_vy: 3,
            ..Default::default()
        };
        let args = p.build_args().unwrap();
        let i = args
            .iter()
            .position(|a| a == "--velocity-channels")
            .unwrap();
        assert_eq!(args[i + 1], "2");
        assert_eq!(args[i + 2], "3");
    }

    #[test]
    fn build_args_post_ckpt_epoch_only_when_post_enabled() {
        let mut p = TrainParams {
            ckpt_epoch: Some(120),
            ..Default::default()
        };
        let args = p.build_args().unwrap();
        assert!(!has_flag(&args, "--post"));
        assert!(
            !has_flag(&args, "--ckpt-epoch"),
            "post 未勾选时不该带 --ckpt-epoch"
        );

        p.post = true;
        let args2 = p.build_args().unwrap();
        assert!(has_flag(&args2, "--post"));
        assert_eq!(flag_value(&args2, "--ckpt-epoch"), Some("120"));
    }

    #[test]
    fn build_args_bool_flags_are_flags_not_key_value() {
        let p = TrainParams {
            bottleneck: true,
            no_positional_encoding: true,
            prefer_sensor_input: true,
            ..Default::default()
        };
        let args = p.build_args().unwrap();
        assert!(has_flag(&args, "--bottleneck"));
        assert!(has_flag(&args, "--no-positional-encoding"));
        assert!(has_flag(&args, "--prefer-sensor-input"));
        // 不能出现 "--bottleneck" 后面跟着一个值这种 key-value 形式
        let i = args.iter().position(|a| a == "--bottleneck").unwrap();
        assert_ne!(args.get(i + 1).map(String::as_str), Some("true"));
    }

    #[test]
    fn build_args_rejects_invalid_blocks() {
        let p = TrainParams {
            blocks: "3,6".to_string(),
            ..Default::default()
        };
        assert!(p.build_args().is_err());
    }

    #[test]
    fn build_args_rejects_non_positive_ckpt_freq() {
        let p = TrainParams {
            ckpt_freq: 0,
            ..Default::default()
        };
        assert!(p.build_args().is_err());
    }

    #[test]
    fn build_args_num_workers_none_is_omitted() {
        let p = TrainParams::default();
        let args = p.build_args().unwrap();
        assert!(!has_flag(&args, "--num-workers"));
    }

    #[test]
    fn build_args_num_workers_some_is_emitted() {
        let p = TrainParams {
            num_workers: Some(4),
            ..Default::default()
        };
        let args = p.build_args().unwrap();
        assert_eq!(flag_value(&args, "--num-workers"), Some("4"));
    }
}
