//! 训练循环：全局梯度裁剪、Adam、自写 ReduceLROnPlateau。
//!
//! 移植自 `train.py`。验收基准是黄金数据第 7 层（`step/*`）：同一份权重、同一个
//! batch，前向 → 损失 → 裁剪后的梯度 → Adam 更新后的全部参数，逐元素对上。
//!
//! # 三个框架语义缺口
//!
//! 1. **梯度裁剪必须自己写**。Burn 的 `GradientClipping::Norm` 是**逐张量**裁剪
//!    （`clip_gradient` 的入参就是单个张量），而 PyTorch 的 `clip_grad_norm_` 算的是
//!    **所有参数拼起来的全局范数**，再用同一个系数缩放全部梯度。50 个参数张量下
//!    两者给出的更新方向完全不同，而且都不报错。见 [`clip_grad_norm`]。
//! 2. **Adam 的数学是一致的**（已核对 Burn 源码）：weight decay 走 L2-in-gradient
//!    （`grad += penalty * param`，与 PyTorch 的 `Adam` 而非 `AdamW` 一致），偏差
//!    修正也同式。但 **Burn 的默认 epsilon 不是 1e-8**，必须显式设置。
//! 3. **BatchNorm 的 running_var 更新有系统性差异**：PyTorch 用**无偏**方差
//!    （乘 n/(n-1)）更新 running_var，Burn 用有偏方差。归一化本身两边都用有偏方差，
//!    所以**训练时的前向、损失、梯度、参数更新全都不受影响**；只有推理时用到的
//!    running_var 会偏 1/(n-1)。本项目 n = batch*H*W 通常上千，偏差约 0.05%。
//!    见 [`BATCHNORM_RUNNING_VAR_NOTE`]。

use burn::module::{AutodiffModule, ModuleVisitor, ParamId};
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor};

/// BatchNorm running_var 的已知差异，写成常量是为了让它出现在文档与搜索结果里，
/// 而不是只躺在某条注释里被忘掉。
pub const BATCHNORM_RUNNING_VAR_NOTE: &str =
    "Burn 用有偏方差更新 running_var，PyTorch 用无偏（乘 n/(n-1)）。\
     归一化本身两边都用有偏方差，训练路径不受影响；只有推理用的 running_var 偏 1/(n-1)。";

// ------------------------------------------------------------ 全局梯度裁剪

/// 累加所有梯度的平方和。
struct NormAccumulator<'a> {
    grads: &'a GradientsParams,
    sum_sq: f64,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for NormAccumulator<'_> {
    fn visit_float<const D: usize>(&mut self, id: ParamId, _tensor: &Tensor<B, D>) {
        if let Some(g) = self.grads.get::<B::InnerBackend, D>(id) {
            self.sum_sq += g.powi_scalar(2).sum().into_scalar().elem::<f64>();
        }
    }
}

/// 用同一个系数缩放所有梯度。
struct Scaler<'a> {
    grads: &'a mut GradientsParams,
    coef: f64,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for Scaler<'_> {
    fn visit_float<const D: usize>(&mut self, id: ParamId, _tensor: &Tensor<B, D>) {
        if let Some(g) = self.grads.get::<B::InnerBackend, D>(id) {
            self.grads
                .register::<B::InnerBackend, D>(id, g.mul_scalar(self.coef));
        }
    }
}

/// PyTorch `torch.nn.utils.clip_grad_norm_` 的等价实现，返回裁剪前的全局范数。
///
/// ```text
/// total = sqrt(Σ_p ||grad_p||²)          // 所有参数拼起来算一个范数
/// coef  = max_norm / (total + 1e-6)
/// if coef < 1: 所有 grad *= coef
/// ```
///
/// **两个容易写漏的细节**：分母上的 `1e-6`，以及判据是 `coef < 1` 而不是
/// `total > max_norm`——两者在 `total` 恰好等于 `max_norm` 附近不等价。
pub fn clip_grad_norm<B, M>(
    model: &M,
    mut grads: GradientsParams,
    max_norm: f64,
) -> (GradientsParams, f64)
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let mut acc = NormAccumulator {
        grads: &grads,
        sum_sq: 0.0,
    };
    model.visit(&mut acc);
    let total = acc.sum_sq.sqrt();

    let coef = max_norm / (total + 1e-6);
    if coef < 1.0 {
        let mut scaler = Scaler {
            grads: &mut grads,
            coef,
        };
        model.visit(&mut scaler);
    }
    (grads, total)
}

// ------------------------------------------------------ ReduceLROnPlateau

/// 验证指标不再下降时把学习率减半。Burn 只提供 Noam / Cosine / Linear 三种调度器，
/// 这条要自己写。
///
/// 语义严格对齐 PyTorch 的
/// `ReduceLROnPlateau(mode='min', threshold_mode='rel', threshold=1e-4)`：
///
/// * 「有改善」的判据是 `metric < best * (1 - threshold)`，**相对**判据不是绝对差
/// * `num_bad_epochs > patience` 才降，所以 `patience=5` 实际是连续 6 次没改善
/// * 降完把坏计数清零，并进入 cooldown（本项目 cooldown=0，但保留语义）
/// * 新旧学习率之差不足 `1e-8` 时不动（PyTorch 的 `eps`），避免在 `min_lr` 处空转
#[derive(Debug, Clone)]
pub struct ReduceLrOnPlateau {
    pub factor: f64,
    pub patience: usize,
    pub threshold: f64,
    pub min_lr: f64,
    pub cooldown: usize,
    best: f64,
    num_bad_epochs: usize,
    cooldown_counter: usize,
}

impl ReduceLrOnPlateau {
    /// 与 `train.py` 一致的默认值：factor=0.5、threshold=1e-4、min_lr=1e-6。
    pub fn new(patience: usize) -> Self {
        Self {
            factor: 0.5,
            patience,
            threshold: 1e-4,
            min_lr: 1e-6,
            cooldown: 0,
            best: f64::INFINITY,
            num_bad_epochs: 0,
            cooldown_counter: 0,
        }
    }

    fn is_better(&self, metric: f64) -> bool {
        metric < self.best * (1.0 - self.threshold)
    }

    /// 喂一个 epoch 的验证指标，返回新的学习率。
    pub fn step(&mut self, metric: f64, lr: f64) -> f64 {
        if self.is_better(metric) {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            self.num_bad_epochs = 0;
        }

        if self.num_bad_epochs > self.patience {
            self.cooldown_counter = self.cooldown;
            self.num_bad_epochs = 0;
            let new_lr = (lr * self.factor).max(self.min_lr);
            // PyTorch 的 eps：差得太小就不动，否则在 min_lr 处会无谓地反复「降」
            if lr - new_lr > 1e-8 {
                return new_lr;
            }
        }
        lr
    }

    /// 当前的最优指标（尚未喂过数据时是 +inf）。
    pub fn best(&self) -> f64 {
        self.best
    }

    /// 连续没有改善的 epoch 数，用于日志。
    pub fn num_bad_epochs(&self) -> usize {
        self.num_bad_epochs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plateau_needs_more_than_patience_bad_epochs() {
        // patience=2 意味着连续 3 次没改善才降——`num_bad_epochs > patience`
        let mut s = ReduceLrOnPlateau::new(2);
        let mut lr = 1e-3;
        lr = s.step(1.0, lr); // 首次，best=1.0
        assert_eq!(lr, 1e-3);
        for i in 1..=2 {
            lr = s.step(1.0, lr);
            assert_eq!(lr, 1e-3, "第 {} 次没改善就降了，patience 没生效", i);
        }
        lr = s.step(1.0, lr);
        assert!(
            (lr - 5e-4).abs() < 1e-12,
            "第 3 次没改善应当减半，实得 {}",
            lr
        );
    }

    #[test]
    fn improvement_is_relative_not_absolute() {
        // threshold=1e-4 是**相对**的：`metric < best * (1 - 1e-4)`。
        //
        // **指标量级必须远离 1.0 才测得出区别**：best=1.0 时 `best*(1-1e-4)` 与
        // `best-1e-4` 恰好相等，两种实现给出同样的结果，测了等于没测。这里用
        // best=100（有量纲 RMSE 的典型量级）：相对判据的门槛是 99.99，绝对判据的
        // 门槛是 99.9999，99.995 落在两者之间。
        let mut s = ReduceLrOnPlateau::new(1);
        let mut lr = 1e-3;
        lr = s.step(100.0, lr); // best = 100
        lr = s.step(99.995, lr); // 相对判据：没改善 → bad=1（绝对判据会当成改善）
        assert_eq!(lr, 1e-3, "bad=1 不超过 patience=1，不该降");
        lr = s.step(99.995, lr); // 相对判据：bad=2 > 1 → 降
        assert!(
            (lr - 5e-4).abs() < 1e-12,
            "不到相对阈值的微小改善不该重置坏计数，实得 lr={}",
            lr
        );
    }

    #[test]
    fn patience_zero_reduces_on_first_bad_epoch() {
        // patience=0 时 `bad=1 > 0` 立刻成立，这是 PyTorch 的语义，不是 off-by-one
        let mut s = ReduceLrOnPlateau::new(0);
        let mut lr = 1e-3;
        lr = s.step(1.0, lr);
        lr = s.step(1.0, lr);
        assert!((lr - 5e-4).abs() < 1e-12, "实得 {}", lr);
    }

    #[test]
    fn improvement_resets_bad_counter() {
        let mut s = ReduceLrOnPlateau::new(1);
        let mut lr = 1e-3;
        lr = s.step(1.0, lr);
        lr = s.step(1.0, lr); // bad=1
        lr = s.step(0.5, lr); // 明确改善，bad 清零
        assert_eq!(s.num_bad_epochs(), 0);
        lr = s.step(0.5, lr); // bad=1
        assert_eq!(lr, 1e-3, "改善后坏计数没清零");
        lr = s.step(0.5, lr); // bad=2 > patience=1
        assert!((lr - 5e-4).abs() < 1e-12);
    }

    #[test]
    fn never_goes_below_min_lr() {
        let mut s = ReduceLrOnPlateau::new(0);
        let mut lr = 1e-6; // 已经在 min_lr 上
        s.step(1.0, lr);
        for _ in 0..5 {
            lr = s.step(1.0, lr);
        }
        assert!(lr >= 1e-6, "学习率跌破了 min_lr：{}", lr);
        assert!((lr - 1e-6).abs() < 1e-18, "min_lr 处应当停住，实得 {}", lr);
    }
}

// ---------------------------------------------------------------- checkpoint

use std::path::{Path, PathBuf};

use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};

/// 自描述 checkpoint 里的非权重部分。
///
/// 对应 Python 版 `save_checkpoint` 存的 `epoch` / `args` / `stats`：光有权重是不够的，
/// **推理侧要靠归一化统计量把网络输出还原成物理量纲**，靠超参重建同构的网络。
/// 少存任何一样，加载方就得靠猜，而猜错了不报错——只是预测场整体偏掉一个尺度。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CheckpointMeta {
    pub epoch: usize,
    /// 逐通道极值，与 `PhysicsLoss::y_min` / `y_max` 同一份
    pub y_min: Vec<f32>,
    pub y_max: Vec<f32>,
    /// 完整统计量的其余部分。**Python 版存的是整个 `stats` 字典**，只留 min/max 会让
    /// 推理侧走「复用训练时的统计量」那条路时不得不给 mean/std 编一个值。
    /// 用 `Option` 而不是默认零：缺失就是缺失，不能装作有。
    pub y_mean: Option<Vec<f32>>,
    pub y_std: Option<Vec<f32>>,
    pub n_samples: Option<usize>,
    pub stats_shape: Option<Vec<usize>>,
    /// 重建网络所需的超参，键值对形式。用扁平的字符串表而不是结构体，是为了让
    /// 将来新增参数不至于让旧 checkpoint 读不出来
    pub hyper: Vec<(String, String)>,
}

impl CheckpointMeta {
    /// 序列化成一份人能读、也能逐行解析的文本。
    ///
    /// 不引 serde_json 这类依赖：这里只有标量和短字符串，键值行足够了，
    /// 而且出问题时能直接 `cat` 出来看。
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str("format 3\n");
        out.push_str(&format!("epoch {}\n", self.epoch));
        out.push_str(&format!("y_min {}\n", join_floats(&self.y_min)));
        out.push_str(&format!("y_max {}\n", join_floats(&self.y_max)));
        if let Some(v) = &self.y_mean {
            out.push_str(&format!("y_mean {}\n", join_floats(v)));
        }
        if let Some(v) = &self.y_std {
            out.push_str(&format!("y_std {}\n", join_floats(v)));
        }
        if let Some(n) = self.n_samples {
            out.push_str(&format!("n_samples {}\n", n));
        }
        if let Some(sh) = &self.stats_shape {
            let joined = sh
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",");
            out.push_str(&format!("stats_shape {}\n", joined));
        }
        for (k, v) in &self.hyper {
            assert!(
                !k.contains(' ') && !k.contains('\n') && !v.contains('\n'),
                "超参键值不能含空格或换行：{:?} = {:?}",
                k,
                v
            );
            out.push_str(&format!("hp {} {}\n", k, v));
        }
        out
    }

    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut meta = Self::default();
        let mut saw_format = false;
        for (n, line) in text.lines().enumerate() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            let (key, rest) = line.split_once(' ').unwrap_or((line, ""));
            match key {
                "format" => {
                    saw_format = true;
                    if rest != "3" {
                        return Err(format!("不认识的 checkpoint 格式版本 {:?}", rest));
                    }
                }
                "epoch" => {
                    meta.epoch = rest
                        .parse()
                        .map_err(|e| format!("第 {} 行 epoch 解析失败：{}", n + 1, e))?
                }
                "y_min" => meta.y_min = parse_floats(rest, n + 1)?,
                "y_max" => meta.y_max = parse_floats(rest, n + 1)?,
                "y_mean" => meta.y_mean = Some(parse_floats(rest, n + 1)?),
                "y_std" => meta.y_std = Some(parse_floats(rest, n + 1)?),
                "n_samples" => {
                    meta.n_samples = Some(
                        rest.parse()
                            .map_err(|e| format!("第 {} 行 n_samples 解析失败：{}", n + 1, e))?,
                    )
                }
                "stats_shape" => {
                    let dims: Result<Vec<usize>, String> = rest
                        .split(',')
                        .filter(|t| !t.is_empty())
                        .map(|t| {
                            t.parse::<usize>()
                                .map_err(|e| format!("第 {} 行的 {:?} 不是整数：{}", n + 1, t, e))
                        })
                        .collect();
                    meta.stats_shape = Some(dims?);
                }
                "hp" => {
                    // 值可以是空串（`hp key` 或 `hp key ` 都算），不能因此报错——
                    // 写出去时行尾空格会被 `trim_end` 吃掉，读回来就没有那个分隔符了
                    let (k, v) = rest.split_once(' ').unwrap_or((rest, ""));
                    if k.is_empty() {
                        return Err(format!("第 {} 行的超参没有键名", n + 1));
                    }
                    meta.hyper.push((k.to_string(), v.to_string()));
                }
                other => return Err(format!("第 {} 行不认识的键 {:?}", n + 1, other)),
            }
        }
        if !saw_format {
            return Err("缺少 format 行，这不是本项目的 checkpoint".into());
        }
        if meta.y_min.len() != meta.y_max.len() {
            return Err(format!(
                "y_min 有 {} 项、y_max 有 {} 项，对不上",
                meta.y_min.len(),
                meta.y_max.len()
            ));
        }
        Ok(meta)
    }

    /// 由训练时用的统计量构造，**不丢字段**。
    pub fn from_stats(epoch: usize, stats: &OutputStats, hyper: Vec<(String, String)>) -> Self {
        Self {
            epoch,
            y_min: stats.y_min.clone(),
            y_max: stats.y_max.clone(),
            y_mean: Some(stats.y_mean.clone()),
            y_std: Some(stats.y_std.clone()),
            n_samples: Some(stats.n_samples),
            stats_shape: Some(stats.shape.clone()),
            hyper,
        }
    }

    /// 还原成 [`OutputStats`]，供推理侧直接塞进 `DatasetOptions.stats`。
    ///
    /// 字段不全时返回 `None` 而不是拿 0 顶上——**编出来的 mean/std 会让下游按错的
    /// 尺度还原，而且不报错**。
    pub fn stats(&self) -> Option<OutputStats> {
        Some(OutputStats {
            y_min: self.y_min.clone(),
            y_max: self.y_max.clone(),
            y_mean: self.y_mean.clone()?,
            y_std: self.y_std.clone()?,
            n_samples: self.n_samples?,
            shape: self.stats_shape.clone()?,
        })
    }

    /// 按键取超参。
    pub fn hp(&self, key: &str) -> Option<&str> {
        self.hyper
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

fn join_floats(v: &[f32]) -> String {
    v.iter()
        .map(|x| format!("{:?}", x))
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_floats(s: &str, line: usize) -> Result<Vec<f32>, String> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|t| {
            t.parse::<f32>()
                .map_err(|e| format!("第 {} 行的 {:?} 不是数字：{}", line, t, e))
        })
        .collect()
}

/// 保存 checkpoint：权重存 `<path>.mpk`，元信息存 `<path>.meta`。
///
/// 分两个文件而不是塞进一个：权重走 Burn 的二进制记录器（跨版本有它自己的兼容策略），
/// 元信息是纯文本，出问题时不用工具就能看。
pub fn save_checkpoint<B, M>(model: M, meta: &CheckpointMeta, path: &Path) -> Result<(), String>
where
    B: burn::tensor::backend::Backend,
    M: burn::module::Module<B>,
{
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file(path.to_path_buf(), &recorder)
        .map_err(|e| format!("写权重失败：{}", e))?;
    std::fs::write(meta_path(path), meta.to_text()).map_err(|e| format!("写元信息失败：{}", e))?;
    Ok(())
}

/// 读回 checkpoint。`model` 是一个结构相同、权重待覆盖的空壳。
pub fn load_checkpoint<B, M>(model: M, path: &Path) -> Result<(M, CheckpointMeta), String>
where
    B: burn::tensor::backend::Backend,
    M: burn::module::Module<B>,
{
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(path.to_path_buf(), &Default::default())
        .map_err(|e| format!("读权重失败：{}", e))?;
    let model = model.load_record(record);
    let text =
        std::fs::read_to_string(meta_path(path)).map_err(|e| format!("读元信息失败：{}", e))?;
    let meta = CheckpointMeta::from_text(&text)?;
    Ok((model, meta))
}

fn meta_path(path: &Path) -> PathBuf {
    let mut p = path.to_path_buf();
    let name = p
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    p.set_file_name(format!("{}.meta", name));
    p
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::{Linear, LinearConfig};

    type B = NdArray<f32>;

    fn sample_meta() -> CheckpointMeta {
        CheckpointMeta {
            epoch: 37,
            y_min: vec![-10.0, -1.0, -0.4],
            y_max: vec![30.0, 2.0, 0.6],
            y_mean: Some(vec![1.5, 0.25, 0.1]),
            y_std: Some(vec![7.0, 0.8, 0.2]),
            n_samples: Some(24),
            stats_shape: Some(vec![24, 3, 137, 137]),
            hyper: vec![
                ("channel_order".to_string(), "pvvt".to_string()),
                ("task".to_string(), "dispersion".to_string()),
                ("diffusivity".to_string(), "0.00032".to_string()),
            ],
        }
    }

    #[test]
    fn meta_round_trips() {
        let m = sample_meta();
        let back = CheckpointMeta::from_text(&m.to_text()).expect("解析失败");
        assert_eq!(back, m, "元信息往返不一致");
    }

    #[test]
    fn normalization_stats_survive_round_trip_exactly() {
        // 统计量是**推理侧把输出还原成物理量纲的唯一依据**，丢精度不会报错，
        // 只会让整张预测场偏一个尺度。所以往返必须逐位一致，不是「差不多」
        let m = CheckpointMeta {
            y_min: vec![-147.86183, 1.0e-8, -0.0],
            y_max: vec![147.86183, 3.4028235e38, 1.0 / 3.0],
            ..Default::default()
        };
        let back = CheckpointMeta::from_text(&m.to_text()).unwrap();
        assert_eq!(back.y_min, m.y_min, "y_min 往返丢了精度");
        assert_eq!(back.y_max, m.y_max, "y_max 往返丢了精度");
    }

    #[test]
    fn full_stats_round_trip_and_convert_back() {
        // Python 版存的是整个 stats 字典。只留 min/max 的话，推理侧要复用训练时的
        // 统计量就得给 mean/std 编个值——编出来的值会让下游按错的尺度还原且不报错
        let m = sample_meta();
        let back = CheckpointMeta::from_text(&m.to_text()).expect("解析失败");
        assert_eq!(back, m, "带完整统计量的元信息往返不一致");

        let stats = back.stats().expect("字段齐全时应当能还原成 OutputStats");
        assert_eq!(stats.y_mean, vec![1.5, 0.25, 0.1]);
        assert_eq!(stats.n_samples, 24);
        assert_eq!(stats.shape, vec![24, 3, 137, 137]);
    }

    #[test]
    fn partial_stats_yield_none_not_zeros() {
        // 老的、只有 min/max 的元信息仍然读得出来，但 stats() 必须返回 None——
        // **缺失就是缺失，不能拿 0 顶上**：编出来的 mean/std 会让下游按错的尺度还原。
        //
        // 四个可选字段要**逐个**验。只验其中一个的话，另外三个上的 `unwrap_or_default`
        // 退化不会被抓到（第一版就是这么漏的：只置空了 y_std，而变异改的是 y_mean）。
        let full = sample_meta();
        assert!(full.stats().is_some(), "字段齐全时应当能还原");

        type Blanker = (&'static str, fn(&mut CheckpointMeta));
        let blankers: [Blanker; 4] = [
            ("y_mean", |m| m.y_mean = None),
            ("y_std", |m| m.y_std = None),
            ("n_samples", |m| m.n_samples = None),
            ("stats_shape", |m| m.stats_shape = None),
        ];
        for (name, blank) in blankers {
            let mut m = sample_meta();
            blank(&mut m);
            let back = CheckpointMeta::from_text(&m.to_text()).unwrap();
            assert!(
                back.stats().is_none(),
                "缺了 {} 却还原出了一份 OutputStats——那份 mean/std 是编的",
                name
            );
            assert_eq!(back.y_min, full.y_min, "缺 {} 不该影响已有的极值", name);
        }
    }

    #[test]
    fn empty_hyper_value_round_trips() {
        // `hp {k} {v}` 里 v 为空串时，行尾空格会被 trim_end 吃掉，读回来就没有分隔符。
        // 早先这里会报「超参缺少值」——一个只在特定取值下才炸的解析缺陷
        let m = CheckpointMeta {
            hyper: vec![
                ("t_geometry".to_string(), String::new()),
                ("channel_order".to_string(), "pvv".to_string()),
                // 值里带空格也要过（内联 JSON 的几何 spec 就是这样）
                ("domain".to_string(), "{\"a\": 1, \"b\": 2}".to_string()),
            ],
            ..Default::default()
        };
        let back = CheckpointMeta::from_text(&m.to_text()).expect("空值超参解析失败");
        assert_eq!(back.hyper, m.hyper);
        assert_eq!(back.hp("t_geometry"), Some(""));
        assert_eq!(back.hp("domain"), Some("{\"a\": 1, \"b\": 2}"));
    }

    #[test]
    fn unknown_format_is_rejected() {
        let text = "format 99";
        assert!(
            CheckpointMeta::from_text(text).is_err(),
            "不认识的格式版本必须报错，不能当成兼容的读下去"
        );
        assert!(
            CheckpointMeta::from_text("epoch 1").is_err(),
            "缺 format 行的文件必须报错"
        );
    }

    #[test]
    fn mismatched_stats_length_is_rejected() {
        let bad = "format 3|y_min 1.0,2.0|y_max 3.0".replace('|', "\n");
        let err = CheckpointMeta::from_text(&bad).unwrap_err();
        assert!(err.contains("对不上"), "长度不符没被抓到：{}", err);
    }

    #[test]
    fn weights_survive_save_and_load() {
        let device = Default::default();
        let dir = std::env::temp_dir().join("eser_ckpt_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model");

        let saved: Linear<B> = LinearConfig::new(3, 4).init(&device);
        let want: Vec<f32> = saved.weight.val().into_data().to_vec().unwrap();
        save_checkpoint::<B, _>(saved, &sample_meta(), &path).expect("保存失败");

        // 另起一个**随机初始化**的空壳，读回来必须变成保存时的那份权重
        let shell: Linear<B> = LinearConfig::new(3, 4).init(&device);
        let before: Vec<f32> = shell.weight.val().into_data().to_vec().unwrap();
        assert_ne!(before, want, "两次随机初始化恰好相同，这个用例失去意义");

        let (loaded, meta) = load_checkpoint::<B, _>(shell, &path).expect("加载失败");
        let got: Vec<f32> = loaded.weight.val().into_data().to_vec().unwrap();
        assert_eq!(got, want, "读回来的权重与保存的不一致");
        assert_eq!(meta, sample_meta(), "元信息没跟着权重一起回来");

        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ---------------------------------------------------------------- 训练驱动

use crate::data::{CfdDataset, OutputStats};
use crate::physics::{LossParts, PhysicsLoss, PhysicsScaler, Task};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, Optimizer};

/// 训练循环的超参。对应 `train.py` 里从 `args` 取的那一组。
#[derive(Debug, Clone)]
pub struct FitConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub weight_decay: f32,
    /// <= 0 表示不裁剪
    pub grad_clip: f64,
    /// 学习率调度器。放整个而不是只放 `patience`：`threshold` / `factor` / `min_lr`
    /// 都是它的语义的一部分，散在两处迟早对不上
    pub scheduler: ReduceLrOnPlateau,
    /// 打乱训练样本顺序用的种子
    pub seed: u32,
    /// 每多少个 epoch 存一次 checkpoint；0 表示只在结束时存
    pub ckpt_freq: usize,
    pub ckpt_dir: Option<PathBuf>,
    /// 是否启用物理项的 EMA 量级归一化
    pub physics_scaling: bool,
    /// 验证集用的批大小。`None` 时复用 `batch_size`（对应 `--test-batch-size`）
    pub test_batch_size: Option<usize>,
    /// dispersion 任务下速度场在**输入通道**里的下标，对应 `--velocity-channels`
    pub velocity_channels: (usize, usize),
    /// 每轮往 stdout 打一行进度。
    ///
    /// **GUI 靠 stdout 实时滚日志**：不打的话跑起来日志区一片空白，用户没法判断是
    /// 在训练还是卡住了。格式对齐 `train.py` 的那一行。
    pub verbose: bool,
    /// 是否额外维护一份「验证 RMSE 最优」的 checkpoint（`best` / `best.meta`）。
    ///
    /// Python 每轮拿验证 RMSE 跟历史最优比，刷新就另存一份 `model_best.pth`，
    /// **那才是真正交付给推理侧的模型**——只留末轮权重等于把过拟合的那一版交出去。
    pub keep_best: bool,
    /// 额外写进 checkpoint 的超参。
    ///
    /// `nic` / `noc` / `ndim` 由 [`fit`] 从数据形状自动推出并写入，**不要在这里重复给**
    /// （给了会报错，而不是静默覆盖）。其余重建网络所需的超参（channel_order、blocks、
    /// growth_rate 之类）由调用方负责——`fit` 看不到它们。
    pub hyper: Vec<(String, String)>,
}

impl Default for FitConfig {
    /// **默认值逐条对齐 `args.py` 的 CLI 默认值**，不是随手填的：
    /// batch_size 直接改变 BatchNorm 的 batch 统计与梯度噪声，lr_patience 决定学习率
    /// 衰减的激进程度，两边对不上时训练轨迹会分岔，而这不会报任何错。
    fn default() -> Self {
        Self {
            epochs: 200,
            batch_size: 16,
            lr: 3e-3,
            weight_decay: 5e-4,
            grad_clip: 1.0,
            scheduler: ReduceLrOnPlateau::new(10),
            seed: 1,
            ckpt_freq: 25,
            ckpt_dir: None,
            physics_scaling: true,
            test_batch_size: Some(100),
            velocity_channels: (0, 1),
            verbose: true,
            keep_best: true,
            hyper: Vec::new(),
        }
    }
}

/// 一个 epoch 的记录，供日志与曲线图使用。
#[derive(Debug, Clone, PartialEq)]
pub struct EpochLog {
    pub epoch: usize,
    /// 训练集上各损失分项的均值
    pub train_loss: LossParts,
    pub train_r2: f64,
    /// 训练集 RMSE。少了它日志里的「train RMSE」只能拿验证值顶，读日志的人会被误导
    pub train_rmse: f64,
    pub val_r2: f64,
    /// 分通道 R²。诊断侧靠它看是压力还是速度在拖后腿，缺了那一列就是空的
    pub val_r2_per_channel: Vec<f64>,
    pub val_rmse: f64,
    pub lr: f64,
    /// 本 epoch 因为 NaN/Inf 被跳过的 batch 数
    pub skipped_batches: usize,
    /// 有量纲的物理残差 RMS（`div_rms` / `ns_rms` / `advdiff_rms`）。
    ///
    /// **这是「物理约束到底生效没有」的唯一监控**。Python 每个 epoch 都算并落盘，
    /// 诊断脚本只从这些数里判断；不产出的话诊断报告整列打 `--`，等于这条监控是断的。
    pub physics: crate::physics::PhysicsReport,
    /// 归一化后的目标是否跑出了 [0, 1]。
    ///
    /// **只 `eprintln!` 是不够的**：没人能对一条日志下断言，于是这条告警会在某次
    /// 重构里悄悄失效而没人发现。放进返回值，测试才咬得住它。
    pub normalization_out_of_range: bool,
}

/// 把若干个 batch 的分项损失求平均。
fn average_parts(acc: &[LossParts]) -> LossParts {
    let n = acc.len().max(1) as f64;
    let mut out = acc.first().copied().unwrap_or(ZERO_PARTS);
    out.base = acc.iter().map(|p| p.base).sum::<f64>() / n;
    out.div = acc.iter().map(|p| p.div).sum::<f64>() / n;
    out.ns = acc.iter().map(|p| p.ns).sum::<f64>() / n;
    out.advdiff = acc.iter().map(|p| p.advdiff).sum::<f64>() / n;
    out.smooth = acc.iter().map(|p| p.smooth).sum::<f64>() / n;
    out.total = acc.iter().map(|p| p.total).sum::<f64>() / n;
    out
}

const ZERO_PARTS: LossParts = LossParts {
    base: 0.0,
    div: 0.0,
    ns: 0.0,
    advdiff: 0.0,
    smooth: 0.0,
    total: 0.0,
    w_div: 0.0,
    w_ns: 0.0,
    w_advdiff: 0.0,
    w_smooth: 0.0,
    scale_div: 1.0,
    scale_ns: 1.0,
    scale_advdiff: 1.0,
    scale_smooth: 1.0,
};

/// 训练主循环。返回训练后的模型与逐 epoch 记录。
///
/// `loss` 里的 `epoch` / `scales` / `velocity` 每轮由本函数覆盖，其余字段（掩码、
/// 归一化统计量、权重、任务类型）由调用方给定。
///
/// # 为什么要两个前向闭包
///
/// 训练在 autodiff 后端上跑，**验证必须在内层后端上跑**（见 [`evaluate`]），两者的
/// 张量类型不同，一个闭包盖不住。函数体通常都是 `|m, x| m.forward(x)`，写两遍是
/// 类型系统的要求，不是重复劳动。
///
/// # 接上的三条告警 / 通路
///
/// * **归一化越界**：统计量若来自另一个文件，归一化后的目标会跑出 [0, 1]，而这
///   **不会报错**，只会让 R² 莫名其妙地差。这里逐 batch 检查并记进 [`EpochLog`]。
/// * **常数通道**：max == min 的通道归一化后恒为 0，等于白喂一个通道。
/// * **dispersion 的速度场按 batch 现取**：速度来自**当前 batch 的输入通道**，不是
///   一份全局张量。`FitConfig::velocity_channels` 给出通道下标，本函数逐 batch 切出来
///   塞进 `PhysicsLoss::velocity`。不接这条通路的话对流扩散残差会**静默恒为 0**，
///   损失照样有限、训练照样收敛，而阶段二等于完全没有物理约束。
// 8 个参数：模型、训练集、验证集、损失配置、训练配置、设备、两个前向闭包。
// 拆结构体只会把「调用方必须提供什么」藏起来，对照 train.py 时反而更难看
#[allow(clippy::too_many_arguments)]
pub fn fit<B, M, F, G>(
    mut model: M,
    train: &CfdDataset,
    val: Option<&CfdDataset>,
    loss: &PhysicsLoss<B>,
    cfg: &FitConfig,
    device: &B::Device,
    mut forward: F,
    mut forward_eval: G,
) -> Result<(M, Vec<EpochLog>), String>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + burn::module::Module<B> + Clone,
    F: FnMut(&M, Tensor<B, 4>) -> Tensor<B, 4>,
    G: FnMut(&M::InnerModule, Tensor<B::InnerBackend, 4>) -> Tensor<B::InnerBackend, 4>,
{
    if train.is_empty() {
        return Err("训练集为空，请检查 --val-split 与数据文件".into());
    }
    let n_channels = *train
        .y_shape()
        .get(1)
        .ok_or_else(|| "数据集的 y 形状缺通道维".to_string())?;

    // checkpoint 里必须带够「重建同构网络」所需的信息：`load_checkpoint` 要先有一个
    // 结构相同的空壳模型，调用方在**加载之前**就得知道通道数与维度。只存权重是装不回去的。
    let mut hyper = vec![
        (
            "nic".to_string(),
            train
                .x_shape()
                .get(1)
                .ok_or_else(|| "数据集的 x 形状缺通道维".to_string())?
                .to_string(),
        ),
        ("noc".to_string(), n_channels.to_string()),
        ("ndim".to_string(), (train.y_shape().len() - 2).to_string()),
    ];
    for (k, v) in &cfg.hyper {
        if hyper.iter().any(|(hk, _)| hk == k) {
            return Err(format!(
                "超参 {:?} 由 fit 从数据形状推出，不要在 FitConfig.hyper 里重复给",
                k
            ));
        }
        hyper.push((k.clone(), v.clone()));
    }

    // dispersion 任务必须能从输入通道里切出速度场。通道下标越界时**立刻报错**——
    // Python 侧 train.py:222-226 也是抛 ValueError，不是静默降级
    if loss.task == Task::Dispersion {
        let n_in = *train
            .x_shape()
            .get(1)
            .ok_or_else(|| "数据集的 x 形状缺通道维".to_string())?;
        let (ix, iy) = cfg.velocity_channels;
        if ix.max(iy) >= n_in {
            return Err(format!(
                "velocity_channels {:?} 超出输入通道数 {}，请检查数据集或该参数",
                cfg.velocity_channels, n_in
            ));
        }
    }

    for c in train.stats().degenerate_channels() {
        eprintln!(
            "[fit] 第 {} 个输出通道是常数场，归一化后恒为 0，等于白喂一个通道",
            c
        );
    }

    let mut optim = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        // Burn 的默认 epsilon 不是 1e-8，必须显式对齐 PyTorch
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(cfg.weight_decay)))
        .init::<B, M>();
    let mut sched = cfg.scheduler.clone();
    let mut scaler = PhysicsScaler::new(0.99, cfg.physics_scaling);
    let mut lr = cfg.lr;
    let mut logs = Vec::with_capacity(cfg.epochs);
    let mut warned_range = false;
    let mut out_of_range = false;
    let mut best_rmse = f64::INFINITY;

    for epoch in 1..=cfg.epochs {
        let mut metrics = crate::data::RunningMetrics::new(n_channels);
        let mut parts_acc: Vec<LossParts> = Vec::new();
        let mut skipped = 0usize;

        for sel in crate::data::batch_indices(
            train.len(),
            cfg.batch_size,
            Some(cfg.seed.wrapping_add(epoch as u32)),
            false,
        ) {
            let (x, y) = train.batch4::<B>(&sel, device)?;

            if !warned_range {
                let lo = y.clone().min().into_scalar().elem::<f32>();
                let hi = y.clone().max().into_scalar().elem::<f32>();
                if OutputStats::out_of_range(lo, hi) {
                    out_of_range = true;
                    eprintln!(
                        "[fit] 归一化后的目标跑出 [0,1]（实测 {:.4}~{:.4}）——统计量很可能来自\
                         另一个文件。这一条不修的话只会表现为 R² 莫名其妙地差",
                        lo, hi
                    );
                    warned_range = true;
                }
            }

            let mut step_cfg = loss.clone();
            step_cfg.epoch = epoch;
            step_cfg.total_epochs = cfg.epochs;
            step_cfg.scales = scaler.current();
            if step_cfg.task == Task::Dispersion {
                // 速度场来自**当前 batch 的输入通道**：两阶段架构里流场是阶段一的
                // 产物，作为输入喂进来，不在 output 里，也不参与 output 的归一化，
                // 所以切出来就是有量纲的物理速度
                let [b, _, h, w] = x.dims();
                let (ix, iy) = cfg.velocity_channels;
                step_cfg.velocity = Some((
                    x.clone().slice([0..b, ix..ix + 1, 0..h, 0..w]),
                    x.clone().slice([0..b, iy..iy + 1, 0..h, 0..w]),
                ));
            }
            let out = forward(&model, x);
            let (l, parts) = step_cfg.compute(out.clone(), y.clone());

            if !parts.total.is_finite() {
                eprintln!("[fit] epoch {}：损失出现 NaN/Inf，跳过该 batch", epoch);
                skipped += 1;
                continue;
            }
            scaler.update(&parts);
            parts_acc.push(parts);

            let grads = burn::optim::GradientsParams::from_grads(l.backward(), &model);
            let grads = if cfg.grad_clip > 0.0 {
                clip_grad_norm(&model, grads, cfg.grad_clip).0
            } else {
                grads
            };
            model = optim.step(lr, model, grads);
            metrics.update(out.inner(), y.inner(), loss.mask.clone().map(|m| m.inner()));
        }

        if parts_acc.is_empty() {
            return Err(format!(
                "epoch {} 的所有 batch 都被跳过了，训练无法继续",
                epoch
            ));
        }

        let (val_r2, val_r2_per_channel, val_rmse, physics) = match val {
            Some(v) => evaluate(
                &model,
                v,
                loss,
                cfg.test_batch_size.unwrap_or(cfg.batch_size),
                n_channels,
                device,
                &mut forward_eval,
                cfg.velocity_channels,
            )?,
            // 没有独立验证集时用训练集指标兜底，物理残差这一轮就不报
            None => (
                metrics.r2(),
                metrics.r2_per_channel(),
                metrics.rmse(),
                crate::physics::PhysicsReport::default(),
            ),
        };
        lr = sched.step(val_rmse, lr);

        logs.push(EpochLog {
            epoch,
            train_loss: average_parts(&parts_acc),
            train_r2: metrics.r2(),
            train_rmse: metrics.rmse(),
            val_r2,
            val_r2_per_channel,
            val_rmse,
            lr,
            skipped_batches: skipped,
            physics,
            normalization_out_of_range: out_of_range,
        });

        if cfg.verbose {
            print_epoch(cfg.epochs, logs.last().expect("刚 push 过"), loss.task);
        }

        if let Some(dir) = &cfg.ckpt_dir {
            let due = cfg.ckpt_freq > 0 && epoch.is_multiple_of(cfg.ckpt_freq);
            // **验证 RMSE 刷新就另存一份 best**：只留末轮权重等于把过拟合的那一版
            // 交给推理侧。Python 侧同理（train.py 的 model_best.pth）
            let improved = cfg.keep_best && val_rmse.is_finite() && val_rmse < best_rmse;
            if improved {
                best_rmse = val_rmse;
            }
            if due || epoch == cfg.epochs || improved {
                std::fs::create_dir_all(dir)
                    .map_err(|e| format!("建 checkpoint 目录失败：{}", e))?;
                let meta = CheckpointMeta::from_stats(epoch, train.stats(), hyper.clone());
                if due || epoch == cfg.epochs {
                    save_checkpoint::<B, M>(
                        model.clone(),
                        &meta,
                        &dir.join(format!("epoch{}", epoch)),
                    )?;
                }
                if improved {
                    save_checkpoint::<B, M>(model.clone(), &meta, &dir.join("best"))?;
                }
            }
        }
    }

    Ok((model, logs))
}

/// 3D 训练主循环。与 [`fit`] 结构相同，差别只在张量的阶数。
///
/// # 为什么单独写一份而不是把 [`fit`] 泛化
///
/// 数据源（`batch4` / `batch5`）、模型前向、以及损失所需的阶数三者各不相同，
/// 用 const 泛型串起来要给「5D 怎么变成损失要的 4D」造一整套 trait，换来的是
/// 更难对照 Python 原版——`train.py` 也是按 `--ndim` 分支走两条路。
///
/// # 深度维怎么处理
///
/// 物理损失只吃 4D，所以 5D 的输出与目标都先经 [`crate::physics::flatten_depth`]
/// 压成 (B*D, C, H, W)。**参与平均的元素集合没变**，损失值与在 5D 上算完全相同；
/// 掩码同理由调用方给成可广播的形状。
#[allow(clippy::too_many_arguments)]
pub fn fit_3d<B, M>(
    mut model: M,
    train: &CfdDataset,
    val: Option<&CfdDataset>,
    loss: &PhysicsLoss<B>,
    cfg: &FitConfig,
    device: &B::Device,
) -> Result<(M, Vec<EpochLog>), String>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + burn::module::Module<B> + Forward3d<B> + Clone,
    M::InnerModule: Forward3d<B::InnerBackend>,
{
    if train.is_empty() {
        return Err("训练集为空，请检查 val_split 与数据文件".into());
    }
    let n_channels = *train
        .y_shape()
        .get(1)
        .ok_or_else(|| "数据集的 y 形状缺通道维".to_string())?;
    if train.y_shape().len() != 5 {
        return Err(format!(
            "fit_3d 需要 (N, C, D, H, W) 的数据集，实得 {:?}",
            train.y_shape()
        ));
    }

    let mut hyper = vec![
        (
            "nic".to_string(),
            train
                .x_shape()
                .get(1)
                .ok_or_else(|| "数据集的 x 形状缺通道维".to_string())?
                .to_string(),
        ),
        ("noc".to_string(), n_channels.to_string()),
        ("ndim".to_string(), "3".to_string()),
    ];
    for (k, v) in &cfg.hyper {
        if hyper.iter().any(|(hk, _)| hk == k) {
            return Err(format!(
                "超参 {:?} 由 fit_3d 从数据形状推出，不要在 FitConfig.hyper 里重复给",
                k
            ));
        }
        hyper.push((k.clone(), v.clone()));
    }

    let mut optim = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(cfg.weight_decay)))
        .init::<B, M>();
    let mut sched = cfg.scheduler.clone();
    let mut scaler = PhysicsScaler::new(0.99, cfg.physics_scaling);
    let mut lr = cfg.lr;
    let mut logs = Vec::with_capacity(cfg.epochs);
    let mut best_rmse = f64::INFINITY;

    for epoch in 1..=cfg.epochs {
        let mut metrics = crate::data::RunningMetrics::new(n_channels);
        let mut parts_acc: Vec<LossParts> = Vec::new();
        let mut skipped = 0usize;

        for sel in crate::data::batch_indices(
            train.len(),
            cfg.batch_size,
            Some(cfg.seed.wrapping_add(epoch as u32)),
            false,
        ) {
            let (x, y) = train.batch5::<B>(&sel, device)?;
            let out = model.forward_3d(x);

            // 损失只吃 4D：先把深度维折进 batch 维
            let (out4, b, d) = crate::physics::flatten_depth(out.clone());
            let (y4, _, _) = crate::physics::flatten_depth(y.clone());
            let _ = (b, d);

            let mut step_cfg = loss.clone();
            step_cfg.epoch = epoch;
            step_cfg.total_epochs = cfg.epochs;
            step_cfg.scales = scaler.current();
            let (l, parts) = step_cfg.compute(out4, y4);

            if !parts.total.is_finite() {
                eprintln!("[fit_3d] epoch {}：损失出现 NaN/Inf，跳过该 batch", epoch);
                skipped += 1;
                continue;
            }
            scaler.update(&parts);
            parts_acc.push(parts);

            let grads = burn::optim::GradientsParams::from_grads(l.backward(), &model);
            let grads = if cfg.grad_clip > 0.0 {
                clip_grad_norm(&model, grads, cfg.grad_clip).0
            } else {
                grads
            };
            model = optim.step(lr, model, grads);
            metrics.update(out.inner(), y.inner(), None);
        }

        if parts_acc.is_empty() {
            return Err(format!(
                "epoch {} 的所有 batch 都被跳过了，训练无法继续",
                epoch
            ));
        }

        let (val_r2, val_r2_per_channel, val_rmse) = match val {
            Some(v) => eval_3d_metrics::<B::InnerBackend, M::InnerModule>(
                &model.valid(),
                v,
                cfg.test_batch_size.unwrap_or(cfg.batch_size),
                n_channels,
                device,
            )?,
            None => (metrics.r2(), metrics.r2_per_channel(), metrics.rmse()),
        };
        lr = sched.step(val_rmse, lr);

        logs.push(EpochLog {
            epoch,
            train_loss: average_parts(&parts_acc),
            train_r2: metrics.r2(),
            train_rmse: metrics.rmse(),
            val_r2,
            val_r2_per_channel,
            val_rmse,
            lr,
            skipped_batches: skipped,
            physics: crate::physics::PhysicsReport::default(),
            normalization_out_of_range: false,
        });

        if cfg.verbose {
            print_epoch(cfg.epochs, logs.last().expect("刚 push 过"), loss.task);
        }

        if let Some(dir) = &cfg.ckpt_dir {
            let due = cfg.ckpt_freq > 0 && epoch.is_multiple_of(cfg.ckpt_freq);
            let improved = cfg.keep_best && val_rmse.is_finite() && val_rmse < best_rmse;
            if improved {
                best_rmse = val_rmse;
            }
            if due || epoch == cfg.epochs || improved {
                std::fs::create_dir_all(dir)
                    .map_err(|e| format!("建 checkpoint 目录失败：{}", e))?;
                let meta = CheckpointMeta::from_stats(epoch, train.stats(), hyper.clone());
                if due || epoch == cfg.epochs {
                    save_checkpoint::<B, M>(
                        model.clone(),
                        &meta,
                        &dir.join(format!("epoch{}", epoch)),
                    )?;
                }
                if improved {
                    save_checkpoint::<B, M>(model.clone(), &meta, &dir.join("best"))?;
                }
            }
        }
    }

    Ok((model, logs))
}

/// 能在推理后端上做 3D 前向的模型。
///
/// 单独抽一个 trait 而不是再要一个闭包，是因为调用方几乎总是写
/// `|m, x| m.forward(x)`，两个闭包只会让签名更长而不会更清楚。
pub trait Forward3d<B: burn::tensor::backend::Backend> {
    fn forward_3d(&self, x: Tensor<B, 5>) -> Tensor<B, 5>;
}

impl<B: burn::tensor::backend::Backend> Forward3d<B> for crate::model3d::DenseEd3d<B> {
    fn forward_3d(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        self.forward(x)
    }
}

/// 在验证集上评估。**必须走推理后端**（`valid()`）。
///
/// Burn 不靠模块上的 train/eval 开关区分模式，靠**后端**：`BatchNorm::forward` 里写的是
/// `match B::ad_enabled() { true => forward_train, false => forward_inference }`，
/// Dropout 同理。拿 autodiff 后端的模型跑验证会有两个后果，而且都不报错：
///
/// 1. 验证指标用的是 **batch 统计量**而不是 running 统计量，系统性偏乐观；
/// 2. 更糟的是 `forward_train` 通过内部可变性**改写 running_mean / running_var**，
///    于是每个验证 batch 都把验证集的分布写进推理时要用的统计量——Python 的
///    `model.eval()` 下这两个量是冻结的。
///
/// 连带影响：`sched.step(val_rmse, lr)` 拿到的是另一个数，整条学习率轨迹会与 Python
/// 分岔；存进 checkpoint 的 running 统计量含验证集贡献，推理输出同样对不上。
#[allow(clippy::too_many_arguments)]
fn evaluate<B, M, G>(
    model: &M,
    val: &CfdDataset,
    loss: &PhysicsLoss<B>,
    batch_size: usize,
    n_channels: usize,
    device: &B::Device,
    forward_eval: &mut G,
    velocity_channels: (usize, usize),
) -> Result<(f64, Vec<f64>, f64, crate::physics::PhysicsReport), String>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    G: FnMut(&M::InnerModule, Tensor<B::InnerBackend, 4>) -> Tensor<B::InnerBackend, 4>,
{
    // `valid()` 把模型搬到内层（非 autodiff）后端：BatchNorm 与 Dropout 走推理分支，
    // running 统计量也不再被改写
    let eval_model = model.valid();
    let mut metrics = crate::data::RunningMetrics::new(n_channels);
    let mut report = crate::physics::PhysicsReport::default();
    for (i, sel) in crate::data::batch_indices(val.len(), batch_size, None, false)
        .into_iter()
        .enumerate()
    {
        let (x, y) = val.batch4::<B::InnerBackend>(&sel, device)?;
        let out = forward_eval(&eval_model, x.clone());
        if i == 0 {
            // 只在第一个 batch 上报，与 Python 的 evaluate 一致——这是监控不是指标，
            // 全量算一遍不值那个钱
            report = physics_report_for(loss, out.clone(), x, velocity_channels);
        }
        metrics.update(out, y, loss.mask.clone().map(|m| m.inner()));
    }
    Ok((
        metrics.r2(),
        metrics.r2_per_channel(),
        metrics.rmse(),
        report,
    ))
}

/// 在内层（非 autodiff）后端上算一次物理残差 RMS。
///
/// dispersion 任务同样要按 batch 从输入通道里切速度场，否则残差恒为 `None`，
/// 监控会一直显示「没算」而看不出是没接上还是真的没约束。
fn physics_report_for<B: AutodiffBackend>(
    loss: &PhysicsLoss<B>,
    out: Tensor<B::InnerBackend, 4>,
    x: Tensor<B::InnerBackend, 4>,
    velocity_channels: (usize, usize),
) -> crate::physics::PhysicsReport {
    let mut cfg = PhysicsLoss::<B::InnerBackend> {
        epoch: loss.epoch,
        total_epochs: loss.total_epochs,
        div_weight: loss.div_weight,
        ns_weight: loss.ns_weight,
        smooth_weight: loss.smooth_weight,
        advdiff_weight: loss.advdiff_weight,
        channel_order: loss.channel_order,
        task: loss.task,
        rho: loss.rho,
        nu: loss.nu,
        diffusivity: loss.diffusivity,
        dx: loss.dx,
        dy: loss.dy,
        physics_warmup: loss.physics_warmup,
        scales: loss.scales,
        mask: loss.mask.clone().map(|m| m.inner()),
        y_min: loss.y_min.clone(),
        y_max: loss.y_max.clone(),
        velocity: None,
        dcdt: None,
        source: None,
    };
    if cfg.task == Task::Dispersion {
        let [b, _, h, w] = x.dims();
        let (ix, iy) = velocity_channels;
        cfg.velocity = Some((
            x.clone().slice([0..b, ix..ix + 1, 0..h, 0..w]),
            x.slice([0..b, iy..iy + 1, 0..h, 0..w]),
        ));
    }
    cfg.report(out)
}

/// 一行进度，格式对齐 `train.py`。
///
/// 物理项**按任务分开报**：把 dispersion 的残差塞进 div/ns 的位置会让日志串味，
/// 事后翻记录的人分不清那个数到底是什么。
fn print_epoch(total: usize, l: &EpochLog, task: Task) {
    let phys = match task {
        Task::Dispersion => format!("advdiff={:.3e}", l.train_loss.advdiff),
        Task::Flow => format!("div={:.3e} ns={:.3e}", l.train_loss.div, l.train_loss.ns),
    };
    println!(
        "epoch {:>4}/{}｜train R²={:.5} RMSE={:.5}｜test R²={:.5} RMSE={:.5}｜\
base={:.4e} {} smooth={:.3e}｜lr={:.2e}",
        l.epoch,
        total,
        l.train_r2,
        l.train_rmse,
        l.val_r2,
        l.val_rmse,
        l.train_loss.base,
        phys,
        l.train_loss.smooth,
        l.lr
    );
}

/// 把逐 epoch 记录落成 `.txt` 数值文件，**诊断脚本只认这些文件**。
///
/// 对应 Python 的 `utils.plot.save_stats`：每个指标一个文件、每行一个 epoch。
/// 不写的话 `diagnose` 读不到任何东西，报告整列打 `--`——「物理约束到底生效没有」
/// 这条监控就是断的，而断了不会报错，只会看起来「没数据」。
///
/// 数值用 `{:.6e}`：诊断侧要解析，定长科学计数法比默认格式稳。
pub fn save_stats(dir: &Path, logs: &[EpochLog]) -> Result<(), String> {
    if logs.is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("建统计目录失败：{}", e))?;

    let series: Vec<(&str, Vec<f64>)> = vec![
        (
            "loss_train",
            logs.iter().map(|l| l.train_loss.total).collect(),
        ),
        (
            "loss_base",
            logs.iter().map(|l| l.train_loss.base).collect(),
        ),
        ("loss_div", logs.iter().map(|l| l.train_loss.div).collect()),
        ("loss_ns", logs.iter().map(|l| l.train_loss.ns).collect()),
        (
            "loss_advdiff",
            logs.iter().map(|l| l.train_loss.advdiff).collect(),
        ),
        (
            "loss_smooth",
            logs.iter().map(|l| l.train_loss.smooth).collect(),
        ),
        ("r2_train", logs.iter().map(|l| l.train_r2).collect()),
        ("rmse_train", logs.iter().map(|l| l.train_rmse).collect()),
        ("r2_test", logs.iter().map(|l| l.val_r2).collect()),
        ("rmse_test", logs.iter().map(|l| l.val_rmse).collect()),
        ("lr", logs.iter().map(|l| l.lr).collect()),
    ];
    for (name, values) in series {
        write_series(dir, name, &values)?;
    }

    write_test_series(
        dir,
        &logs
            .iter()
            .map(|l| l.val_r2_per_channel.clone())
            .collect::<Vec<_>>(),
        &logs.iter().map(|l| l.physics).collect::<Vec<_>>(),
    )
}

/// `r2_test_per_channel.txt` 与三个 `*_rms.txt`——训练与仅评估两条路**共用这一份实现**。
///
/// 不给 `--post` 另抄一份：上一次「同一个函数留了两份拷贝」的后果是测试盯住其中一份、
/// 生产跑的是另一份，改坏了也不红。
fn write_test_series(
    dir: &Path,
    per_channel: &[Vec<f64>],
    physics: &[crate::physics::PhysicsReport],
) -> Result<(), String> {
    // 分通道 R²：每行一个 epoch、每列一个通道，与 Python 的 np.savetxt 同形
    if per_channel.iter().any(|r| !r.is_empty()) {
        let mut text = String::new();
        for row in per_channel {
            let cells: Vec<String> = row.iter().map(|v| format!("{:.6e}", v)).collect();
            text.push_str(&cells.join(" "));
            text.push('\n');
        }
        std::fs::write(dir.join("r2_test_per_channel.txt"), text)
            .map_err(|e| format!("写 r2_test_per_channel.txt 失败：{}", e))?;
    }

    // 物理残差按 key 各归各位：把 dispersion 的残差硬塞进 div_rms 会让历史数据串味
    for (name, pick) in [("div_rms", 0usize), ("ns_rms", 1), ("advdiff_rms", 2)] {
        let vals: Vec<f64> = physics
            .iter()
            .filter_map(|p| match pick {
                0 => p.div_rms,
                1 => p.ns_rms,
                _ => p.advdiff_rms,
            })
            .collect();
        // 本次任务没算的项不落盘，而不是写一列 nan——诊断侧读到文件就当它有意义
        if !vals.is_empty() {
            write_series(dir, name, &vals)?;
        }
    }
    Ok(())
}

fn write_series(dir: &Path, name: &str, values: &[f64]) -> Result<(), String> {
    let mut text = String::new();
    for v in values {
        text.push_str(&format!("{:.6e}", v));
        text.push('\n');
    }
    std::fs::write(dir.join(format!("{}.txt", name)), text)
        .map_err(|e| format!("写 {}.txt 失败：{}", name, e))
}

/// 把逐 epoch 记录画成训练曲线（接的是 P7 的绘图层）。
pub fn plot_history(dir: &Path, logs: &[EpochLog]) -> Result<(), String> {
    if logs.is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("建图目录失败：{}", e))?;
    let train_r2: Vec<f32> = logs.iter().map(|l| l.train_r2 as f32).collect();
    let val_r2: Vec<f32> = logs.iter().map(|l| l.val_r2 as f32).collect();
    let rmse: Vec<f32> = logs.iter().map(|l| l.val_rmse as f32).collect();
    let total: Vec<f32> = logs.iter().map(|l| l.train_loss.total as f32).collect();

    crate::plot::plot_curves(
        &dir.join("r2.svg"),
        "R2",
        &[("train", &train_r2), ("val", &val_r2)],
    )
    .map_err(|e| format!("画 R² 曲线失败：{}", e))?;
    crate::plot::plot_curves(&dir.join("rmse.svg"), "RMSE", &[("val", &rmse)])
        .map_err(|e| format!("画 RMSE 曲线失败：{}", e))?;
    crate::plot::plot_curves(&dir.join("loss.svg"), "loss", &[("train", &total)])
        .map_err(|e| format!("画 loss 曲线失败：{}", e))?;
    Ok(())
}

// ================================================================ 仅评估（--post）

/// 单次评估的结果。
///
/// **刻意不复用 [`EpochLog`]**：`--post` 根本没跑训练，`train_r2` / `train_loss`
/// 没有真值。塞 0 进去，诊断侧会把「没训练过」读成「训练完 R² = 0」——这正是本项目
/// 反复栽跟头的那类静默错误。
#[derive(Debug, Clone)]
pub struct EvalReport {
    /// 被评估的 checkpoint 对应的 epoch
    pub epoch: usize,
    pub r2: f64,
    pub r2_per_channel: Vec<f64>,
    pub rmse: f64,
    pub physics: crate::physics::PhysicsReport,
}

/// 对已训练好的模型跑一次评估，不做任何参数更新。对应 Python `train.py` 的 `--post` 分支。
///
/// 内部直接复用训练循环里的 [`evaluate`]，所以 `valid()`（BatchNorm/Dropout 走推理分支、
/// running 统计量不被改写）这条语义是**同一份代码**保证的，不是这里重写一遍。
#[allow(clippy::too_many_arguments)]
pub fn evaluate_once<B, M, G>(
    model: &M,
    val: &CfdDataset,
    loss: &PhysicsLoss<B>,
    batch_size: usize,
    n_channels: usize,
    epoch: usize,
    velocity_channels: (usize, usize),
    device: &B::Device,
    mut forward_eval: G,
) -> Result<EvalReport, String>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    G: FnMut(&M::InnerModule, Tensor<B::InnerBackend, 4>) -> Tensor<B::InnerBackend, 4>,
{
    let (r2, r2_per_channel, rmse, physics) = evaluate(
        model,
        val,
        loss,
        batch_size,
        n_channels,
        device,
        &mut forward_eval,
        velocity_channels,
    )?;
    Ok(EvalReport {
        epoch,
        r2,
        r2_per_channel,
        rmse,
        physics,
    })
}

/// 3D 评估循环。`fit_3d` 每轮的验证与 [`evaluate_once_3d`]（`--post`）**共用这一份**。
///
/// 传进来的 `eval_model` 必须已经是 `valid()` 出来的推理侧模型：BatchNorm 走 running
/// 统计量、且不被改写。
fn eval_3d_metrics<BI, MI>(
    eval_model: &MI,
    v: &CfdDataset,
    batch_size: usize,
    n_channels: usize,
    device: &BI::Device,
) -> Result<(f64, Vec<f64>, f64), String>
where
    BI: burn::tensor::backend::Backend,
    MI: Forward3d<BI>,
{
    let mut m = crate::data::RunningMetrics::new(n_channels);
    for sel in crate::data::batch_indices(v.len(), batch_size, None, false) {
        let (x, y) = v.batch5::<BI>(&sel, device)?;
        m.update(eval_model.forward_3d(x), y, None);
    }
    Ok((m.r2(), m.r2_per_channel(), m.rmse()))
}

/// [`evaluate_once`] 的 3D 版本。
///
/// **`physics` 恒为空**，这不是偷懒：`fit_3d` 的每轮验证同样不算物理残差（3D 的残差
/// 要沿深度方向再补一套算子，见方案 §10）。这里跟着它保持一致，而不是编一个数出来。
pub fn evaluate_once_3d<B, M>(
    model: &M,
    val: &CfdDataset,
    batch_size: usize,
    n_channels: usize,
    epoch: usize,
    device: &B::Device,
) -> Result<EvalReport, String>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    M::InnerModule: Forward3d<B::InnerBackend>,
{
    let (r2, r2_per_channel, rmse) = eval_3d_metrics::<B::InnerBackend, M::InnerModule>(
        &model.valid(),
        val,
        batch_size,
        n_channels,
        device,
    )?;
    Ok(EvalReport {
        epoch,
        r2,
        r2_per_channel,
        rmse,
        physics: crate::physics::PhysicsReport::default(),
    })
}

/// 落盘仅评估的指标。
///
/// **只写测试侧的文件**（`r2_test` / `rmse_test` / `r2_test_per_channel` / `*_rms`），
/// 不碰 `r2_train` / `loss_*`——那些这次没算，写出来就是假的。诊断侧读不到训练曲线时
/// 会照实说「训练曲线缺失」，这比读到一列 0 强。
pub fn save_eval_stats(dir: &Path, rep: &EvalReport) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("建统计目录失败：{}", e))?;
    write_series(dir, "r2_test", &[rep.r2])?;
    write_series(dir, "rmse_test", &[rep.rmse])?;
    write_test_series(
        dir,
        std::slice::from_ref(&rep.r2_per_channel),
        &[rep.physics],
    )
}

#[cfg(test)]
mod fit_tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use rust_hdf5::H5File;

    use burn::module::Module;

    use crate::data::{CfdDataset, DatasetOptions, OutputStats};
    use crate::model::{DenseEd, DenseEdConfig};
    use crate::physics::ChannelOrder;

    type AD = Autodiff<NdArray<f32>>;

    /// 用 autodiff 的测试必须串行跑。
    ///
    /// **这是 Burn 0.18 的问题，不是测试写法问题**：`AutodiffServer` 的图内存管理
    /// （`burn-autodiff/src/runtime/server.rs`）在多个图并存时会把别的图的节点当成
    /// 不可达而回收掉，随后 `backward` 里 `self.steps.remove(&node_id).expect(...)`
    /// 直接 panic：`Node should have a step registered`。cargo 默认并行跑测试，
    /// 于是这类失败随机出现（实测 12 次里 2 次）。
    ///
    /// 用一把进程内的锁把它们串起来。锁中毒（前一个测试 panic）时继续用，
    /// 否则一个失败会级联成一片失败，掩盖真正的首因。
    static AUTODIFF_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn autodiff_guard() -> std::sync::MutexGuard<'static, ()> {
        AUTODIFF_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// 造一个极小的 HDF5：8 个样本、2 进 3 出、16x16。
    fn tiny_dataset(tag: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("eser_fit_{}.hdf5", tag));
        let _ = std::fs::remove_file(&path);
        let (n, ci, co, h, w) = (8usize, 2usize, 3usize, 16usize, 16usize);
        // 确定性数据：不用随机数，测试才可复现
        let x: Vec<f32> = (0..n * ci * h * w)
            .map(|i| ((i % 37) as f32 / 37.0) - 0.5)
            .collect();
        let y: Vec<f32> = (0..n * co * h * w)
            .map(|i| ((i % 53) as f32 / 53.0) * 4.0 - 2.0)
            .collect();
        let file = H5File::create(&path).expect("建不了临时 HDF5");
        file.new_dataset::<f32>()
            .shape([n, ci, h, w])
            .create("input")
            .expect("建不了 input")
            .write_raw(&x)
            .expect("写不进 input");
        file.new_dataset::<f32>()
            .shape([n, co, h, w])
            .create("output")
            .expect("建不了 output")
            .write_raw(&y)
            .expect("写不进 output");
        file.close().expect("关不掉文件");
        path
    }

    fn model_and_loss() -> (DenseEd<AD>, PhysicsLoss<AD>) {
        let device = Default::default();
        let mut cfg = DenseEdConfig::new(2, 3, vec![2, 2, 2]);
        cfg.growth_rate = 4;
        cfg.init_features = 8;
        cfg.output_size = Some([16, 16]);
        let model = cfg.init::<AD>(&device).expect("建模失败");
        let mask = Tensor::<AD, 4>::ones([1, 1, 16, 16], &device);
        let loss = PhysicsLoss::<AD> {
            channel_order: ChannelOrder::Pvv,
            dx: 0.05,
            dy: 0.05,
            mask: Some(mask),
            y_min: Some(vec![-2.0, -2.0, -2.0]),
            y_max: Some(vec![2.0, 2.0, 2.0]),
            ..Default::default()
        };
        (model, loss)
    }

    /// 把模型里所有参数（含 BatchNorm 的 running 统计量）拍平成一个向量。
    struct Snapshot<Bk: burn::tensor::backend::Backend> {
        values: Vec<f32>,
        _p: std::marker::PhantomData<Bk>,
    }

    impl<Bk: burn::tensor::backend::Backend> ModuleVisitor<Bk> for Snapshot<Bk> {
        fn visit_float<const D: usize>(&mut self, _id: ParamId, t: &Tensor<Bk, D>) {
            self.values
                .extend(t.clone().into_data().to_vec::<f32>().unwrap());
        }
    }

    fn snapshot(model: &DenseEd<AD>) -> Vec<f32> {
        let mut v = Snapshot::<AD> {
            values: Vec::new(),
            _p: std::marker::PhantomData,
        };
        model.visit(&mut v);
        v.values
    }

    /// 先 `valid()` 再拍照。
    ///
    /// **必须这么做才能比 running 统计量**：`RunningState` 把逐次更新先存在线程局部的
    /// map 里，只有 `sync()` 才合并进共享值，而 `Module::visit` **不触发 sync**
    /// （`into_record` 和 `valid` 才会）。直接 visit 拿到的值可能落后一次更新，
    /// 于是「跑过 evaluate 的那次」因为多同步了一轮而显得不一样——那是同步时机，
    /// 不是污染。两边都先 `valid()`，时机就被拉平了。
    fn snapshot_synced(model: &DenseEd<AD>) -> Vec<f32> {
        let inner = model.valid();
        let mut v = Snapshot::<<AD as AutodiffBackend>::InnerBackend> {
            values: Vec::new(),
            _p: std::marker::PhantomData,
        };
        inner.visit(&mut v);
        v.values
    }

    #[test]
    fn caller_cannot_silently_override_derived_hyperparams() {
        let _serial = autodiff_guard();
        // nic/noc/ndim 是从数据形状推出来的**事实**。调用方给了不一致的值时，
        // 静默覆盖会让 checkpoint 描述一个不存在的网络——必须报错
        let path = tiny_dataset("conflict");
        let ds = CfdDataset::open(&path, &DatasetOptions::default()).expect("打不开数据集");
        let (model, loss) = model_and_loss();
        let cfg = FitConfig {
            epochs: 1,
            batch_size: 8,
            hyper: vec![("noc".to_string(), "99".to_string())],
            ..Default::default()
        };
        let err = fit(
            model,
            &ds,
            None,
            &loss,
            &cfg,
            &Default::default(),
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )
        .expect_err("重复给 noc 应当报错");
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("noc"), "报错没点明具体是哪个键：{}", err);
    }

    #[test]
    fn optimizer_actually_updates_the_model() {
        let _serial = autodiff_guard();
        // 「loss 在降」不足以说明优化器起作用了：权重斜坡随 epoch 变、批次顺序每轮
        // 不同，冻结的模型照样能让平均损失上下波动。直接比参数才咬得住
        let path = tiny_dataset("update");
        let ds = CfdDataset::open(&path, &DatasetOptions::default()).expect("打不开数据集");
        let (model, loss) = model_and_loss();
        let before = snapshot(&model);
        let cfg = FitConfig {
            epochs: 1,
            batch_size: 8,
            ..Default::default()
        };
        let (trained, _) = fit(
            model,
            &ds,
            None,
            &loss,
            &cfg,
            &Default::default(),
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )
        .expect("训练失败");
        let after = snapshot(&trained);
        let _ = std::fs::remove_file(&path);

        assert_eq!(before.len(), after.len(), "参数数量变了");
        let moved = before
            .iter()
            .zip(after.iter())
            .filter(|(a, b)| (**a - **b).abs() > 0.0)
            .count();
        assert!(
            moved > before.len() / 2,
            "一轮之后只有 {} / {} 个参数动过，优化器没有真的接上",
            moved,
            before.len()
        );
    }

    #[test]
    fn scheduler_is_wired_to_the_validation_metric() {
        let _serial = autodiff_guard();
        // 光看 lr 有没有变不够稳（要不要降取决于训练动态）。这里改成**重放**：
        // 拿日志里的 RMSE 序列重新喂一遍调度器，必须逐位复现日志里的 lr 序列。
        // 调度器没接上时 lr 恒定，重放会立刻对不上
        let path = tiny_dataset("sched");
        let ds = CfdDataset::open(&path, &DatasetOptions::default()).expect("打不开数据集");
        let (model, loss) = model_and_loss();
        let cfg = FitConfig {
            epochs: 4,
            batch_size: 8,
            lr: 1e-3,
            // **靠训练动态制造「平台期」是不可靠的**：早先把 lr 调到让每轮改善落在
            // 相对阈值 1e-4 以下（实测 4.7e-5），只有 2 倍余量；而 ndarray 后端在并行
            // 负载下浮点归约顺序会抖，10 次里有 2 次改善量翻过阈值、调度器不触发，
            // 测试随机变红。改成把判据阈值直接设成 0.5——除非指标腰斩否则一律算
            // 「没改善」，调度器每轮必触发，与训练动态彻底解耦
            scheduler: ReduceLrOnPlateau {
                threshold: 0.5,
                ..ReduceLrOnPlateau::new(0)
            },
            ..Default::default()
        };
        let (_, logs) = fit(
            model,
            &ds,
            None,
            &loss,
            &cfg,
            &Default::default(),
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )
        .expect("训练失败");
        let _ = std::fs::remove_file(&path);

        let mut replay = cfg.scheduler.clone();
        let mut lr = cfg.lr;
        for l in &logs {
            lr = replay.step(l.val_rmse, lr);
            assert!(
                (l.lr - lr).abs() < 1e-15,
                "第 {} 轮日志里的 lr 是 {}，按 RMSE 序列重放应当是 {}",
                l.epoch,
                l.lr,
                lr
            );
        }
        assert!(
            logs.iter().any(|l| l.lr < cfg.lr),
            "这组数据下调度器一次都没触发（lr 全程 {}），本测试失去鉴别力，需要换数据",
            cfg.lr
        );
    }

    /// 造一个带速度通道的小数据集：输入 3 通道（vx, vy, 其它），输出 1 通道浓度。
    fn dispersion_dataset(tag: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("eser_disp_{}.hdf5", tag));
        let _ = std::fs::remove_file(&path);
        let (n, ci, co, h, w) = (8usize, 3usize, 1usize, 16usize, 16usize);
        // 速度场取非零常量：全 0 的速度会让平流项恒为 0，接没接上都看不出来
        let mut x = vec![0.0f32; n * ci * h * w];
        for i in 0..n {
            for c in 0..ci {
                for k in 0..h * w {
                    let v = match c {
                        0 => 0.4,  // vx
                        1 => -0.3, // vy
                        _ => ((k % 29) as f32 / 29.0) - 0.5,
                    };
                    x[(i * ci + c) * h * w + k] = v;
                }
            }
        }
        let y: Vec<f32> = (0..n * co * h * w)
            .map(|i| ((i % 41) as f32 / 41.0) * 2.0)
            .collect();
        let file = H5File::create(&path).expect("建不了临时 HDF5");
        file.new_dataset::<f32>()
            .shape([n, ci, h, w])
            .create("input")
            .expect("建不了 input")
            .write_raw(&x)
            .expect("写不进 input");
        file.new_dataset::<f32>()
            .shape([n, co, h, w])
            .create("output")
            .expect("建不了 output")
            .write_raw(&y)
            .expect("写不进 output");
        file.close().expect("关不掉文件");
        path
    }

    #[test]
    fn dispersion_gets_its_velocity_field_per_batch() {
        let _serial = autodiff_guard();
        let advdiff = run_dispersion_once("wired", (0, 1));
        assert!(advdiff > 0.0, "对流扩散残差应为正，实得 {}", advdiff);
    }

    /// 跑一轮 dispersion 训练，返回第一轮的对流扩散残差。
    ///
    /// `tag` 必须逐用例唯一：cargo 默认并行跑测试，共用临时文件名会互相删对方的
    /// 数据集，表现成随机失败
    fn run_dispersion_once(tag: &str, velocity_channels: (usize, usize)) -> f64 {
        // **这是整条流水线最容易静默失效的地方**：速度场来自当前 batch 的输入通道。
        // 接不上的话 advection_diffusion_loss 返回 None，advdiff 停在零占位符上，
        // 损失照样有限、训练照样收敛，而阶段二等于完全没有物理约束。
        let path = dispersion_dataset(tag);
        let ds = CfdDataset::open(&path, &DatasetOptions::default()).expect("打不开数据集");
        let device = Default::default();
        let mut mcfg = DenseEdConfig::new(3, 1, vec![2, 2, 2]);
        mcfg.growth_rate = 4;
        mcfg.init_features = 8;
        mcfg.output_size = Some([16, 16]);
        let model = mcfg.init::<AD>(&device).expect("建模失败");
        let loss = PhysicsLoss::<AD> {
            channel_order: ChannelOrder::C,
            task: Task::Dispersion,
            diffusivity: 1e-3,
            advdiff_weight: 0.1,
            dx: 0.05,
            dy: 0.05,
            mask: Some(Tensor::<AD, 4>::ones([1, 1, 16, 16], &device)),
            y_min: Some(vec![0.0]),
            y_max: Some(vec![2.0]),
            ..Default::default()
        };
        let cfg = FitConfig {
            epochs: 1,
            batch_size: 4,
            velocity_channels,
            ..Default::default()
        };
        let (_, logs) = fit(
            model,
            &ds,
            None,
            &loss,
            &cfg,
            &device,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )
        .expect("训练失败");
        let _ = std::fs::remove_file(&path);

        assert!(
            logs[0].train_loss.advdiff > 0.0,
            "对流扩散残差是 {}——速度场没有被逐 batch 喂进去，物理约束静默失效了",
            logs[0].train_loss.advdiff
        );
        assert_eq!(logs[0].train_loss.div, 0.0, "dispersion 不该再算散度");
        assert_eq!(logs[0].train_loss.ns, 0.0, "dispersion 不该再算动量");
        logs[0].train_loss.advdiff
    }

    #[test]
    fn dispersion_uses_the_configured_velocity_channels() {
        let _serial = autodiff_guard();
        // 「advdiff > 0」这一条太松：把 vx 取成 vy 也照样大于 0。
        // 换一对通道下标，残差必须跟着变——数据里 vx=0.4、vy=-0.3，方向都不同
        let a = run_dispersion_once("swapA", (0, 1));
        let b = run_dispersion_once("swapB", (1, 0));
        assert!(
            (a - b).abs() > 1e-9 * a.abs().max(1.0),
            "换了 velocity_channels 残差却没变（{} vs {}）——通道下标没有真的被用上",
            a,
            b
        );
    }

    #[test]
    fn out_of_range_velocity_channels_are_rejected() {
        let _serial = autodiff_guard();
        // Python 侧对越界的 --velocity-channels 抛 ValueError，不是静默降级
        let path = dispersion_dataset("oob");
        let ds = CfdDataset::open(&path, &DatasetOptions::default()).expect("打不开数据集");
        let device = Default::default();
        let mut mcfg = DenseEdConfig::new(3, 1, vec![2, 2, 2]);
        mcfg.growth_rate = 4;
        mcfg.init_features = 8;
        mcfg.output_size = Some([16, 16]);
        let model = mcfg.init::<AD>(&device).expect("建模失败");
        let loss = PhysicsLoss::<AD> {
            channel_order: ChannelOrder::C,
            task: Task::Dispersion,
            dx: 0.05,
            dy: 0.05,
            ..Default::default()
        };
        let cfg = FitConfig {
            epochs: 1,
            batch_size: 8,
            velocity_channels: (0, 7), // 输入只有 3 个通道
            ..Default::default()
        };
        let err = fit(
            model,
            &ds,
            None,
            &loss,
            &cfg,
            &device,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )
        .expect_err("越界的速度通道应当报错");
        let _ = std::fs::remove_file(&path);
        assert!(
            err.contains("velocity_channels"),
            "报错没点明是哪个参数：{}",
            err
        );
    }

    #[test]
    fn cloning_a_model_shares_its_running_statistics() {
        let _serial = autodiff_guard();
        // **Burn 的坑，记在这里**：`RunningState` 的字段是 `Arc<Mutex<..>>` 而结构体
        // `#[derive(Clone)]`，所以 `model.clone()` **共享** BatchNorm 的 running 统计量。
        // 想留一份「历史最优」的底稿时，克隆留不住这两个量——两边会一起被改。
        // 要独立副本得走 checkpoint 往返（`from_record` 会新建 Arc）。
        let device = Default::default();
        let (a, _) = model_and_loss();
        let b = a.clone();

        let before = snapshot_synced(&b);
        // 只在 a 上跑一次训练模式前向，b 不碰
        let x = Tensor::<AD, 4>::ones([2, 2, 16, 16], &device);
        let _ = a.forward(x);
        let after = snapshot_synced(&b);

        let moved = before
            .iter()
            .zip(after.iter())
            .filter(|(p, q)| (**p - **q).abs() > 0.0)
            .count();
        assert!(
            moved > 0,
            "克隆体的 running 统计量没跟着变——Burn 改成深拷贝了？那 model_best 那条路要重新评估"
        );
    }

    #[test]
    fn evaluate_does_not_touch_running_statistics() {
        let _serial = autodiff_guard();
        // **验证集绝不能改写 BatchNorm 的 running 统计量**。Burn 靠后端而不是
        // train()/eval() 开关区分模式：拿 autodiff 后端跑验证会走 forward_train，
        // 它通过内部可变性把验证 batch 的分布写进 running_mean/running_var，于是
        // 推理输出跟着变、checkpoint 里的统计量含验证集贡献，而全程不报错。
        //
        // 这里**直接验性质**，不去比较「带验证集训练」与「不带」两次的结果——
        // 那种比法要求训练逐位确定，而 ndarray 后端在并行负载下浮点归约顺序会抖，
        // 实测单独跑时 0/8170 差异、跟别的测试一起跑就有大片差异。差分法在这里
        // 天然不可靠，改成：跑一次 evaluate，前后快照必须一模一样。
        let path = tiny_dataset("evalpure");
        let device = Default::default();
        let opts_tr = DatasetOptions {
            indices: Some(vec![0, 1, 2, 3]),
            ..Default::default()
        };
        let train = CfdDataset::open(&path, &opts_tr).expect("打不开训练集");
        let opts_va = DatasetOptions {
            indices: Some(vec![4, 5, 6, 7]),
            stats: Some(train.stats().clone()),
            ..Default::default()
        };
        let val = CfdDataset::open(&path, &opts_va).expect("打不开验证集");

        let (model, loss) = model_and_loss();
        let init = snapshot_synced(&model);
        let cfg = FitConfig {
            epochs: 1,
            batch_size: 4,
            ..Default::default()
        };
        let (trained, _) = fit(
            model,
            &train,
            None, // 先不带验证集训练，把 running 统计量推离初值
            &loss,
            &cfg,
            &device,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )
        .expect("训练失败");

        let before = snapshot_synced(&trained);
        assert!(
            init.iter()
                .zip(before.iter())
                .any(|(a, b)| (a - b).abs() > 0.0),
            "训练一轮后模型状态没变，这个用例失去意义"
        );

        // 只跑评估，什么都不该动
        let (r2, _per_ch, rmse, _phys) = evaluate(
            &trained,
            &val,
            &loss,
            4,
            3,
            &device,
            &mut |m: &DenseEd<<AD as AutodiffBackend>::InnerBackend>, x| m.forward(x),
            (0, 1),
        )
        .expect("评估失败");
        assert!(r2.is_finite() && rmse.is_finite(), "评估指标应为有限值");

        let after = snapshot_synced(&trained);
        let moved = before
            .iter()
            .zip(after.iter())
            .filter(|(a, b)| (**a - **b).abs() > 0.0)
            .count();
        assert_eq!(
            moved,
            0,
            "跑完一次 evaluate 之后有 {} / {} 个参数或 running 统计量变了——\
             评估没有切到推理后端，验证集正在污染模型状态",
            moved,
            before.len()
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fit_config_defaults_match_the_python_cli() {
        // 这些值不是随手填的：batch_size 直接改变 BatchNorm 的 batch 统计与梯度噪声，
        // lr_patience 决定学习率衰减的激进程度。两边对不上时训练轨迹会分岔，
        // 而这不会报任何错——只会表现为「Rust 版好像调不出 Python 那个效果」
        let d = FitConfig::default();
        assert_eq!(d.batch_size, 16, "对应 args.py 的 --batch-size");
        assert_eq!(d.test_batch_size, Some(100), "对应 --test-batch-size");
        assert_eq!(d.scheduler.patience, 10, "对应 --lr-patience");
        assert_eq!(d.ckpt_freq, 25, "对应 --ckpt-freq");
        assert_eq!(d.lr, 3e-3, "对应 --lr");
        assert_eq!(d.weight_decay, 5e-4, "对应 --weight-decay");
        assert_eq!(d.grad_clip, 1.0, "对应 --grad-clip");
        assert_eq!(d.seed, 1, "对应 --seed");
        assert_eq!(d.velocity_channels, (0, 1), "对应 --velocity-channels");
        // 调度器的其余常量对应 train.py 里写死的那组
        assert_eq!(d.scheduler.factor, 0.5);
        assert_eq!(d.scheduler.threshold, 1e-4);
        assert_eq!(d.scheduler.min_lr, 1e-6);
    }

    #[test]
    fn best_checkpoint_tracks_the_lowest_validation_rmse() {
        // **只留末轮权重等于把过拟合的那一版交给推理侧**。Python 每当验证 RMSE
        // 刷新就另存一份 model_best.pth，那才是真正交付的模型
        let path = tiny_dataset("bestck");
        let dir = std::env::temp_dir().join("eser_best_ck");
        let _ = std::fs::remove_dir_all(&dir);
        let device = Default::default();

        let opts_tr = DatasetOptions {
            indices: Some(vec![0, 1, 2, 3]),
            ..Default::default()
        };
        let train = CfdDataset::open(&path, &opts_tr).expect("打不开训练集");
        let opts_va = DatasetOptions {
            indices: Some(vec![4, 5, 6, 7]),
            stats: Some(train.stats().clone()),
            ..Default::default()
        };
        let val = CfdDataset::open(&path, &opts_va).expect("打不开验证集");
        let (model, loss) = model_and_loss();
        let cfg = FitConfig {
            epochs: 3,
            batch_size: 4,
            ckpt_freq: 0,
            keep_best: true,
            ckpt_dir: Some(dir.clone()),
            ..Default::default()
        };
        let (_, logs) = fit(
            model,
            &train,
            Some(&val),
            &loss,
            &cfg,
            &device,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )
        .expect("训练失败");

        let meta_text =
            std::fs::read_to_string(dir.join("best.meta")).expect("best checkpoint 没写出来");
        let meta = CheckpointMeta::from_text(&meta_text).expect("元信息解析失败");

        // best 必须停在验证 RMSE 最小的那一轮，而不是最后一轮
        let best_epoch = logs
            .iter()
            .min_by(|a, b| a.val_rmse.total_cmp(&b.val_rmse))
            .map(|l| l.epoch)
            .unwrap();
        assert_eq!(
            meta.epoch,
            best_epoch,
            "best 存的是第 {} 轮，而验证 RMSE 最小的是第 {} 轮（序列 {:?}）",
            meta.epoch,
            best_epoch,
            logs.iter().map(|l| l.val_rmse).collect::<Vec<_>>()
        );

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stats_files_are_readable_by_the_diagnostics() {
        // **诊断脚本只认这些 .txt**。不写的话报告整列打 `--`，而那看起来像
        // 「还没训练」而不是「监控断了」。这里端到端验到 diagnose 真的读得出数
        let path = tiny_dataset("statsout");
        let dir = std::env::temp_dir().join("eser_stats_out");
        let _ = std::fs::remove_dir_all(&dir);
        let device = Default::default();

        let opts_tr = DatasetOptions {
            indices: Some(vec![0, 1, 2, 3]),
            ..Default::default()
        };
        let train = CfdDataset::open(&path, &opts_tr).expect("打不开训练集");
        let opts_va = DatasetOptions {
            indices: Some(vec![4, 5, 6, 7]),
            stats: Some(train.stats().clone()),
            ..Default::default()
        };
        let val = CfdDataset::open(&path, &opts_va).expect("打不开验证集");
        let (model, loss) = model_and_loss();
        let cfg = FitConfig {
            epochs: 2,
            batch_size: 4,
            ..Default::default()
        };
        let (_, logs) = fit(
            model,
            &train,
            Some(&val),
            &loss,
            &cfg,
            &device,
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )
        .expect("训练失败");

        // flow 任务必须产出散度/动量残差——这是「物理约束生效没有」的唯一监控
        assert!(
            logs[0].physics.div_rms.is_some(),
            "flow 任务没有产出散度残差，物理监控是断的"
        );
        assert!(
            logs[0].physics.ns_rms.is_some(),
            "flow 任务没有产出动量残差"
        );

        save_stats(&dir, &logs).expect("落盘失败");

        // 用诊断侧自己的读取函数验，而不是我这边再写一个解析器。
        // 注意按**元素总数**比而不是行数：load_series 复刻了 Python 的 atleast_2d
        // 语义，单列文件会被压成「一行 N 列」，那是 diagnose.py 的既有行为
        let n = logs.len();
        for (key, cols) in [
            ("r2_test", 1usize),
            ("rmse_test", 1),
            ("div_rms", 1),
            ("ns_rms", 1),
            ("r2_test_per_channel", 3),
        ] {
            let rows = crate::diagnose::load_series(&dir, key)
                .unwrap_or_else(|| panic!("诊断读不到 {}.txt", key));
            let total: usize = rows.iter().map(|r| r.len()).sum();
            assert_eq!(total, n * cols, "{} 的元素数与 epoch 数对不上", key);
        }
        // 本次是 flow 任务，advdiff 没算，就不该落盘（读到文件就当它有意义）
        assert!(
            crate::diagnose::load_series(&dir, "advdiff_rms").is_none(),
            "flow 任务不该写出 advdiff_rms.txt"
        );

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fit_3d_trains_and_keeps_the_depth_axis() {
        // 3D 训练此前**完全没有驱动**：`fit` 在类型上锁死 4D，`batch5` 全仓零调用点，
        // `flatten_depth` 只在 physics.rs 自己的单测里被用到——整条 3D 路径是断的
        use crate::model3d::{DenseEd3d, DenseEd3dConfig};

        let (n, ci, co, d, h, w) = (6usize, 2usize, 3usize, 2usize, 16usize, 16usize);
        let path = std::env::temp_dir().join("eser_fit3d.hdf5");
        let _ = std::fs::remove_file(&path);
        let x: Vec<f32> = (0..n * ci * d * h * w)
            .map(|i| ((i % 31) as f32 / 31.0) - 0.5)
            .collect();
        let y: Vec<f32> = (0..n * co * d * h * w)
            .map(|i| ((i % 47) as f32 / 47.0) * 3.0 - 1.5)
            .collect();
        {
            let file = H5File::create(&path).expect("建不了临时 HDF5");
            file.new_dataset::<f32>()
                .shape([n, ci, d, h, w])
                .create("input")
                .unwrap()
                .write_raw(&x)
                .unwrap();
            file.new_dataset::<f32>()
                .shape([n, co, d, h, w])
                .create("output")
                .unwrap()
                .write_raw(&y)
                .unwrap();
            file.close().unwrap();
        }

        let device = Default::default();
        let ds = CfdDataset::open(&path, &DatasetOptions::default()).expect("打不开数据集");
        assert_eq!(ds.y_shape().len(), 5, "这应当是个 5D 数据集");

        let mut mcfg = DenseEd3dConfig::new(ci, co, vec![2, 2, 2]);
        mcfg.growth_rate = 4;
        mcfg.init_features = 8;
        mcfg.output_size = Some([d, h, w]);
        let model: DenseEd3d<AD> = mcfg.init(&device).expect("建模失败");
        let loss = PhysicsLoss::<AD> {
            channel_order: ChannelOrder::Pvv,
            dx: 0.05,
            dy: 0.05,
            y_min: Some(vec![-1.5, -1.5, -1.5]),
            y_max: Some(vec![1.5, 1.5, 1.5]),
            ..Default::default()
        };
        let cfg = FitConfig {
            epochs: 2,
            batch_size: 3,
            ..Default::default()
        };
        let (trained, logs) = fit_3d(model, &ds, None, &loss, &cfg, &device).expect("3D 训练失败");
        let _ = std::fs::remove_file(&path);

        assert_eq!(logs.len(), 2);
        for l in &logs {
            assert!(
                l.train_loss.total.is_finite(),
                "第 {} 轮损失不是有限值",
                l.epoch
            );
            assert_eq!(l.skipped_batches, 0);
        }
        // 深度维必须原样保留
        let out =
            trained
                .valid()
                .forward(Tensor::<<AD as AutodiffBackend>::InnerBackend, 5>::zeros(
                    [1, ci, d, h, w],
                    &device,
                ));
        assert_eq!(out.dims(), [1, co, d, h, w], "3D 前向把深度维弄丢了");
    }

    #[test]
    fn fit_3d_rejects_a_4d_dataset() {
        // 拿 4D 数据集喂 3D 驱动必须报错，而不是靠 batch5 在底层崩出一句看不懂的话
        let path = tiny_dataset("fit3dwrong");
        let ds = CfdDataset::open(&path, &DatasetOptions::default()).expect("打不开数据集");
        let device = Default::default();
        use crate::model3d::{DenseEd3d, DenseEd3dConfig};
        let mut mcfg = DenseEd3dConfig::new(2, 3, vec![2, 2, 2]);
        mcfg.growth_rate = 4;
        mcfg.init_features = 8;
        let model: DenseEd3d<AD> = mcfg.init(&device).unwrap();
        let (_, loss) = model_and_loss();
        let err = fit_3d(
            model,
            &ds,
            None,
            &loss,
            &FitConfig {
                epochs: 1,
                ..Default::default()
            },
            &device,
        )
        .expect_err("4D 数据集喂给 fit_3d 应当报错");
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("fit_3d"), "报错没说清是谁的问题：{}", err);
    }

    #[test]
    fn fit_runs_and_reduces_loss() {
        let _serial = autodiff_guard();
        let path = tiny_dataset("basic");
        let opts = DatasetOptions::default();
        let ds = CfdDataset::open(&path, &opts).expect("打不开数据集");
        let (model, loss) = model_and_loss();
        let cfg = FitConfig {
            epochs: 3,
            batch_size: 4,
            lr: 3e-3,
            ..Default::default()
        };
        let (_, logs) = fit(
            model,
            &ds,
            None,
            &loss,
            &cfg,
            &Default::default(),
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )
        .expect("训练失败");
        let _ = std::fs::remove_file(&path);

        assert_eq!(logs.len(), 3);
        for l in &logs {
            assert!(
                l.train_loss.total.is_finite(),
                "第 {} 轮损失不是有限值",
                l.epoch
            );
            assert_eq!(l.skipped_batches, 0, "不该有 batch 被跳过");
        }
        assert!(
            logs[2].train_loss.total < logs[0].train_loss.total,
            "三轮之后损失没有下降：{:.6e} -> {:.6e}",
            logs[0].train_loss.total,
            logs[2].train_loss.total
        );
        assert!(
            !logs[0].normalization_out_of_range,
            "统计量是自己算的，不该报越界"
        );
    }

    #[test]
    fn stale_stats_are_reported_not_silently_swallowed() {
        let _serial = autodiff_guard();
        // 用**另一份数据的统计量**打开数据集：归一化后的目标会跑出 [0,1]。
        // 这在 Python 侧是一条 print，在这里必须是可断言的返回值——否则
        // 「用了另一个文件的统计量」只会表现为 R² 莫名其妙地差
        let path = tiny_dataset("stale");
        let opts = DatasetOptions {
            stats: Some(OutputStats {
                y_min: vec![-0.5, -0.5, -0.5],
                y_max: vec![0.5, 0.5, 0.5],
                y_mean: vec![0.0; 3],
                y_std: vec![1.0; 3],
                n_samples: 8,
                shape: vec![8, 3, 16, 16],
            }),
            ..Default::default()
        };
        let ds = CfdDataset::open(&path, &opts).expect("打不开数据集");
        let (model, loss) = model_and_loss();
        let cfg = FitConfig {
            epochs: 1,
            batch_size: 8,
            ..Default::default()
        };
        let (_, logs) = fit(
            model,
            &ds,
            None,
            &loss,
            &cfg,
            &Default::default(),
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )
        .expect("训练失败");
        let _ = std::fs::remove_file(&path);

        assert!(
            logs[0].normalization_out_of_range,
            "统计量过期导致的归一化越界没有被报出来"
        );
    }

    #[test]
    fn checkpoint_is_written_with_the_stats_used_for_training() {
        let _serial = autodiff_guard();
        let path = tiny_dataset("ckpt");
        let dir = std::env::temp_dir().join("eser_fit_ckpt_dir");
        let _ = std::fs::remove_dir_all(&dir);
        let opts = DatasetOptions::default();
        let ds = CfdDataset::open(&path, &opts).expect("打不开数据集");
        let want_min = ds.stats().y_min.clone();
        let (model, loss) = model_and_loss();
        let cfg = FitConfig {
            epochs: 2,
            batch_size: 8,
            ckpt_dir: Some(dir.clone()),
            ..Default::default()
        };
        let (_, _) = fit(
            model,
            &ds,
            None,
            &loss,
            &cfg,
            &Default::default(),
            |m, x| m.forward(x),
            |m, x| m.forward(x),
        )
        .expect("训练失败");

        let meta_text = std::fs::read_to_string(dir.join("epoch2.meta"))
            .expect("最后一轮的 checkpoint 元信息没写出来");
        let meta = CheckpointMeta::from_text(&meta_text).expect("元信息解析失败");
        assert_eq!(meta.epoch, 2);
        assert_eq!(
            meta.y_min, want_min,
            "checkpoint 里的统计量与训练时用的不是同一份——推理侧会按错的尺度还原"
        );

        // **只存权重是装不回去的**：load_checkpoint 要先有一个结构相同的空壳模型，
        // 调用方在加载之前就得知道通道数与维度
        assert_eq!(meta.hp("nic"), Some("2"), "checkpoint 没带输入通道数");
        assert_eq!(meta.hp("noc"), Some("3"), "checkpoint 没带输出通道数");
        assert_eq!(meta.hp("ndim"), Some("2"), "checkpoint 没带空间维度");
        assert!(
            meta.stats().is_some(),
            "checkpoint 里的统计量不完整，推理侧没法直接复用"
        );

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
