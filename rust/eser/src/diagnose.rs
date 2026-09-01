//! P6 诊断侧：`diagnose.py` 的等价物。
//!
//! 读 `experiments/` 下最新一次实验的 `args.txt` 与曲线数值，出一份中文体检报告。
//!
//! # 为什么配置检查排在指标前面
//!
//! 「训练集与测试集是同一个文件且 `val_split=0`」这类问题会让再漂亮的 R² 也失去
//! 意义——报告的是训练集内指标。这些**配置层面的硬伤比任何调参建议都值得先看**，
//! 所以 [`check_configuration`] 的产出单独作为 [`Report::problems`] 返回，而不是
//! 混在日志行里。
//!
//! # 与 Python 版的差异
//!
//! * **不输出 ANSI 颜色**。Rust 侧没有 CLI 入口（`Cargo.toml` 不在 P6 的改动范围），
//!   而 GUI 要的是结构不是转义序列——[`Report`] 保留了 `problems` 与逐行文本，上层
//!   要着色自己按前缀套即可。除颜色外的每一个字符都与 Python 版逐行一致，
//!   `tests/predict_golden.rs` 拿 Python 的真实输出对照。
//! * `args.txt` 读失败时的错误措辞不同（Python 打的是它自己的异常文本）。

use std::path::{Path, PathBuf};

use crate::predict::json::Json;

/// 报告宽度，与 Python 的 `'=' * 72` / `title.center(72)` 一致。
const WIDTH: usize = 72;

/// 一份诊断报告。`lines` 是**不含颜色**的逐行文本。
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub lines: Vec<String>,
    /// 配置层面的硬伤。非空时任何指标都不可信。
    pub problems: Vec<String>,
    /// 与 Python `main()` 的返回值一致：找不到实验目录是 1，其余 0。
    pub exit_code: i32,
}

impl Report {
    fn plain(&mut self, text: impl Into<String>) {
        self.lines.push(text.into());
    }

    /// `print('\n{}...')` 会先空一行，`print('...{}\n')` 会后空一行。
    fn header(&mut self, title: &str) {
        self.plain("");
        self.plain("=".repeat(WIDTH));
        self.plain(center(title, WIDTH));
        self.plain("=".repeat(WIDTH));
        self.plain("");
    }

    fn info(&mut self, msg: impl AsRef<str>) {
        self.plain(format!("[INFO] {}", msg.as_ref()));
    }

    fn ok(&mut self, msg: impl AsRef<str>) {
        self.plain(format!("[ OK ] {}", msg.as_ref()));
    }

    fn warn(&mut self, msg: impl AsRef<str>) {
        self.plain(format!("[WARN] {}", msg.as_ref()));
    }

    fn err(&mut self, msg: impl AsRef<str>) {
        self.plain(format!("[FAIL] {}", msg.as_ref()));
    }

    /// 一行指标。`thresholds` 是 `(good, okay)`，给了才判优秀/达标/欠佳。
    ///
    /// 缺值与 NaN 都打 `--`：一个「0.000000」会被当成真实的差指标，而缺失和差是
    /// 两回事。
    fn metric(
        &mut self,
        name: &str,
        value: Option<f64>,
        thresholds: Option<(f64, f64)>,
        unit: &str,
    ) {
        let Some(v) = value.filter(|v| !v.is_nan()) else {
            self.plain(format!("  {}: --", pad(name, 26)));
            return;
        };
        match thresholds {
            Some((good, okay)) => {
                let higher_better = good > okay;
                let hit = |thr: f64| if higher_better { v >= thr } else { v <= thr };
                let status = if hit(good) {
                    "✓ 优秀"
                } else if hit(okay) {
                    "~ 达标"
                } else {
                    "✗ 欠佳"
                };
                self.plain(format!(
                    "  {}: {:>12.6} {} [{}]",
                    pad(name, 26),
                    v,
                    pad(unit, 8),
                    status
                ));
            }
            // 注意行尾那个空格：Python 是 `'  {:26s}: {:12.6f} {}'`，unit 为空时
            // 就留一个尾随空格。逐行对照要一模一样，不能顺手 trim
            None => self.plain(format!("  {}: {:>12.6} {}", pad(name, 26), v, unit)),
        }
    }

    /// 整份报告的纯文本，行尾换行与 Python 的 `print` 一致。
    pub fn text(&self) -> String {
        let mut s = self.lines.join("\n");
        s.push('\n');
        s
    }
}

/// Python `str.center(width)`：宽度按**字符数**算（不是显示宽度），多出来的空格
/// 偏右——CJK 标题在两个实现里必须对齐到同一列。
fn center(s: &str, width: usize) -> String {
    let n = s.chars().count();
    if n >= width {
        return s.to_string();
    }
    let margin = width - n;
    let left = margin / 2;
    format!("{}{}{}", " ".repeat(left), s, " ".repeat(margin - left))
}

/// Python `'{:Ns}'`：按字符数右侧补空格，够长就原样返回。
fn pad(s: &str, width: usize) -> String {
    let n = s.chars().count();
    if n >= width {
        s.to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - n))
    }
}

/// 找最新一次实验：`experiments/` 下所有含 `args.txt` 的目录里，mtime 最大的那个。
///
/// **mtime 相同时的先后由目录遍历顺序决定**，与 Python 的 `rglob` 顺序不保证一致。
/// 实际使用里两次实验不会同一时刻结束，但测试构造 fixture 时要显式错开。
pub fn find_latest_experiment(root: &Path) -> Option<PathBuf> {
    if !root.exists() {
        return None;
    }
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    collect_experiments(root, &mut candidates);
    if candidates.is_empty() {
        return None;
    }
    // 稳定排序 + 取最后一个，与 Python `candidates.sort(...)` / `[-1]` 一致
    candidates.sort_by_key(|(mtime, _)| *mtime);
    candidates.pop().map(|(_, p)| p)
}

fn collect_experiments(dir: &Path, out: &mut Vec<(std::time::SystemTime, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut subdirs = Vec::new();
    let mut has_args = false;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path);
        } else if path.file_name().is_some_and(|n| n == "args.txt") {
            has_args = true;
        }
    }
    if has_args {
        let mtime = std::fs::metadata(dir)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        out.push((mtime, dir.to_path_buf()));
    }
    subdirs.sort();
    for sub in subdirs {
        collect_experiments(&sub, out);
    }
}

/// 读 `args.txt`。读不了就报一行警告并当成空配置，与 Python 一致——诊断本身
/// 不该因为一个坏文件直接崩掉。
pub fn load_config(path: &Path, report: &mut Report) -> Json {
    let empty = Json::Object(Vec::new());
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            report.warn(format!("读取 {} 失败: {}", path.display(), e));
            return empty;
        }
    };
    match Json::parse(&text) {
        Ok(v @ Json::Object(_)) => v,
        Ok(_) => {
            report.warn(format!("读取 {} 失败: 顶层不是 JSON 对象", path.display()));
            empty
        }
        Err(e) => {
            report.warn(format!("读取 {} 失败: {}", path.display(), e));
            empty
        }
    }
}

/// 读一份 `np.savetxt` 写出的数值文件，按行返回。文件不存在 / 空 / 解析失败都是 `None`。
///
/// **单列文件会被压成一行**——这不是随手写的，是 numpy 的语义：`np.loadtxt` 对
/// 单列文件返回 1D 数组，`np.atleast_2d` 再把它变成 `(1, N)`。于是
/// `r2_test_per_channel.txt` 在**单通道**任务（`--channel-order c`，即地铁站阶段二）
/// 里会被当成「一个 epoch、N 个通道」，诊断报告把每个 epoch 打成一个 ch。
/// 这是 Python 版的既有行为，此处照搬以保证两边报告逐行一致。
pub fn load_series(dir: &Path, name: &str) -> Option<Vec<Vec<f64>>> {
    let path = dir.join(format!("{}.txt", name));
    let text = std::fs::read_to_string(&path).ok()?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut row = Vec::new();
        for token in line.split_whitespace() {
            row.push(parse_np_float(token)?);
        }
        if !row.is_empty() {
            rows.push(row);
        }
    }
    if rows.is_empty() {
        return None;
    }
    if rows.iter().all(|r| r.len() == 1) && rows.len() > 1 {
        let flat: Vec<f64> = rows.into_iter().map(|r| r[0]).collect();
        return Some(vec![flat]);
    }
    Some(rows)
}

/// `np.loadtxt` 认得的数值字面量，包括 numpy 写出的 `nan` / `inf`。
fn parse_np_float(token: &str) -> Option<f64> {
    match token.to_ascii_lowercase().as_str() {
        "nan" | "+nan" => Some(f64::NAN),
        "-nan" => Some(f64::NAN),
        "inf" | "+inf" | "infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        _ => token.parse::<f64>().ok(),
    }
}

fn last_value(rows: &[Vec<f64>]) -> Option<f64> {
    rows.last()?.last().copied()
}

fn flat(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

/// 从曲线文件里读出来的指标。全为 `None` 时对应 Python 的空 `metrics` 字典。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metrics {
    pub final_r2: Option<f64>,
    pub final_rmse: Option<f64>,
    pub r2_series: Option<Vec<f64>>,
    pub r2_per_channel: Option<Vec<f64>>,
    pub div_rms: Option<f64>,
    pub ns_rms: Option<f64>,
    pub advdiff_rms: Option<f64>,
}

impl Metrics {
    pub fn is_empty(&self) -> bool {
        self.final_r2.is_none()
            && self.final_rmse.is_none()
            && self.r2_series.is_none()
            && self.r2_per_channel.is_none()
            && self.div_rms.is_none()
            && self.ns_rms.is_none()
            && self.advdiff_rms.is_none()
    }

    /// 从 `<run>/training/` 下的一堆 `.txt` 装配指标。
    pub fn load(train_dir: &Path) -> Self {
        let mut m = Metrics::default();
        if let Some(rows) = load_series(train_dir, "r2_test") {
            let series = flat(&rows);
            m.final_r2 = series.last().copied();
            m.r2_series = Some(series);
        }
        if let Some(rows) = load_series(train_dir, "rmse_test") {
            m.final_rmse = flat(&rows).last().copied();
        }
        if let Some(rows) = load_series(train_dir, "r2_test_per_channel") {
            m.r2_per_channel = rows.last().cloned();
        }
        m.div_rms = load_series(train_dir, "div_rms")
            .as_deref()
            .and_then(last_value);
        m.ns_rms = load_series(train_dir, "ns_rms")
            .as_deref()
            .and_then(last_value);
        m.advdiff_rms = load_series(train_dir, "advdiff_rms")
            .as_deref()
            .and_then(last_value);
        m
    }
}

/// 配置合理性检查 —— 比任何调参建议都更值得先看。返回硬伤列表。
pub fn check_configuration(cfg: &Json, report: &mut Report) -> Vec<String> {
    report.header("配置合理性检查");
    let mut problems: Vec<String> = Vec::new();

    let train_file = cfg.get("train_file");
    let test_file = cfg.get("test_file");
    let val_split = cfg.get("val_split");
    let truthy = |v: Option<&Json>| v.is_some_and(|v| v.truthy());
    let same_file = match (train_file, test_file) {
        (Some(a), Some(b)) => a.py_repr() == b.py_repr(),
        _ => false,
    };

    if truthy(train_file) && truthy(test_file) && same_file {
        if !truthy(val_split) {
            problems.push(
                "训练集与测试集是同一个文件且 val_split=0 —— 报告的 R² 是训练集内指标，\
                 不代表泛化能力。请设 --val-split 0.2 或提供独立测试集。"
                    .into(),
            );
        } else {
            report.ok(format!(
                "训练/测试同文件，但已按 val_split={} 切出独立验证集",
                py(val_split)
            ));
        }
    } else if truthy(train_file) && truthy(test_file) {
        report.ok("使用独立的测试集文件");
    } else if !truthy(val_split) {
        problems.push("未提供独立测试集且 val_split=0，评估指标不可信".into());
    } else {
        report.ok(format!("已按 val_split={} 切分验证集", py(val_split)));
    }

    let order = cfg.get("channel_order");
    if truthy(order) {
        let name = py(order);
        let mapping = match name.as_str() {
            "pvv" => "(P, Vx, Vy)",
            "vvp" => "(Vx, Vy, P)",
            "pvvt" => "(P, Vx, Vy, T)",
            "c" => "(C,)",
            _ => "未知",
        };
        report.warn(format!(
            "通道顺序设为 {} -> {}。请确认与数据集 output 的实际语义一致，\
             否则物理约束全部作用在错误通道上。",
            name, mapping
        ));
    } else {
        problems.push("args.txt 里没有 channel_order，可能是旧版实验产物".into());
    }

    let task_is_dispersion = cfg.get("task").and_then(Json::as_str) == Some("dispersion");
    if task_is_dispersion {
        report.ok("污染物输运任务（对流扩散方程）");
        // None 或恰好等于默认值 1e-5 都算「没设过」
        let default_diffusivity = match cfg.get("diffusivity") {
            None | Some(Json::Null) => true,
            Some(v) => v.as_f64() == Some(1e-5),
        };
        if default_diffusivity {
            report.warn(
                "diffusivity 仍是默认值 1e-5，若与真实污染物不符，\
                 对流扩散残差的绝对量级就没有参考意义",
            );
        }
        let vch = cfg.get("velocity_channels");
        if truthy(vch) {
            report.warn(format!(
                "速度场取自输入通道 {}。取错通道不会报错，只会让物理约束\
                 作用在错误的场上——请核对数据集的 input 通道顺序",
                py(vch)
            ));
        }
    } else if cfg.get("ns_weight").and_then(Json::as_f64) == Some(0.0) {
        report.ok("流场任务且 --ns-weight=0：只保留连续性方程，已跳过层流动量残差");
    }

    if truthy(cfg.get("no_mask")) {
        report.warn("本次实验关闭了计算域掩码（--no-mask），指标覆盖整幅网格");
    }

    // Python 是 `is False`：只有布尔 False 触发，0 不算
    if cfg.get("physics_in_physical_units").and_then(Json::as_bool) == Some(false) {
        report.warn(
            "物理损失运行在归一化空间（--physics-in-physical-units false），\
             散度/动量残差不代表真实物理量",
        );
    }

    let default_domain = cfg
        .get("domain_size")
        .and_then(Json::as_array)
        .is_some_and(|v| v.len() == 2 && v.iter().all(|e| e.as_f64() == Some(1.0)));
    if truthy(cfg.get("domain_size")) && default_domain {
        report.warn(
            "domain_size 仍是默认的 1x1，若真实计算域不是单位尺寸，\
             dx/dy 与雷诺数就对不上，物理残差的绝对数值没有意义",
        );
    }

    if problems.is_empty() {
        report.ok("未发现配置层面的硬伤");
    } else {
        for p in &problems {
            report.err(p);
        }
    }
    problems
}

fn py(v: Option<&Json>) -> String {
    v.map(Json::py_repr).unwrap_or_else(|| "None".into())
}

/// 模型质量诊断：核心指标 + 分通道 + 物理残差 + 建议。
pub fn diagnose_quality(m: &Metrics, report: &mut Report) {
    report.header("模型质量诊断");
    let (Some(r2), Some(rmse)) = (m.final_r2, m.final_rmse) else {
        report.warn("指标数据不足，无法诊断");
        return;
    };

    report.plain("核心指标:");
    report.metric("R² Score", Some(r2), Some((0.95, 0.90)), "");
    report.metric("RMSE (归一化尺度)", Some(rmse), Some((0.03, 0.05)), "");

    if let Some(per_channel) = &m.r2_per_channel {
        report.plain("");
        report.plain("分通道 R²:");
        for (c, value) in per_channel.iter().enumerate() {
            report.metric(&format!("  ch{}", c), Some(*value), Some((0.95, 0.90)), "");
        }
        // np.nanmax / np.nanmin：忽略 NaN
        let finite: Vec<f64> = per_channel
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .collect();
        if let (Some(hi), Some(lo)) = (
            finite.iter().copied().reduce(f64::max),
            finite.iter().copied().reduce(f64::min),
        ) {
            let spread = hi - lo;
            if spread > 0.15 {
                report.warn(format!(
                    "各通道 R² 差距 {:.3}，整体指标被好通道拉高了，\
                     建议单独看最差的那个通道",
                    spread
                ));
            }
        }
    }

    if let Some(div) = m.div_rms {
        report.plain("");
        report.plain("物理残差（有量纲）:");
        report.metric("  连续性 |div v| RMS", Some(div), None, "");
        report.metric("  动量方程残差 RMS", m.ns_rms, None, "");
    }

    report.plain("");
    report.plain("建议:");
    let mut suggestions: Vec<&str> = Vec::new();
    if r2 < 0.80 {
        suggestions.extend([
            "提高模型容量：--growth-rate 24~32 / --init-features 64",
            "训练轮数加到 400+",
            "确认输入信息是否足以决定输出（传感器点位是否太稀疏）",
        ]);
    } else if r2 < 0.90 {
        suggestions.extend(["再跑 300~400 轮", "微调学习率与 --lr-patience"]);
    } else if r2 < 0.95 {
        suggestions.push("已经不错；可尝试更大的 growth-rate 或更长训练");
    } else {
        report.ok("R² ≥ 0.95");
    }

    if rmse > 0.08 {
        suggestions.extend([
            "检查归一化统计量是否过期（删掉 dataset/*.stats.npz 重算）",
            "检查 --t-geometry 是否与实际管道几何匹配",
        ]);
    } else if rmse > 0.05 {
        suggestions.push("适当提高 --smooth-weight 或延长训练");
    }

    if let Some(series) = &m.r2_series {
        if series.len() > 3 {
            let tail = std::cmp::max(3, series.len() / 5);
            let recent = &series[series.len() - tail..];
            let hi = recent.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lo = recent.iter().copied().fold(f64::INFINITY, f64::min);
            if hi - lo < 1e-4 {
                suggestions.push("验证 R² 已经平台化，继续训练收益有限");
            }
            let peak = series.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let last = *series.last().expect("len > 3");
            if series.len() > 5 && last < peak - 0.02 {
                report.warn(format!(
                    "验证 R² 从峰值回落了 {:.3}，可能开始过拟合，\
                     请用 model_best.pth 而不是最后一轮",
                    peak - last
                ));
            }
        }
    }

    for s in suggestions {
        report.info(format!("• {}", s));
    }
}

/// 完整诊断流程，对应 `diagnose.main()`。
pub fn run(exp_root: &Path) -> Report {
    let mut report = Report::default();
    report.header("流场 / 污染物代理模型 · 训练诊断");

    let Some(exp_dir) = find_latest_experiment(exp_root) else {
        report.err(format!(
            "未找到实验目录（{}/ 下需要有含 args.txt 的文件夹）",
            exp_root.display()
        ));
        report.exit_code = 1;
        return report;
    };

    report.info(format!("实验目录: {}", exp_dir.display()));
    let train_dir = exp_dir.join("training");
    let pred_dir = train_dir.join("predictions");

    let cfg = load_config(&exp_dir.join("args.txt"), &mut report);

    report.header("实验配置");
    for key in [
        "ndim",
        "task",
        "train_file",
        "test_file",
        "val_split",
        "channel_order",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "blocks",
        "growth_rate",
        "init_features",
        "div_weight",
        "ns_weight",
        "advdiff_weight",
        "smooth_weight",
        "diffusivity",
        "velocity_channels",
        "t_geometry",
        "domain_size",
        "rho",
        "nu",
        "n_params",
        "training_time",
    ] {
        if let Some(v) = cfg.get(key) {
            report.plain(format!("  {}: {}", pad(key, 24), v.py_repr()));
        }
    }

    let problems = check_configuration(&cfg, &mut report);
    let metrics = Metrics::load(&train_dir);

    report.header("训练结果");
    if metrics.is_empty() {
        report.warn("没有找到 r2_test.txt / rmse_test.txt，训练可能未正常结束");
    } else {
        report.metric("最终 R²", metrics.final_r2, Some((0.95, 0.90)), "");
        report.metric("最终 RMSE", metrics.final_rmse, Some((0.03, 0.05)), "");
    }

    report.header("产物文件");
    let ckpt_dir = exp_dir.join("checkpoints");
    if ckpt_dir.exists() {
        let ckpts = sorted_glob(&ckpt_dir, |name| name.ends_with(".pth"));
        report.info(format!("checkpoint 数量: {}", ckpts.len()));
        let best = ckpt_dir.join("model_best.pth");
        if best.exists() {
            report.ok(format!("最优权重: {}", best.display()));
        } else if let Some(last) = ckpts.last() {
            report.info(format!("最新权重: {}", last.display()));
        }
    }
    if pred_dir.exists() {
        // Python 的 glob 是 `pred_*.*`：前缀加一个点号后缀
        let images = sorted_glob(&pred_dir, |name| {
            name.starts_with("pred_") && name[5..].contains('.')
        });
        report.info(format!("预测对比图: {} 张", images.len()));
        if let Some(last) = images.last() {
            report.ok(format!("最新一张: {}", last.display()));
        }
    }

    if !metrics.is_empty() {
        diagnose_quality(&metrics, &mut report);
    }

    report.header("下一步");
    if !problems.is_empty() {
        report.err("先解决上面的配置问题，否则任何指标都不可信");
    }
    report.info("1. 打开预测对比图：第 1 行真值 / 第 2 行预测 / 第 3 行残差");
    report.info("   残差行用以 0 为中心的对称色阶，接近纯白即拟合良好");
    report.info(format!(
        "2. 曲线图：{}/r2.svg, {}/rmse.svg, {}/r2_per_channel.svg",
        train_dir.display(),
        train_dir.display(),
        train_dir.display()
    ));
    if cfg.get("task").and_then(Json::as_str) == Some("dispersion") {
        report.info(format!(
            "3. 物理残差曲线：{}/advdiff_rms.svg",
            train_dir.display()
        ));
    } else {
        report.info(format!(
            "3. 物理残差曲线：{}/div_rms.svg, {}/ns_rms.svg",
            train_dir.display(),
            train_dir.display()
        ));
    }
    report.info(format!(
        "4. 离线推理：python surrogate_predict.py --ckpt-file {}/model_best.pth \
         --data-file <你的数据>",
        ckpt_dir.display()
    ));

    report.problems = problems;
    report
}

/// 目录里符合条件的文件，按 Python `sorted(Path.glob(...))` 的顺序。
///
/// Windows 上 `PurePath` 的比较是大小写不敏感的（比的是整串小写化后的路径），
/// 所以这里按小写路径排序而不是按原始字节。
fn sorted_glob(dir: &Path, pred: impl Fn(&str) -> bool) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.file_name().and_then(|n| n.to_str()).is_some_and(&pred))
        .collect();
    out.sort_by_key(|p| p.to_string_lossy().to_lowercase());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(text: &str) -> Json {
        Json::parse(text).unwrap()
    }

    fn run_check(text: &str) -> (Vec<String>, String) {
        let mut r = Report::default();
        let problems = check_configuration(&cfg(text), &mut r);
        (problems, r.text())
    }

    #[test]
    fn same_train_and_test_file_without_split_is_a_hard_problem() {
        let (problems, _) = run_check(
            r#"{"train_file": "a.hdf5", "test_file": "a.hdf5", "val_split": 0,
                "channel_order": "pvv"}"#,
        );
        assert_eq!(problems.len(), 1, "{:?}", problems);
        assert!(problems[0].contains("训练集与测试集是同一个文件"));

        // 同文件但切了验证集就不是硬伤
        let (problems, text) = run_check(
            r#"{"train_file": "a.hdf5", "test_file": "a.hdf5", "val_split": 0.2,
                "channel_order": "pvv"}"#,
        );
        assert!(problems.is_empty(), "{:?}", problems);
        assert!(
            text.contains("已按 val_split=0.2 切出独立验证集"),
            "{}",
            text
        );
    }

    #[test]
    fn missing_channel_order_is_a_problem_and_present_one_warns() {
        let (problems, _) = run_check(r#"{"train_file": "a", "test_file": "b"}"#);
        assert!(problems.iter().any(|p| p.contains("没有 channel_order")));

        let (_, text) = run_check(r#"{"train_file": "a", "test_file": "b", "channel_order": "c"}"#);
        assert!(text.contains("通道顺序设为 c -> (C,)"), "{}", text);
    }

    #[test]
    fn dispersion_defaults_are_flagged() {
        let (_, text) = run_check(
            r#"{"train_file": "a", "test_file": "b", "channel_order": "c",
                "task": "dispersion", "diffusivity": 1e-05, "velocity_channels": [0, 1]}"#,
        );
        assert!(text.contains("污染物输运任务"), "{}", text);
        assert!(text.contains("diffusivity 仍是默认值 1e-5"), "{}", text);
        assert!(text.contains("速度场取自输入通道 [0, 1]"), "{}", text);

        // 改过扩散系数就不该再告警
        let (_, text) = run_check(
            r#"{"train_file": "a", "test_file": "b", "channel_order": "c",
                "task": "dispersion", "diffusivity": 0.002}"#,
        );
        assert!(!text.contains("diffusivity 仍是默认值"), "{}", text);
    }

    #[test]
    fn normalized_space_physics_is_flagged_only_for_boolean_false() {
        let base = r#""train_file": "a", "test_file": "b", "channel_order": "pvv""#;
        let (_, text) = run_check(&format!(
            r#"{{{}, "physics_in_physical_units": false}}"#,
            base
        ));
        assert!(text.contains("物理损失运行在归一化空间"), "{}", text);

        // Python 用的是 `is False`，0 不触发
        let (_, text) = run_check(&format!(r#"{{{}, "physics_in_physical_units": 0}}"#, base));
        assert!(!text.contains("物理损失运行在归一化空间"), "{}", text);
    }

    #[test]
    fn default_domain_size_is_flagged() {
        let base = r#""train_file": "a", "test_file": "b", "channel_order": "pvv""#;
        let (_, text) = run_check(&format!(r#"{{{}, "domain_size": [1.0, 1.0]}}"#, base));
        assert!(text.contains("domain_size 仍是默认的 1x1"), "{}", text);
        let (_, text) = run_check(&format!(r#"{{{}, "domain_size": [12.0, 40.0]}}"#, base));
        assert!(!text.contains("domain_size 仍是默认的 1x1"), "{}", text);
    }

    #[test]
    fn metric_formatting_matches_python_widths() {
        let mut r = Report::default();
        r.metric("R² Score", Some(0.97), Some((0.95, 0.90)), "");
        r.metric("RMSE (归一化尺度)", Some(0.06), Some((0.03, 0.05)), "");
        r.metric("缺值", None, Some((0.95, 0.90)), "");
        r.metric("  连续性 |div v| RMS", Some(1.5), None, "");
        assert_eq!(
            r.lines,
            vec![
                // 逐字来自 Python 的 `'  {:26s}: {:12.6f} {:8s} [{}]'.format(...)`：
                // 宽度按**字符数**算，CJK 不折算成两列，行尾那个空格也要留着
                "  R² Score                  :     0.970000          [✓ 优秀]".to_string(),
                "  RMSE (归一化尺度)              :     0.060000          [✗ 欠佳]".to_string(),
                "  缺值                        : --".to_string(),
                "    连续性 |div v| RMS         :     1.500000 ".to_string(),
            ]
        );
    }

    #[test]
    fn header_centering_matches_python() {
        let mut r = Report::default();
        r.header("实验配置");
        assert_eq!(r.lines[2].chars().count(), WIDTH);
        assert_eq!(r.lines[2].trim(), "实验配置");
        // 多出来的空格偏右：左 32 右 32 时长度 4 -> 34/34
        assert_eq!(r.lines[2].chars().take_while(|c| *c == ' ').count(), 34);
    }

    #[test]
    fn single_column_series_is_squeezed_like_numpy() {
        let dir = std::env::temp_dir().join(format!("eser_p6_series_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("one.txt"), "0.1\n0.2\n0.3\n").unwrap();
        std::fs::write(dir.join("two.txt"), "0.1 0.9\n0.2 0.8\n").unwrap();
        std::fs::write(dir.join("empty.txt"), "\n").unwrap();

        // 单列 -> numpy 压成 1D -> atleast_2d 变成 (1, N)
        assert_eq!(load_series(&dir, "one").unwrap(), vec![vec![0.1, 0.2, 0.3]]);
        assert_eq!(
            load_series(&dir, "two").unwrap(),
            vec![vec![0.1, 0.9], vec![0.2, 0.8]]
        );
        assert_eq!(load_series(&dir, "empty"), None);
        assert_eq!(load_series(&dir, "missing"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_experiment_root_reports_failure() {
        let root = std::env::temp_dir().join("eser_p6_definitely_missing_dir");
        let _ = std::fs::remove_dir_all(&root);
        let r = run(&root);
        assert_eq!(r.exit_code, 1);
        assert!(r.text().contains("[FAIL] 未找到实验目录"), "{}", r.text());
    }

    #[test]
    fn quality_suggestions_follow_the_thresholds() {
        let mut r = Report::default();
        diagnose_quality(
            &Metrics {
                final_r2: Some(0.75),
                final_rmse: Some(0.12),
                ..Default::default()
            },
            &mut r,
        );
        let text = r.text();
        assert!(text.contains("提高模型容量"), "{}", text);
        assert!(text.contains("检查归一化统计量是否过期"), "{}", text);

        let mut r = Report::default();
        diagnose_quality(
            &Metrics {
                final_r2: Some(0.99),
                final_rmse: Some(0.01),
                r2_series: Some(vec![0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]),
                ..Default::default()
            },
            &mut r,
        );
        let text = r.text();
        assert!(text.contains("[ OK ] R² ≥ 0.95"), "{}", text);
        assert!(text.contains("验证 R² 已经平台化"), "{}", text);
    }

    #[test]
    fn overfitting_dropoff_is_warned() {
        let mut r = Report::default();
        diagnose_quality(
            &Metrics {
                final_r2: Some(0.90),
                final_rmse: Some(0.04),
                r2_series: Some(vec![0.80, 0.88, 0.95, 0.96, 0.94, 0.92, 0.90]),
                ..Default::default()
            },
            &mut r,
        );
        let text = r.text();
        assert!(text.contains("验证 R² 从峰值回落了 0.060"), "{}", text);
    }
}
