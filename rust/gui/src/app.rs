//! 主界面：左侧参数表单 + 硬件监控，右侧实时日志。
//!
//! 相对 tkinter 版的取舍见 `main.rs` 顶部注释；这里只管“画什么、点什么”，
//! 命令拼装/进程管理/GPU 轮询都委托给同 crate 的纯逻辑模块，界面代码本身
//! 不做任何可测的判断——这部分只能靠人在真机上跑起来看。

use std::path::PathBuf;
use std::time::Duration;

use eframe::egui;

use crate::env as env_detect;
use crate::gpu::{self, GpuStatus};
use crate::log_buffer::LogBuffer;
use crate::params::{ChannelOrder, PhysicsScaling, PlotFn, Task, TrainParams};
use crate::process::{self, LogEvent, Phase, PhaseEvent, ProcessHandle};

/// 训练入口配置：现阶段 Rust 训练循环（P5）还没做完，GUI 起的子进程指向 Python
/// 的 `train.py`；将来切到 Rust 训练二进制时，只需要改这一行常量（连同调用处的
/// 解释器参数一起替换成直接执行该二进制）。
/// 训练与诊断的可执行文件名（不带扩展名）。
///
/// 已从 `python train.py` 切到 Rust 二进制：`eser-train` 的旗标与 `args.py` 逐字
/// 一致，所以 `params.rs::build_args` 一个字都没改，只换了被启动的程序。
const TRAIN_BIN: &str = "eser-train";
const DIAGNOSE_BIN: &str = "eser-diagnose";

/// 正在运行的是训练还是诊断——只影响日志里的提示文案，不影响进程管理逻辑。
#[derive(Clone, Copy, PartialEq, Eq)]
enum RunKind {
    Training,
    Diagnose,
}

pub struct App {
    params: TrainParams,
    project_dir: String,

    phase: Phase,
    current: Option<(RunKind, ProcessHandle)>,
    build_error: Option<String>,

    log: LogBuffer,

    gpu_rx: std::sync::mpsc::Receiver<GpuStatus>,
    gpu_status: GpuStatus,
    /// 中文字体装载结果。装不上时界面全是方块，必须显式告诉用户
    font_status: crate::fonts::FontStatus,
}

impl App {
    pub fn new(font_status: crate::fonts::FontStatus) -> Self {
        let start = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let project_dir = env_detect::detect_project_dir(&start).unwrap_or_else(|| start.clone());

        Self {
            params: TrainParams::default(),
            project_dir: project_dir.display().to_string(),
            phase: Phase::Idle,
            current: None,
            build_error: None,
            log: LogBuffer::new(5000),
            gpu_rx: gpu::spawn_poller(Duration::from_secs(2)),
            gpu_status: GpuStatus::Unknown,
            font_status,
        }
    }

    // ---------------------------------------------------------------- 轮询

    fn poll_gpu(&mut self) {
        while let Ok(status) = self.gpu_rx.try_recv() {
            self.gpu_status = status;
        }
    }

    fn poll_process(&mut self) {
        let mut finished = false;
        if let Some((_, handle)) = &self.current {
            for event in handle.drain() {
                match event {
                    LogEvent::Line(line) => self.log.push_line(&line),
                    LogEvent::Exited { code, killed } => {
                        let verb = if killed { "已停止" } else { "进程结束" };
                        self.log.push_line("");
                        self.log.push_line(&format!(
                            "{} {}（exit code: {:?}） {}",
                            "=".repeat(22),
                            verb,
                            code,
                            "=".repeat(22)
                        ));
                        finished = true;
                    }
                }
            }
        }
        if finished {
            self.current = None;
            self.phase = self.phase.apply(PhaseEvent::Exited);
        }
    }

    // ---------------------------------------------------------------- 子进程控制

    fn start_training(&mut self) {
        self.build_error = None;
        let args = match self.params.build_args() {
            Ok(a) => a,
            Err(e) => {
                self.build_error = Some(e);
                return;
            }
        };
        let exe = match self.resolve_bin(TRAIN_BIN) {
            Ok(p) => p,
            Err(e) => {
                self.build_error = Some(e);
                return;
            }
        };
        self.launch(RunKind::Training, exe, args);
    }

    fn run_diagnose(&mut self) {
        if self.phase.is_busy() {
            return;
        }
        let exe = match self.resolve_bin(DIAGNOSE_BIN) {
            Ok(p) => p,
            Err(e) => {
                self.build_error = Some(e);
                return;
            }
        };
        self.launch(RunKind::Diagnose, exe, Vec::new());
    }

    /// 找到同目录下的可执行文件。
    ///
    /// 优先与 GUI 自身同目录（`cargo build` 后两个二进制都在 `target/<profile>/`），
    /// 其次退到 PATH。**找不到就报错并说清去哪儿构建**，而不是让子进程启动失败后
    /// 只留一句看不懂的系统错误码。
    fn resolve_bin(&self, name: &str) -> Result<PathBuf, String> {
        let exe_name = if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.to_string()
        };
        if let Ok(me) = std::env::current_exe() {
            if let Some(dir) = me.parent() {
                let p = dir.join(&exe_name);
                if p.exists() {
                    return Ok(p);
                }
            }
        }
        // 退到 PATH：让系统去找
        Ok(PathBuf::from(exe_name))
    }

    fn launch(&mut self, kind: RunKind, exe: PathBuf, full_args: Vec<String>) {
        let project_dir = PathBuf::from(self.project_dir.trim());

        self.log.clear();
        self.log
            .push_line(&format!("工作目录: {}", project_dir.display()));
        self.log.push_line(&format!(
            "启动命令: {} {}",
            exe.display(),
            full_args.join(" ")
        ));
        self.log.push_line(&"-".repeat(68));

        match process::spawn(&exe, &full_args, &project_dir) {
            Ok(handle) => {
                self.current = Some((kind, handle));
                self.phase = self.phase.apply(PhaseEvent::Started);
            }
            Err(e) => {
                self.log.push_line(&format!("启动失败: {e}"));
            }
        }
    }

    fn stop_training(&mut self) {
        if let Some((_, handle)) = &self.current {
            self.log.push_line("[正在请求进程退出...]");
            handle.request_stop();
            self.phase = self.phase.apply(PhaseEvent::StopRequested);
        }
    }

    // ---------------------------------------------------------------- UI：分组

    fn ui_data_group(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("数据配置")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("data_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("训练集:");
                        ui.text_edit_singleline(&mut self.params.train_file);
                        ui.end_row();

                        ui.label("测试集(可空):");
                        ui.text_edit_singleline(&mut self.params.test_file);
                        ui.end_row();

                        ui.label("验证集比例 val-split:");
                        ui.add(
                            egui::DragValue::new(&mut self.params.val_split)
                                .speed(0.01)
                                .range(0.0..=0.9),
                        );
                        ui.end_row();
                    });
                ui.checkbox(&mut self.params.ndim3, "3D 体素模型 (--ndim 3)");
                ui.label(
                    egui::RichText::new("测试集留空时，会从训练集里按比例切出互不重叠的验证集")
                        .color(egui::Color32::GRAY)
                        .small(),
                );
            });
    }

    fn ui_physics_group(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("物理设置（PINN）")
            .default_open(true)
            .show(ui, |ui| {
                egui::ComboBox::from_label("物理任务")
                    .selected_text(self.params.task.to_string())
                    .show_ui(ui, |ui| {
                        for t in Task::ALL {
                            ui.selectable_value(&mut self.params.task, t, t.to_string());
                        }
                    });
                egui::ComboBox::from_label("通道顺序")
                    .selected_text(self.params.channel_order.to_string())
                    .show_ui(ui, |ui| {
                        for c in ChannelOrder::ALL {
                            ui.selectable_value(&mut self.params.channel_order, c, c.to_string());
                        }
                    });
                ui.label(
                    egui::RichText::new(
                        "务必与数据集实际通道顺序一致，否则物理约束全部作用在错误通道上",
                    )
                    .color(egui::Color32::from_rgb(200, 80, 80))
                    .small(),
                );

                egui::Grid::new("physics_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("计算域尺寸 Ly:");
                        ui.add(egui::DragValue::new(&mut self.params.domain_size_ly).speed(0.01));
                        ui.end_row();
                        ui.label("计算域尺寸 Lx:");
                        ui.add(egui::DragValue::new(&mut self.params.domain_size_lx).speed(0.01));
                        ui.end_row();

                        ui.label("计算域掩码(.npy/.json):");
                        ui.text_edit_singleline(&mut self.params.t_geometry);
                        ui.end_row();

                        ui.label("密度 rho:");
                        ui.add(egui::DragValue::new(&mut self.params.rho).speed(0.01));
                        ui.end_row();
                        ui.label("运动粘度 nu:");
                        ui.add(egui::DragValue::new(&mut self.params.nu).speed(0.001));
                        ui.end_row();
                        ui.label("散度权重:");
                        ui.add(egui::DragValue::new(&mut self.params.div_weight).speed(0.001));
                        ui.end_row();
                        ui.label("动量权重(0=跳过):");
                        ui.add(egui::DragValue::new(&mut self.params.ns_weight).speed(0.001));
                        ui.end_row();
                        ui.label("扩散系数 D:");
                        ui.add(egui::DragValue::new(&mut self.params.diffusivity).speed(1e-6));
                        ui.end_row();
                        ui.label("对流扩散权重:");
                        ui.add(egui::DragValue::new(&mut self.params.advdiff_weight).speed(0.001));
                        ui.end_row();
                        ui.label("平滑权重:");
                        ui.add(egui::DragValue::new(&mut self.params.smooth_weight).speed(0.001));
                        ui.end_row();
                        ui.label("物理项预热比例:");
                        ui.add(
                            egui::DragValue::new(&mut self.params.physics_warmup)
                                .speed(0.01)
                                .range(0.0..=1.0),
                        );
                        ui.end_row();
                    });
                ui.checkbox(
                    &mut self.params.no_mask,
                    "关闭计算域掩码（在整幅网格上训练）",
                );
            });
    }

    fn ui_network_group(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("网络结构")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("network_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("dense blocks (奇数个，逗号分隔):");
                        ui.text_edit_singleline(&mut self.params.blocks);
                        ui.end_row();
                        ui.label("网络宽度 growth-rate:");
                        ui.add(egui::DragValue::new(&mut self.params.growth_rate).range(1..=256));
                        ui.end_row();
                        ui.label("初始特征 init-features:");
                        ui.add(egui::DragValue::new(&mut self.params.init_features).range(1..=512));
                        ui.end_row();
                        ui.label("注意力头数:");
                        ui.add(egui::DragValue::new(&mut self.params.num_heads).range(1..=32));
                        ui.end_row();
                    });
                ui.checkbox(
                    &mut self.params.no_positional_encoding,
                    "关闭 Transformer 位置编码（消融实验）",
                );
            });
    }

    fn ui_hyperparam_group(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("训练超参数")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("hyper_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("总轮数 epochs:");
                        ui.add(egui::DragValue::new(&mut self.params.epochs).range(1..=1_000_000));
                        ui.end_row();
                        ui.label("批大小 batch-size:");
                        ui.add(egui::DragValue::new(&mut self.params.batch_size).range(1..=4096));
                        ui.end_row();
                        ui.label("学习率 lr:");
                        ui.add(egui::DragValue::new(&mut self.params.lr).speed(0.0001));
                        ui.end_row();
                        ui.label("权重衰减 weight-decay:");
                        ui.add(egui::DragValue::new(&mut self.params.weight_decay).speed(0.0001));
                        ui.end_row();
                        ui.label("梯度裁剪 grad-clip (<=0 关闭):");
                        ui.add(egui::DragValue::new(&mut self.params.grad_clip).speed(0.01));
                        ui.end_row();
                        ui.label("随机种子 seed:");
                        ui.add(egui::DragValue::new(&mut self.params.seed));
                        ui.end_row();
                    });
            });
    }

    fn ui_experiment_group(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("实验管理")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("exp_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("实验名:");
                        ui.text_edit_singleline(&mut self.params.exp_name);
                        ui.end_row();
                        ui.label("存档频率 ckpt-freq:");
                        ui.add(egui::DragValue::new(&mut self.params.ckpt_freq).range(1..=100_000));
                        ui.end_row();
                        ui.label("出图频率 plot-freq:");
                        ui.add(egui::DragValue::new(&mut self.params.plot_freq).range(1..=100_000));
                        ui.end_row();
                    });
            });
    }

    fn ui_advanced_group(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("高级参数（默认不用改）")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("advanced_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Dropout drop-rate:");
                        ui.add(
                            egui::DragValue::new(&mut self.params.drop_rate)
                                .speed(0.01)
                                .range(0.0..=1.0),
                        );
                        ui.end_row();
                        ui.label("bn-size:");
                        ui.add(egui::DragValue::new(&mut self.params.bn_size).range(1..=64));
                        ui.end_row();
                        ui.label("学习率 patience:");
                        ui.add(egui::DragValue::new(&mut self.params.lr_patience).range(1..=1000));
                        ui.end_row();
                        ui.label("测试 batch-size:");
                        ui.add(
                            egui::DragValue::new(&mut self.params.test_batch_size)
                                .range(1..=10_000),
                        );
                        ui.end_row();
                        ui.label("日志频率 log-freq:");
                        ui.add(egui::DragValue::new(&mut self.params.log_freq).range(1..=10_000));
                        ui.end_row();
                        ui.label("注意力 token 上限:");
                        ui.add(
                            egui::DragValue::new(&mut self.params.attn_max_tokens)
                                .range(1..=1_000_000),
                        );
                        ui.end_row();
                        ui.label("实验根目录 exp-dir:");
                        ui.text_edit_singleline(&mut self.params.exp_dir);
                        ui.end_row();
                        ui.label("数据根目录 data-dir:");
                        ui.text_edit_singleline(&mut self.params.data_dir);
                        ui.end_row();
                        ui.label("HDF5 输入数据集名(留空自动探测):");
                        ui.text_edit_singleline(&mut self.params.input_key);
                        ui.end_row();

                        ui.label("DataLoader worker 数:");
                        ui.horizontal(|ui| {
                            let mut auto = self.params.num_workers.is_none();
                            ui.checkbox(&mut auto, "自动");
                            if auto {
                                self.params.num_workers = None;
                            } else {
                                let mut n = self.params.num_workers.unwrap_or(0);
                                ui.add(egui::DragValue::new(&mut n).range(0..=64));
                                self.params.num_workers = Some(n);
                            }
                        });
                        ui.end_row();

                        ui.label("物理量级归一化:");
                        egui::ComboBox::from_id_salt("physics_scaling")
                            .selected_text(self.params.physics_loss_scaling.to_string())
                            .show_ui(ui, |ui| {
                                for s in PhysicsScaling::ALL {
                                    ui.selectable_value(
                                        &mut self.params.physics_loss_scaling,
                                        s,
                                        s.to_string(),
                                    );
                                }
                            });
                        ui.end_row();

                        ui.label("源强缩放 source-scale (dispersion):");
                        ui.add(egui::DragValue::new(&mut self.params.source_scale).speed(0.01));
                        ui.end_row();

                        ui.label("速度通道 Vx, Vy (dispersion):");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.params.velocity_vx).range(0..=32),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.params.velocity_vy).range(0..=32),
                            );
                        });
                        ui.end_row();

                        ui.label("预测图渲染:");
                        egui::ComboBox::from_id_salt("plot_fn")
                            .selected_text(self.params.plot_fn.to_string())
                            .show_ui(ui, |ui| {
                                for p in PlotFn::ALL {
                                    ui.selectable_value(&mut self.params.plot_fn, p, p.to_string());
                                }
                            });
                        ui.end_row();
                    });

                ui.checkbox(&mut self.params.bottleneck, "启用 bottleneck 层");
                ui.checkbox(
                    &mut self.params.prefer_sensor_input,
                    "优先用 sensor_input 数据集",
                );
                ui.checkbox(
                    &mut self.params.physics_in_physical_units,
                    "物理损失前先反归一化（有量纲场上计算）",
                );

                ui.separator();
                ui.checkbox(
                    &mut self.params.post,
                    "仅评估已有 checkpoint（--post，不训练）",
                );
                if self.params.post {
                    ui.horizontal(|ui| {
                        // 留空走 best 而不是最后一轮：最后一轮不一定是最好的那个，
                        // 拿它评出来的指标会低估模型。二进制侧的默认值就是 best
                        ui.label("评估哪个 epoch 的 checkpoint（留空 = best）:");
                        let mut has_value = self.params.ckpt_epoch.is_some();
                        ui.checkbox(&mut has_value, "指定");
                        if has_value {
                            let mut e = self.params.ckpt_epoch.unwrap_or(self.params.epochs);
                            ui.add(egui::DragValue::new(&mut e).range(1..=1_000_000));
                            self.params.ckpt_epoch = Some(e);
                        } else {
                            self.params.ckpt_epoch = None;
                        }
                    });
                }
            });
    }

    fn ui_env_group(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("运行环境")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("env_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("项目目录:");
                        ui.text_edit_singleline(&mut self.project_dir);
                        ui.end_row();
                        ui.end_row();
                    });
                if ui.button("重新自动探测").clicked() {
                    let dir = env_detect::detect_project_dir(&PathBuf::from(&self.project_dir))
                        .or_else(|| {
                            std::env::current_dir()
                                .ok()
                                .and_then(|d| env_detect::detect_project_dir(&d))
                        });
                    if let Some(dir) = dir {
                        self.project_dir = dir.display().to_string();
                    }
                }
                ui.label(
                    egui::RichText::new(
                        "GUI 是独立可执行文件，不像 tkinter 版能靠 sys.executable 自动对齐 venv；\
                     这里探测不到就手动填，两个字段都可以直接编辑。",
                    )
                    .color(egui::Color32::GRAY)
                    .small(),
                );
            });
    }

    /// 中文字体没装上时的横幅。
    ///
    /// **不能只在 stderr 打一行**：GUI 是双击起的，没人看得到控制台，
    /// 而字体缺失的表现是满屏方块——用户只会以为程序坏了。这条提示必须在界面上。
    fn ui_font_warning(&mut self, ui: &mut egui::Ui) {
        if self.font_status.is_loaded() {
            return;
        }
        ui.colored_label(
            egui::Color32::from_rgb(200, 60, 60),
            "未找到中文字体，界面文字会显示成方块。\n             请安装 simhei.ttf / Deng.ttf 等中文字体后重启。",
        );
        ui.separator();
    }

    fn ui_controls(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.phase.is_busy(), egui::Button::new("开始训练"))
                .clicked()
            {
                self.start_training();
            }
            if ui
                .add_enabled(self.phase == Phase::Running, egui::Button::new("停止训练"))
                .clicked()
            {
                self.stop_training();
            }
            if ui
                .add_enabled(!self.phase.is_busy(), egui::Button::new("运行诊断"))
                .clicked()
            {
                self.run_diagnose();
            }
        });
        if let Some(err) = &self.build_error {
            ui.colored_label(
                egui::Color32::from_rgb(200, 60, 60),
                format!("参数错误: {err}"),
            );
        }
        let status_text = match self.phase {
            Phase::Idle => "空闲".to_string(),
            Phase::Running => "运行中".to_string(),
            Phase::Stopping => "正在停止…".to_string(),
        };
        ui.label(format!("状态: {status_text}"));
    }

    fn ui_gpu_group(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("硬件监控")
            .default_open(true)
            .show(ui, |ui| match self.gpu_status {
                GpuStatus::Unknown => {
                    ui.label("GPU 使用率: 获取中…");
                    ui.label("显存: 获取中…");
                }
                GpuStatus::Available {
                    util_pct,
                    mem_used_mb,
                    mem_total_mb,
                } => {
                    ui.colored_label(
                        egui::Color32::from_rgb(80, 140, 255),
                        format!("GPU 使用率: {util_pct:>3} %"),
                    );
                    ui.colored_label(
                        egui::Color32::from_rgb(170, 100, 220),
                        format!("显存: {mem_used_mb:>6} / {mem_total_mb} MB"),
                    );
                }
                GpuStatus::Unavailable => {
                    ui.label("GPU: 未检测到 NVIDIA 设备（将使用 CPU）");
                    ui.label("显存: -- / -- MB");
                }
            });
    }

    fn ui_log_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading(format!("实时训练日志（{} 行）", self.log.line_count()));
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.log.text().to_string())
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY)
                        .interactive(false),
                );
            });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_gpu();
        self.poll_process();

        egui::SidePanel::left("params_panel")
            .resizable(true)
            .default_width(440.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.ui_font_warning(ui);
                    ui.add_enabled_ui(!self.phase.is_busy(), |ui| {
                        self.ui_data_group(ui);
                        self.ui_physics_group(ui);
                        self.ui_network_group(ui);
                        self.ui_hyperparam_group(ui);
                        self.ui_experiment_group(ui);
                        self.ui_advanced_group(ui);
                        self.ui_env_group(ui);
                    });
                    self.ui_controls(ui);
                    self.ui_gpu_group(ui);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.ui_log_panel(ui);
        });

        // 日志/GPU 都是后台线程往 channel 里塞数据，这里定时醒来读一次，
        // 训练跑起来时日志才会“实时滚动”而不是等下一次用户交互才刷新。
        ctx.request_repaint_after(Duration::from_millis(120));
    }
}
