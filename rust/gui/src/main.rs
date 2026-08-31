//! T 型流体场 PINN 训练控制台（egui 重写版）。
//!
//! 对应 Python `gui_trainer.py`（356 行 tkinter），按方案文档 §3.5/§八 用 egui
//! 全量重写，借机做了几处调整（细节见各模块顶部注释与随附汇报）：
//!
//! * 参数表单从 tkinter 的裸文本框换成类型化控件（`DragValue`/`ComboBox`），
//!   不再可能因为打错字符导致 `argparse` 报错。
//! * GPU 轮询、日志转发都挪到独立后台线程，界面不会被 `nvidia-smi` 或子进程
//!   输出短暂卡住。
//! * 停止训练由“调用了 terminate/kill”改成“轮询 `try_wait` 确认进程真的退出
//!   才回到空闲状态”，按钮状态不会撒谎。
//! * 去掉了原版的路径浏览对话框（`filedialog.askopenfilename`）：改文本框直接
//!   输入/粘贴路径，省掉一整个新依赖（`rfd`），也符合已定案的“重写不是照搬、
//!   但也别为了对齐旧控件而堆依赖”。
//!
//! 模块划分：
//! * `params`  —— 表单数据模型 + 命令行拼装（纯逻辑，单测覆盖）
//! * `env`     —— 项目目录 / Python 解释器探测（纯逻辑，单测覆盖）
//! * `process` —— 子进程生命周期：状态机 `Phase`（单测覆盖）+ 实际 spawn/kill（集成验证）
//! * `gpu`     —— `nvidia-smi` 轮询与解析（纯解析单测覆盖 + 实际调用集成验证）
//! * `log_buffer` —— 有界日志缓冲（纯逻辑，单测覆盖）
//! * `app`     —— eframe::App 实现，纯 UI 拼装，人工验证

mod app;
mod env;
mod fonts;
mod gpu;
mod log_buffer;
mod params;
mod process;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 860.0])
            .with_min_inner_size([1040.0, 700.0]),
        ..Default::default()
    };
    eframe::run_native(
        "T 型流体场 PINN 训练控制台",
        options,
        Box::new(|cc| {
            // **必须在建 App 之前装中文字体**：egui 内置字体只有拉丁字符，
            // 不装的话满屏汉字都是方块，而这既不报错也不 panic
            let status = fonts::install(&cc.egui_ctx);
            Ok(Box::new(app::App::new(status)))
        }),
    )
}
