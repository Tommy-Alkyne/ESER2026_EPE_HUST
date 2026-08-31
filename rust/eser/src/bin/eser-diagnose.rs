//! Rust 版诊断入口，对应 Python 的 `diagnose.py`。
//!
//! ```text
//! eser-diagnose [实验根目录]     # 默认 ./experiments
//! ```
//!
//! 自动挑最新的实验目录，读 `args.txt` 做配置体检，再读 `*.txt` 曲线数值判断训练
//! 质量。**没有发现问题时退出码 0，有问题时退出码 1**——GUI 与脚本靠退出码分流。

use std::path::PathBuf;

fn main() {
    let mut argv = std::env::args().skip(1);
    let root = match argv.next() {
        Some(a) if a == "--help" || a == "-h" => {
            println!("用法：eser-diagnose [实验根目录]（默认 ./experiments）");
            return;
        }
        Some(a) => PathBuf::from(a),
        None => PathBuf::from("experiments"),
    };

    let report = eser::diagnose::run(&root);
    // 一次性打完：报告本身就是逐行文本，中间插别的输出会打乱与 Python 版逐行一致的格式
    print!("{}", report.text());

    if !report.problems.is_empty() {
        std::process::exit(1);
    }
}
