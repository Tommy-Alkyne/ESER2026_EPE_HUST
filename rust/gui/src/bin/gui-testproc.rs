//! 子进程管线测试用的小助手。
//!
//! `process.rs` 的集成测试要验的是 **spawn / 日志转发 / 停止校验** 这套管线本身，
//! 不是被启动的程序在干什么。原先借 Python 解释器来当那个「被启动的程序」，
//! Python 退役后需要一个自带的替代品——顺带也去掉了「测试依赖外部解释器」这层脆弱性。
//!
//! ```text
//! gui-testproc echo <行数>          逐行往 stdout 打，然后退出
//! gui-testproc stderr               往 stderr 打一行
//! gui-testproc exit <码>            以给定退出码退出
//! gui-testproc sleep <秒>           睡够再退出（验「停止」用）
//! gui-testproc marker <秒> <路径>   睡够之后写一个标记文件
//! ```
//!
//! `marker` 是**与进程树结构无关**的停止校验：不管中间套了几层壳，只要标记文件
//! 出现了，就说明进程没被真的杀掉。

use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let cmd = a.first().map(|s| s.as_str()).unwrap_or("");
    match cmd {
        "echo" => {
            let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
            for i in 0..n {
                println!("line {}", i);
                // 立刻冲出去：测试要验的是**实时**转发，缓冲会让它看起来像卡住
                let _ = std::io::stdout().flush();
            }
        }
        "stderr" => {
            eprintln!("boom");
            let _ = std::io::stderr().flush();
        }
        "exit" => {
            let code: i32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            std::process::exit(code);
        }
        "sleep" => {
            let secs: f64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            std::thread::sleep(std::time::Duration::from_secs_f64(secs));
            println!("slept");
        }
        "marker" => {
            let secs: f64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let path = a.get(2).cloned().unwrap_or_default();
            std::thread::sleep(std::time::Duration::from_secs_f64(secs));
            let _ = std::fs::write(&path, b"reached");
            println!("marker written");
        }
        other => {
            eprintln!("gui-testproc: 不认识的子命令 {:?}", other);
            std::process::exit(2);
        }
    }
}
