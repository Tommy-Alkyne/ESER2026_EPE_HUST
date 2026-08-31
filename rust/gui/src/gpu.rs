//! GPU 状态轮询：调用 `nvidia-smi` 拿利用率/显存，拿不到就优雅降级成「不可用」。
//!
//! 按需求不引入 nvml 绑定依赖，做法与 tkinter 版一致（起 `nvidia-smi` 子进程读
//! CSV），但挪到独立后台线程：tkinter 版是在 Tk 主循环的 `after` 回调里同步调用
//! `subprocess.check_output`，`nvidia-smi` 哪怕只卡 100ms 也会让整个界面跟着卡一下；
//! 这里查询与渲染彻底解耦，界面永远不会被它卡住。

use std::process::Command;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuStatus {
    /// 还没查询到第一次结果（刚启动，还在等第一次轮询）。
    Unknown,
    Available {
        util_pct: u32,
        mem_used_mb: u32,
        mem_total_mb: u32,
    },
    /// 没有 nvidia-smi、命令执行失败、或输出格式不认识——统一按“不可用”处理，
    /// 不区分具体原因，因为对用户来说处理方式都一样：这台机器用不了 GPU 监控。
    Unavailable,
}

/// 解析 `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total
/// --format=csv,noheader,nounits` 的单行输出，形如 `"23, 512, 8192"`。
pub fn parse_nvidia_smi_line(line: &str) -> Option<GpuStatus> {
    let parts: Vec<&str> = line.trim().split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return None;
    }
    let util_pct = parts[0].parse().ok()?;
    let mem_used_mb = parts[1].parse().ok()?;
    let mem_total_mb = parts[2].parse().ok()?;
    Some(GpuStatus::Available {
        util_pct,
        mem_used_mb,
        mem_total_mb,
    })
}

fn query_once() -> GpuStatus {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            text.lines()
                .next()
                .and_then(parse_nvidia_smi_line)
                .unwrap_or(GpuStatus::Unavailable)
        }
        _ => GpuStatus::Unavailable,
    }
}

/// 启动一个每 `interval` 查询一次的后台线程，返回接收端。GUI 侧每帧
/// `try_iter()` 取走已经到达的最新结果即可，从不阻塞渲染线程。
pub fn spawn_poller(interval: Duration) -> Receiver<GpuStatus> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || loop {
        if tx.send(query_once()).is_err() {
            return; // 接收端（窗口）已经没了，线程自己退出
        }
        thread::sleep(interval);
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvidia_smi_line_normal() {
        assert_eq!(
            parse_nvidia_smi_line("23, 512, 8192"),
            Some(GpuStatus::Available {
                util_pct: 23,
                mem_used_mb: 512,
                mem_total_mb: 8192
            })
        );
    }

    #[test]
    fn parse_nvidia_smi_line_no_spaces() {
        assert_eq!(
            parse_nvidia_smi_line("0,0,24576"),
            Some(GpuStatus::Available {
                util_pct: 0,
                mem_used_mb: 0,
                mem_total_mb: 24576
            })
        );
    }

    #[test]
    fn parse_nvidia_smi_line_trailing_newline() {
        assert_eq!(
            parse_nvidia_smi_line("23, 512, 8192\n"),
            Some(GpuStatus::Available {
                util_pct: 23,
                mem_used_mb: 512,
                mem_total_mb: 8192
            })
        );
    }

    #[test]
    fn parse_nvidia_smi_line_wrong_field_count() {
        assert_eq!(parse_nvidia_smi_line("23, 512"), None);
        assert_eq!(parse_nvidia_smi_line("23, 512, 8192, extra"), None);
    }

    #[test]
    fn parse_nvidia_smi_line_non_numeric() {
        assert_eq!(parse_nvidia_smi_line("N/A, N/A, N/A"), None);
    }

    #[test]
    fn parse_nvidia_smi_line_empty() {
        assert_eq!(parse_nvidia_smi_line(""), None);
    }
}
