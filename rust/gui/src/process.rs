//! 子进程训练/诊断管线：非阻塞启动、stdout+stderr 实时转发、停止时校验进程真的退出。
//!
//! `Phase` 是纯状态机，不摸任何 OS 资源，是这个文件里能单独测的部分；真正的
//! `Child`、读取线程、kill 都在 `spawn` 与背后的监督线程里，靠集成测试（用项目自带
//! venv 的 Python 起一个短命子进程跑一遍，见文件末尾 `tests::spawn_*`）而不是纯逻辑
//! 单元测试——按验收要求，这些测试只用 `python -c "print(...)"` 这类秒退的命令，
//! 不会真的拉起训练把 GPU 占满。

use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// 子进程生命周期的纯状态机：只依赖“发生了什么事件”，UI 按钮的可用状态直接读它。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// 空闲，可以点“开始训练”/“运行诊断”。
    Idle,
    /// 正在跑。
    Running,
    /// 用户点了“停止”，已经发出 kill，正等待进程侧确认退出——停止按钮此时应禁用，
    /// 避免用户以为没反应而反复点击、把 kill 发了好几遍。
    Stopping,
}

/// 驱动状态机的事件。
#[derive(Debug, Clone, Copy)]
pub enum PhaseEvent {
    Started,
    StopRequested,
    Exited,
}

impl Phase {
    /// 状态转移。非法组合（比如 Idle 时收到 StopRequested）原地不动——
    /// 调用方本该通过按钮的 enabled 状态挡住这些操作，这里再兜底一层不 panic。
    pub fn apply(self, event: PhaseEvent) -> Phase {
        match (self, event) {
            (Phase::Idle, PhaseEvent::Started) => Phase::Running,
            (Phase::Running, PhaseEvent::StopRequested) => Phase::Stopping,
            (Phase::Running, PhaseEvent::Exited) => Phase::Idle,
            (Phase::Stopping, PhaseEvent::Exited) => Phase::Idle,
            (state, _) => state,
        }
    }

    pub fn is_busy(self) -> bool {
        !matches!(self, Phase::Idle)
    }
}

/// 从子进程转发出来的一条事件：一行日志，或者它的终止结果。
pub enum LogEvent {
    Line(String),
    /// 进程已经通过 `try_wait` 确认退出。`killed` 标记“是不是我们主动杀的”，
    /// 用来在日志里区分“训练自己跑完了”和“用户按了停止”。
    Exited {
        code: Option<i32>,
        killed: bool,
    },
}

/// 正在运行的子进程句柄。UI 线程只通过它收日志、发停止请求，绝不直接摸 `Child`——
/// `Child` 被监督线程独占，这样“kill 之后继续 try_wait 校验”才不用加锁。
pub struct ProcessHandle {
    kill_flag: Arc<AtomicBool>,
    rx: Receiver<LogEvent>,
}

impl ProcessHandle {
    /// 非阻塞地取出目前已经到达的所有事件（日志行 / 退出通知）。
    pub fn drain(&self) -> Vec<LogEvent> {
        self.rx.try_iter().collect()
    }

    /// 请求停止。真正的 `child.kill()` 在监督线程里做——只有它能在 kill 之后
    /// 继续用 `try_wait` 反复确认进程确实没了，而不是“调用成功就当停了”。
    pub fn request_stop(&self) {
        self.kill_flag.store(true, Ordering::SeqCst);
    }
}

/// 启动子进程并立刻返回句柄；日志转发、退出校验都在后台线程完成，不阻塞调用方。
///
/// 与 tkinter 版的 `subprocess.Popen(cmd, cwd=..., env=...)` 一一对应：stdout/stderr
/// 都接管为管道、注入同样的 `PYTHONIOENCODING`/`PYTHONUNBUFFERED`，保证 `print`
/// 不会攒在 Python 的输出缓冲区里等到进程结束才一次性刷出来。
pub fn spawn(program: &Path, args: &[String], cwd: &Path) -> std::io::Result<ProcessHandle> {
    let mut child = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()?;

    let stdout = child
        .stdout
        .take()
        .expect("stdout 已被 Stdio::piped() 接管");
    let stderr = child
        .stderr
        .take()
        .expect("stderr 已被 Stdio::piped() 接管");

    let (tx, rx) = mpsc::channel::<LogEvent>();
    let kill_flag = Arc::new(AtomicBool::new(false));

    spawn_line_reader(stdout, tx.clone());
    spawn_line_reader(stderr, tx.clone());
    spawn_supervisor(child, tx, kill_flag.clone());

    Ok(ProcessHandle { kill_flag, rx })
}

/// 逐行转发一个管道（stdout 或 stderr）。两条管道各自独立开线程转发，不追求
/// 与 Python 端 `stderr=subprocess.STDOUT` 完全一致的字节级交错顺序——
/// 训练日志几乎全走 stdout，这点顺序误差不影响可读性，换来的是不必为了合并
/// 两个句柄再引入一个 `os_pipe` 之类的新依赖。
fn spawn_line_reader<R: Read + Send + 'static>(reader: R, tx: mpsc::Sender<LogEvent>) {
    thread::spawn(move || {
        let buf = BufReader::new(reader);
        for line in buf.lines() {
            match line {
                Ok(text) => {
                    if tx.send(LogEvent::Line(text)).is_err() {
                        break; // 接收端（GUI）已经没了，没必要继续读
                    }
                }
                Err(_) => break, // 管道关闭 / 解码错误，当 EOF 处理
            }
        }
    });
}

/// 杀掉整棵进程树，而不只是 `Child::kill()` 能碰到的那一个直接子进程。
///
/// 实测踩到的坑：本机项目的 `.venv/Scripts/python.exe`（`uv` 建的 venv）是个
/// **启动器 shim**——它自己再 `CreateProcess` 出一个真正跑代码的内层解释器
/// （`%APPDATA%\uv\python\...\python.exe`）。`Child::kill()` 只掐得死 Rust
/// 这边直接持有句柄的那一层；一旦训练还没起稳（内层解释器刚被 shim 拉起、
/// 两者还没来得及被同一个 Job Object 绑定）就点了停止，外层 shim 应声而死，
/// 内层那个真正在吃显存跑训练的解释器却成了孤儿，继续在后台空转——这正是
/// 需求里点名的“点了停止但进程还在跑”，只是它藏在看不见的子进程后面。
///
/// Windows 下改用 `taskkill /PID <pid> /T /F` 递归杀掉以该 PID 为根的整棵树，
/// 不管中间套了几层 shim/子进程都能连根拔起，不需要引入 job-object 相关的
/// 新依赖。非 Windows 平台上 venv 的 `python` 通常是可执行文件本身（或到它的
/// 符号链接），不会有这层套娃，`Child::kill()` 已经够用。
fn kill_tree(child: &mut Child) -> bool {
    #[cfg(windows)]
    {
        let pid = child.id().to_string();
        Command::new("taskkill")
            .args(["/PID", &pid, "/T", "/F"])
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    }
    #[cfg(not(windows))]
    {
        child.kill().is_ok()
    }
}

/// 监督线程：轮询 `try_wait` 而不是阻塞 `wait`，这样才能在轮询间隙响应
/// `kill_flag`；kill 也在这里发起，之后继续轮询直到系统确认进程真的没了
/// 才发 `Exited`——“停止”是被进程返回码证实的，不是 UI 单方面宣称的。
fn spawn_supervisor(mut child: Child, tx: mpsc::Sender<LogEvent>, kill_flag: Arc<AtomicBool>) {
    thread::spawn(move || {
        let mut killed = false;
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let _ = tx.send(LogEvent::Exited {
                        code: status.code(),
                        killed,
                    });
                    return;
                }
                Ok(None) => {
                    if kill_flag.load(Ordering::SeqCst) {
                        // 不是打一次就算数：uv 的 venv shim 拉起内层解释器有短暂延迟，
                        // 第一次 `kill_tree` 时那个内层进程可能还没来得及出现在进程树
                        // 里，一次性的 kill 会跟这个时间差赛跑、偶尔会输（实测跑 5 轮
                        // 就抓到过一次残留）。改成每个轮询周期都重新扫一遍，直到
                        // `try_wait` 确认整个进程真的退出为止，把这个竞态窗口从
                        // “一次没打中就永久漏网”收窄成“最多晚一个 80ms 轮询周期”。
                        if kill_tree(&mut child) {
                            killed = true;
                        }
                    }
                    thread::sleep(Duration::from_millis(80));
                }
                Err(_) => {
                    // try_wait 本身失败（句柄失效等极端情况），没法再确认退出码，
                    // 如实上报“不知道”而不是假装进程还活着。
                    let _ = tx.send(LogEvent::Exited { code: None, killed });
                    return;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------- Phase 状态机

    #[test]
    fn phase_idle_to_running_on_started() {
        assert_eq!(Phase::Idle.apply(PhaseEvent::Started), Phase::Running);
    }

    #[test]
    fn phase_running_to_stopping_on_stop_requested() {
        assert_eq!(
            Phase::Running.apply(PhaseEvent::StopRequested),
            Phase::Stopping
        );
    }

    #[test]
    fn phase_running_to_idle_on_exited_without_stop() {
        // 训练自己跑完了（没人点停止），照样要回到 Idle 让按钮恢复可用。
        assert_eq!(Phase::Running.apply(PhaseEvent::Exited), Phase::Idle);
    }

    #[test]
    fn phase_stopping_to_idle_on_exited() {
        assert_eq!(Phase::Stopping.apply(PhaseEvent::Exited), Phase::Idle);
    }

    #[test]
    fn phase_idle_ignores_stop_requested() {
        assert_eq!(Phase::Idle.apply(PhaseEvent::StopRequested), Phase::Idle);
    }

    #[test]
    fn phase_idle_ignores_exited() {
        assert_eq!(Phase::Idle.apply(PhaseEvent::Exited), Phase::Idle);
    }

    #[test]
    fn phase_stopping_ignores_started() {
        assert_eq!(Phase::Stopping.apply(PhaseEvent::Started), Phase::Stopping);
    }

    #[test]
    fn phase_running_ignores_started() {
        assert_eq!(Phase::Running.apply(PhaseEvent::Started), Phase::Running);
    }

    #[test]
    fn phase_is_busy() {
        assert!(!Phase::Idle.is_busy());
        assert!(Phase::Running.is_busy());
        assert!(Phase::Stopping.is_busy());
    }

    // -------------------------------------------------------------- 子进程管线集成测试
    //
    // 用项目自带 venv 的 Python 起真实子进程，验证的是 spawn/日志转发/停止校验这套
    // 管线本身，不是训练逻辑——全部用秒退的 `python -c "..."` 或 sleep，不触碰
    // train.py，不会占用 GPU。

    /// 测试用的被启动程序：**本 crate 自带的 `gui-testproc`**。
    ///
    /// 原先借项目 venv 里的 Python 当这个角色。Python 退役之后需要一个替代品，
    /// 顺带也去掉了「管线测试依赖外部解释器」这层脆弱性——现在被启动的是
    /// 我们自己编出来的二进制，行为完全可控。
    ///
    /// 它与测试可执行文件同目录（`cargo test` 会一并构建 bin 目标）。
    fn test_proc() -> std::path::PathBuf {
        let name = if cfg!(windows) {
            "gui-testproc.exe"
        } else {
            "gui-testproc"
        };
        let exe = std::env::current_exe().expect("拿不到测试可执行文件路径");
        // 测试二进制在 target/<profile>/deps/，助手在它的上一级
        for dir in exe.ancestors().skip(1).take(3) {
            let p = dir.join(name);
            if p.is_file() {
                return p;
            }
        }
        panic!(
            "找不到 {}（应与测试可执行文件同目录）。先跑一次 cargo build -p gui",
            name
        );
    }

    /// 轮询 `drain()` 直到等到一次 `Exited` 事件或超时，返回收集到的全部事件。
    fn drain_until_exited(handle: &ProcessHandle, timeout: Duration) -> Vec<LogEvent> {
        let start = std::time::Instant::now();
        let mut collected = Vec::new();
        loop {
            let batch = handle.drain();
            let saw_exit = batch.iter().any(|e| matches!(e, LogEvent::Exited { .. }));
            collected.extend(batch);
            if saw_exit {
                return collected;
            }
            if start.elapsed() > timeout {
                panic!("等 {timeout:?} 都没等到 Exited 事件，管线大概率卡死了");
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn spawn_captures_stdout_and_reports_clean_exit() {
        let prog = test_proc();
        let cwd = std::env::current_dir().unwrap();
        let handle = spawn(
            &prog,
            &["echo".to_string(), "1".to_string()],
            &cwd,
        )
        .expect("起子进程失败");

        let events = drain_until_exited(&handle, Duration::from_secs(10));

        let saw_line_1 = events
            .iter()
            .any(|e| matches!(e, LogEvent::Line(text) if text.trim() == "line 0"));
        assert!(
            saw_line_1,
            "没有实时收到子进程打印的 \"1\"（stdout 转发没工作）"
        );

        let exit = events.iter().find_map(|e| match e {
            LogEvent::Exited { code, killed } => Some((*code, *killed)),
            _ => None,
        });
        assert_eq!(
            exit,
            Some((Some(0), false)),
            "正常退出应上报 exit code 0、killed=false"
        );
    }

    #[test]
    fn spawn_nonexistent_program_returns_err() {
        // 程序本身不存在时 spawn 应该在调用处直接失败，而不是起了个“幽灵句柄”。
        let cwd = std::env::current_dir().unwrap();
        let bogus = std::path::PathBuf::from("this-program-does-not-exist-anywhere.exe");
        assert!(spawn(&bogus, &[], &cwd).is_err());
    }

    #[test]
    fn request_stop_actually_kills_and_is_confirmed_by_exit_status() {
        let prog = test_proc();
        let cwd = std::env::current_dir().unwrap();
        // 一个不主动退出的长命令：没有我们发的 kill，10 秒内不会自己结束。
        let handle = spawn(&prog, &["sleep".to_string(), "60".to_string()], &cwd)
            .expect("起子进程失败");

        // 给它一点时间真正跑起来，再发停止请求——这才是“正在运行时点停止”的真实场景。
        thread::sleep(Duration::from_millis(200));
        handle.request_stop();

        let events = drain_until_exited(&handle, Duration::from_secs(10));
        let exit = events.iter().find_map(|e| match e {
            LogEvent::Exited { code, killed } => Some((*code, *killed)),
            _ => None,
        });
        let (_, killed) = exit.expect("发了停止请求却没等到 Exited——进程侧没法核验真的停了");
        assert!(
            killed,
            "Exited 事件里 killed 应为 true，证明是我们主动杀的而不是它自己退出"
        );

        // 双重核验：既然监督线程已经用 try_wait 拿到了 ExitStatus 才发这条事件，
        // 这里再确认一次不会再收到任何新事件（进程真的没了，不是假死后台还在转）。
        thread::sleep(Duration::from_millis(300));
        assert!(handle.drain().is_empty(), "已确认退出后不该再有新事件");
    }

    /// 这个测试不看 Rust 这边 `try_wait` 报了什么——它只信一个跟进程树结构完全
    /// 无关的外部副作用：让子进程在 sleep 一段时间后往磁盘写一个标记文件。
    ///
    /// 动机：这台机器上项目的 `.venv/Scripts/python.exe`（uv 建的 venv）实测是个
    /// **启动器 shim**，它自己又 `CreateProcess` 出一个真正干活的内层解释器
    /// （`AppData\Roaming\uv\python\...\python.exe`）。之前只按 Rust `Child` 的
    /// `try_wait`/`kill()` 校验时，`killed=true` 拿到手了、测试也是绿的——但那只
    /// 证明外层 shim 死了，内层真正在跑代码的进程完全没被动到，会一直在后台空转。
    /// 这正是需求里点名警告的“点了停止但进程还在跑”，只是它藏在一层看不见的
    /// 子进程后面，光看 Rust `Child` 的返回码测不出来。
    ///
    /// 用延迟副作用而不是查系统进程表，是因为副作用是“进程树里不管哪一层、只要
    /// 还有任何一环活着就会发生”的客观事实，不依赖 `tasklist`/WMI 之类的额外
    /// 平台工具、也不用关心 shim 具体套了几层。
    /// `grace` 是“子进程起了多久之后才点停止”。跑一遍完整的
    /// spawn → 等 grace → request_stop → 校验标记文件没出现。
    fn assert_stop_kills_tree(marker_suffix: &str, grace: Duration) {
        let prog = test_proc();
        let cwd = std::env::current_dir().unwrap();
        let marker = std::env::temp_dir().join(format!(
            "gui_stop_tree_test_{}_{}.marker",
            std::process::id(),
            marker_suffix
        ));
        let _ = std::fs::remove_file(&marker); // 清掉可能的上次残留

        // sleep(2) 之后才写标记文件：只要整棵进程树里还有任何一环活到 2 秒，
        // 标记文件就会出现。
        let handle = spawn(
            &prog,
            &[
                "marker".to_string(),
                "2".to_string(),
                marker.display().to_string(),
            ],
            &cwd,
        )
        .expect("起子进程失败");

        thread::sleep(grace);
        handle.request_stop();

        let events = drain_until_exited(&handle, Duration::from_secs(10));
        let killed = events.iter().find_map(|e| match e {
            LogEvent::Exited { killed, .. } => Some(*killed),
            _ => None,
        });
        assert_eq!(killed, Some(true), "应确认是我们主动停止的");

        // 2.5s > 脚本里的 sleep(2)，留够余量：真的没杀干净的话，标记文件这时
        // 应该已经出现了。
        thread::sleep(Duration::from_millis(2500));
        let leaked = marker.exists();
        let _ = std::fs::remove_file(&marker);
        assert!(
            !leaked,
            "停止之后标记文件还是被写出来了——进程树没有被真正杀干净"
        );
    }

    #[test]
    fn request_stop_actually_kills_the_whole_process_tree() {
        // 给 200ms 让 shim 有充分时间把内层解释器起稳，测的是“正常时序”下杀得干不干净。
        assert_stop_kills_tree("normal", Duration::from_millis(200));
    }

    #[test]
    fn request_stop_immediately_after_spawn_still_kills_whole_tree() {
        // 几乎不给 shim 反应时间就点停止，专门压“内层解释器还没起稳、
        // 两层进程还没被同一个 Job Object 绑定”这个竞态窗口——
        // 这正是我们在真机上第一次实测时意外撞见孤儿进程的场景。
        assert_stop_kills_tree("race", Duration::from_millis(1));
    }
}
