//! 项目目录与 Python 解释器探测。
//!
//! tkinter 版靠 `sys.executable`：GUI 本身是用哪个 Python 解释器跑起来的，子进程就用
//! 哪个，天然对齐 venv（`gui_trainer.py` 顶部注释专门提过这一坑）。编译型的 Rust GUI
//! 做不到这一点——它是独立可执行文件，不运行在任何 Python 解释器里。这里改用「向上
//! 找项目根目录 + 优先探测 `.venv` + 界面上始终可手动覆盖」的兜底方案，并集中到本
//! 文件方便测试；真正决定起哪个解释器的逻辑只有这一处。

use std::path::{Path, PathBuf};

/// 从 `start` 开始逐级向上找，直到找到同时含有 `train.py` 与 `args.py` 的目录，
/// 视为项目根目录。找不到就返回 `None`（交给界面提示用户手动填）。
pub fn detect_project_dir(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        // Python 退役后不能再拿 train.py / args.py 当标志。用 Rust 侧的固定物：
        // 工作区清单 + 数据目录，两者同时存在才算命中
        if d.join("rust").join("Cargo.toml").is_file() && d.join("dataset").is_dir() {
            return Some(d.to_path_buf());
        }
        dir = d.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn touch(path: &Path) {
        fs::write(path, b"").unwrap();
    }

    /// 造出项目根的两个标志：`rust/Cargo.toml` 与 `dataset/`。
    ///
    /// Python 退役前用的是 `train.py` + `args.py`，那两个文件已经不存在了。
    fn make_markers(root: &Path) {
        fs::create_dir_all(root.join("rust")).unwrap();
        touch(&root.join("rust").join("Cargo.toml"));
        fs::create_dir_all(root.join("dataset")).unwrap();
    }

    #[test]
    fn detect_project_dir_finds_marker_in_start_dir() {
        let tmp = tempfile::tempdir().unwrap();
        make_markers(tmp.path());
        assert_eq!(
            detect_project_dir(tmp.path()),
            Some(tmp.path().to_path_buf())
        );
    }

    #[test]
    fn detect_project_dir_walks_up_from_nested_dir() {
        let tmp = tempfile::tempdir().unwrap();
        make_markers(tmp.path());
        let nested = tmp.path().join("rust").join("gui").join("target");
        fs::create_dir_all(&nested).unwrap();
        assert_eq!(detect_project_dir(&nested), Some(tmp.path().to_path_buf()));
    }

    #[test]
    fn detect_project_dir_requires_both_markers() {
        // 只有工作区清单、没有 dataset/ 时不算命中——单看 Cargo.toml 会在任何
        // Rust 项目里误判
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("rust")).unwrap();
        touch(&tmp.path().join("rust").join("Cargo.toml"));
        assert_eq!(detect_project_dir(tmp.path()), None);
    }

    #[test]
    fn detect_project_dir_none_when_absent() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(detect_project_dir(tmp.path()), None);
    }


}
