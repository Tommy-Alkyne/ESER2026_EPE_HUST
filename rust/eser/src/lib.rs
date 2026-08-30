//! ESER 流场 / 污染物代理模型 —— Rust 实现。
//!
//! 移植自同仓库的 Python 版。验收标准不是「跑起来了」，而是逐元素对上
//! `tests/golden/golden.npz` 里由 Python 版冻结的 479 项参考数据。
pub mod geometry;
pub mod ops;
