//! ESER 流场 / 污染物代理模型 —— Rust 实现。
//!
//! 移植自同仓库的 Python 版。验收标准不是「跑起来了」，而是逐元素对上
//! `tests/golden/golden.npz` 里由 Python 版冻结的 502 项参考数据。
pub mod attention;
pub mod cli;
pub mod data;
pub mod diagnose;
pub mod geometry;
pub mod interp;
pub mod model;
pub mod model3d;
pub mod ops;
pub mod physics;
pub mod plot;
pub mod predict;
pub mod store;
pub mod train;
pub mod transport;
