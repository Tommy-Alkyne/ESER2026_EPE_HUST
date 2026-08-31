//! 数据管线（P4）：HDF5 读取、全量统计、归一化、train/val 划分、指标累计器。
//!
//! 移植自 `utils/data_common.py` / `utils/load_data.py` / `utils/load_data_3d.py`。
//!
//! 这一层有四处「错了不报错」的坑，是本模块每个设计选择的理由：
//!
//! 1. **统计量必须全量扫描**。旧 Python 版只用前 1000 个样本估 min/max，再
//!    `clamp(0,1)` 把之后所有超范围的真实值永久削平——回流死角、压力峰这些最难
//!    也最重要的区域被系统性抹掉，而 loss 曲线看起来一切正常。
//! 2. **划分下标必须与 Python 逐位一致**。所以本文件自带一份 numpy `RandomState`
//!    的 MT19937，而不是随便挑个 Rust RNG：种子相同但序列不同，训练/验证集就在两个
//!    实现之间分岔了，R² 从此不可对照，且没有任何报错。
//! 3. **`rust-hdf5` 的 `read_raw::<T>` 只核对元素字节数，不核对数据类型**。把 f32
//!    数据集当 i32 读会静默返回垃圾（实测返回 `Ok(1087919182)`）。所以每次读之前
//!    显式验一次 dtype 与字节序。
//! 4. **归一化不做 clamp**。越界值保持原样，由调用方决定是否告警
//!    （[`OutputStats::out_of_range`]），避免静默削平真实极值。

use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rust_hdf5::format::messages::datatype::ByteOrder;
use rust_hdf5::{DatatypeMessage, H5Dataset, H5File};

use crate::interp::{resize_bilinear, resize_trilinear};

/// 与 `data_common.normalize_array` 的 `eps` 一致。分母加它是为了让常数场
/// （max==min）归一化后恒为 0 而不是 NaN。
pub const NORM_EPS: f32 = 1e-8;

// ================================================================ HDF5 读取

/// 读一块数据并统一成 f32。
///
/// Python 侧是 `np.asarray(dset[...], dtype=np.float32)`，什么数值类型都吃；这里
/// 只接 f32 / f64（CFD 求解器的实际输出），别的显式报错而不是转成垃圾。
fn read_block_f32(ds: &H5Dataset, starts: &[usize], counts: &[usize]) -> Result<Vec<f32>, String> {
    let dt = ds
        .datatype()
        .map_err(|e| format!("读 datatype 失败: {}", e))?;
    // 字节序必须先查：rust-hdf5 直接把磁盘字节按主机序重解释，大端文件读出来
    // 是垃圾数值而不是错误。
    let order = match &dt {
        DatatypeMessage::FloatingPoint { byte_order, .. } => Some(*byte_order),
        _ => None,
    };
    if order.is_some_and(|o| o != ByteOrder::LittleEndian) {
        return Err(format!("只支持小端浮点数据集，实得 {:?}", dt));
    }
    match dt {
        DatatypeMessage::FloatingPoint { size: 4, .. } => ds
            .read_slice::<f32>(starts, counts)
            .map_err(|e| format!("读 f32 hyperslab 失败: {}", e)),
        DatatypeMessage::FloatingPoint { size: 8, .. } => ds
            .read_slice::<f64>(starts, counts)
            .map(|v| v.into_iter().map(|x| x as f32).collect())
            .map_err(|e| format!("读 f64 hyperslab 失败: {}", e)),
        other => Err(format!("只支持 f32 / f64 数据集，实得 {:?}", other)),
    }
}

/// 读整个数据集，返回 (扁平 f32 数据, 形状)。
fn read_all_f32(ds: &H5Dataset) -> Result<(Vec<f32>, Vec<usize>), String> {
    let shape = ds.shape();
    let starts = vec![0usize; shape.len()];
    let data = read_block_f32(ds, &starts, &shape)?;
    Ok((data, shape))
}

/// 确定用哪一个数据集作为网络输入。对应 `data_common.resolve_input_key`。
pub fn resolve_input_key(
    file: &H5File,
    input_key: Option<&str>,
    prefer_sensor_input: bool,
) -> Result<String, String> {
    let names = file.dataset_names();
    if let Some(k) = input_key {
        if !names.iter().any(|n| n == k) {
            return Err(format!("HDF5 中不存在数据集 {:?}", k));
        }
        return Ok(k.to_string());
    }
    let candidates: [&str; 2] = if prefer_sensor_input {
        ["sensor_input", "input"]
    } else {
        ["input", "sensor_input"]
    };
    for k in candidates {
        if names.iter().any(|n| n == k) {
            return Ok(k.to_string());
        }
    }
    Err(format!(
        "HDF5 中找不到 input / sensor_input 数据集，现有 {:?}",
        names
    ))
}

// ================================================================ 逐通道统计

/// 逐通道极值与均值/方差。对应 `compute_output_stats` 的返回字典。
#[derive(Debug, Clone, PartialEq)]
pub struct OutputStats {
    pub y_min: Vec<f32>,
    pub y_max: Vec<f32>,
    pub y_mean: Vec<f32>,
    pub y_std: Vec<f32>,
    pub n_samples: usize,
    pub shape: Vec<usize>,
}

impl OutputStats {
    /// max==min 的通道（常数场）。归一化后它们恒为 0，训练时等于白喂一个通道，
    /// 值得让调用方知道。
    pub fn degenerate_channels(&self) -> Vec<usize> {
        self.y_min
            .iter()
            .zip(&self.y_max)
            .enumerate()
            .filter(|(_, (lo, hi))| (**hi - **lo).abs() <= 1e-8 * (1.0 + hi.abs()))
            .map(|(i, _)| i)
            .collect()
    }

    /// 归一化后是否越界。Python 侧对应「统计量可能过期」的那条告警：统计量若来自
    /// 另一个文件，越界会静默发生，只有 R² 莫名其妙地差。
    pub fn out_of_range(lo: f32, hi: f32) -> bool {
        lo < -1e-3 || hi > 1.0 + 1e-3
    }
}

/// 分块累计逐通道统计。**独立于 HDF5**，这样「全量扫描」这条性质可以直接用黄金
/// 数据里的 `data/raw` 验证，不必先造一个文件。
#[derive(Debug, Clone)]
pub struct StatsAccumulator {
    n_channels: usize,
    y_min: Vec<f64>,
    y_max: Vec<f64>,
    total: Vec<f64>,
    total_sq: Vec<f64>,
    count: u64,
    n_samples: usize,
}

impl StatsAccumulator {
    pub fn new(n_channels: usize) -> Self {
        Self {
            n_channels,
            y_min: vec![f64::INFINITY; n_channels],
            y_max: vec![f64::NEG_INFINITY; n_channels],
            total: vec![0.0; n_channels],
            total_sq: vec![0.0; n_channels],
            count: 0,
            n_samples: 0,
        }
    }

    /// 喂一块形状 `(n, C, ...)` 的行优先数据。累加走 f64——f32 累加 1e6 个样本会
    /// 明显丢精度，而统计量的误差会随归一化传遍整个训练。
    pub fn push(&mut self, block: &[f32], n: usize) -> Result<(), String> {
        let c = self.n_channels;
        if n == 0 {
            return Ok(());
        }
        if !block.len().is_multiple_of(n * c) {
            return Err(format!(
                "数据块长度 {} 不是 n*C = {}*{} 的整数倍",
                block.len(),
                n,
                c
            ));
        }
        let per_channel = block.len() / (n * c); // 单样本单通道的空间元素数
        for (e, &v) in block.iter().enumerate() {
            // 行优先 (n, C, S)：下标 = ((i*C)+ch)*S + s
            let ch = (e / per_channel) % c;
            let v = v as f64;
            if v < self.y_min[ch] {
                self.y_min[ch] = v;
            }
            if v > self.y_max[ch] {
                self.y_max[ch] = v;
            }
            self.total[ch] += v;
            self.total_sq[ch] += v * v;
        }
        self.count += (n * per_channel) as u64;
        self.n_samples += n;
        Ok(())
    }

    /// `shape` 是整个数据集的形状，只用于回填 [`OutputStats::shape`]。
    pub fn finish(self, shape: Vec<usize>) -> OutputStats {
        let denom = self.count.max(1) as f64;
        let mean: Vec<f64> = self.total.iter().map(|t| t / denom).collect();
        let std: Vec<f32> = self
            .total_sq
            .iter()
            .zip(&mean)
            .map(|(sq, m)| ((sq / denom - m * m).max(0.0)).sqrt() as f32)
            .collect();
        OutputStats {
            y_min: self.y_min.iter().map(|v| *v as f32).collect(),
            y_max: self.y_max.iter().map(|v| *v as f32).collect(),
            y_mean: mean.iter().map(|v| *v as f32).collect(),
            y_std: std,
            n_samples: self.n_samples,
            shape,
        }
    }
}

/// 全量扫描 HDF5 数据集，返回逐通道统计量。对应 `compute_output_stats`。
///
/// **没有实现 Python 版的 `<file>.stats.npz` 缓存**：那份缓存是 numpy 格式，两个
/// 实现共用一个文件反而危险（谁写的、版本号对不对都要额外约定），而扫描本身是一次
/// 顺序读。要缓存时应当另起一个 Rust 侧格式，别去碰 Python 的。
pub fn compute_output_stats(path: &Path, key: &str, chunk: usize) -> Result<OutputStats, String> {
    let chunk = chunk.max(1);
    let file = H5File::open(path).map_err(|e| format!("打不开 {:?}: {}", path, e))?;
    let ds = file
        .dataset(key)
        .map_err(|e| format!("{:?} 里没有数据集 {:?}: {}", path, key, e))?;
    let shape = ds.shape();
    if shape.len() < 2 {
        return Err(format!("{:?} 至少要有 (N, C) 两维，实得 {:?}", key, shape));
    }
    let (n_samples, n_channels) = (shape[0], shape[1]);

    let mut acc = StatsAccumulator::new(n_channels);
    let mut start = 0usize;
    while start < n_samples {
        let n = chunk.min(n_samples - start);
        let mut starts = vec![0usize; shape.len()];
        starts[0] = start;
        let mut counts = shape.clone();
        counts[0] = n;
        acc.push(&read_block_f32(&ds, &starts, &counts)?, n)?;
        start += n;
    }
    Ok(acc.finish(shape))
}

// ================================================================ 归一化

/// 通道轴：带 batch（ndim>=4）时是 1，单样本时是 0。与 `normalize_array` 一致。
fn channel_axis(ndim: usize) -> usize {
    if ndim >= 4 {
        1
    } else {
        0
    }
}

fn per_channel_tensor<B: Backend, const D: usize>(
    v: &[f32],
    dims: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    let axis = channel_axis(D);
    assert_eq!(
        v.len(),
        dims[axis],
        "逐通道常量长度 {} 与通道数 {} 不符",
        v.len(),
        dims[axis]
    );
    let mut shape = [1usize; D];
    shape[axis] = v.len();
    Tensor::<B, 1>::from_data(TensorData::from(v), device).reshape(shape)
}

/// 逐通道 min-max 归一化到 [0, 1]。**不 clamp**。对应 `normalize_array`。
pub fn normalize<B: Backend, const D: usize>(
    y: Tensor<B, D>,
    y_min: &[f32],
    y_max: &[f32],
) -> Tensor<B, D> {
    let (dims, device) = (y.dims(), y.device());
    let lo = per_channel_tensor::<B, D>(y_min, dims, &device);
    let hi = per_channel_tensor::<B, D>(y_max, dims, &device);
    (y - lo.clone()) / (hi - lo).add_scalar(NORM_EPS)
}

/// [`normalize`] 的逆操作。对应 `denormalize_array`。
pub fn denormalize<B: Backend, const D: usize>(
    y: Tensor<B, D>,
    y_min: &[f32],
    y_max: &[f32],
) -> Tensor<B, D> {
    let (dims, device) = (y.dims(), y.device());
    let lo = per_channel_tensor::<B, D>(y_min, dims, &device);
    let hi = per_channel_tensor::<B, D>(y_max, dims, &device);
    y * (hi - lo.clone()).add_scalar(NORM_EPS) + lo
}

// ================================================================ numpy 的 MT19937

const MT_N: usize = 624;
const MT_M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// numpy `RandomState(seed)` 用的 MT19937。
///
/// 存在的唯一理由是 [`split_indices`] 必须和 Python 逐位一致——生态里的 Rust MT
/// 实现未必和 numpy 的 legacy seeding 与 `random_interval` 拒绝采样对得上，而对不上
/// 的表现是「两边划分不同」，没有任何报错。
pub struct Mt19937 {
    key: [u32; MT_N],
    pos: usize,
    /// randomkit 的高斯缓存：极坐标法一次出两个正态数，另一个存这儿下次直接用
    gauss: f64,
    has_gauss: bool,
}

impl Mt19937 {
    /// 对应 randomkit 的 `mt19937_seed`（与标准 `init_genrand` 等价，只是写法不同）。
    pub fn new(seed: u32) -> Self {
        let mut key = [0u32; MT_N];
        let mut s = seed;
        for (pos, slot) in key.iter_mut().enumerate() {
            *slot = s;
            s = 1812433253u32
                .wrapping_mul(s ^ (s >> 30))
                .wrapping_add(pos as u32 + 1);
        }
        Self {
            key,
            pos: MT_N,
            gauss: 0.0,
            has_gauss: false,
        }
    }

    fn regenerate(&mut self) {
        let twist = |u: u32, v: u32| {
            let y = (u & UPPER_MASK) | (v & LOWER_MASK);
            (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 }
        };
        for i in 0..MT_N - MT_M {
            self.key[i] = self.key[i + MT_M] ^ twist(self.key[i], self.key[i + 1]);
        }
        for i in MT_N - MT_M..MT_N - 1 {
            self.key[i] = self.key[i + MT_M - MT_N] ^ twist(self.key[i], self.key[i + 1]);
        }
        self.key[MT_N - 1] = self.key[MT_M - 1] ^ twist(self.key[MT_N - 1], self.key[0]);
        self.pos = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.pos == MT_N {
            self.regenerate();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// numpy 的 `random_interval(max)`：返回 **闭区间** [0, max] 内的均匀整数。
    /// 用「取最小覆盖位掩码 + 拒绝采样」，不是取模——取模会让分布偏斜，也和 numpy
    /// 的抽取次数对不上，序列从第一次拒绝起就分岔。
    pub fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let v = self.next_u32() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// 对应 `RandomState.random_sample()`（也就是 `rand()`）：[0, 1) 的 f64。
    ///
    /// randomkit 用两次 32 位抽取拼一个 53 位尾数，**顺序是先高 27 位再低 26 位**。
    /// 顺序调换后分布看着一样、统计量也正常，只是数值全变——所以这里不能凭直觉写。
    pub fn rand(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// 对应 `RandomState.randn()`（randomkit 的 `rk_gauss`）：Marsaglia 极坐标法。
    ///
    /// 一次生成两个正态数，缓存另一个。**缓存必须留着**——少缓存一个数就等于多消耗
    /// 两次均匀抽取，整条随机序列从此分岔。
    pub fn randn(&mut self) -> f64 {
        if self.has_gauss {
            self.has_gauss = false;
            let g = self.gauss;
            self.gauss = 0.0;
            return g;
        }
        let (f, x1, x2) = loop {
            let x1 = 2.0 * self.rand() - 1.0;
            let x2 = 2.0 * self.rand() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            // 落在单位圆外或正中心都要重抽，这两种拒绝都会推进序列
            if r2 < 1.0 && r2 != 0.0 {
                break ((-2.0 * r2.ln() / r2).sqrt(), x1, x2);
            }
        };
        self.gauss = f * x1;
        self.has_gauss = true;
        f * x2
    }

    /// 对应 `RandomState.shuffle`：从后往前的 Fisher-Yates。
    fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.interval(i as u32) as usize;
            x.swap(i, j);
        }
    }
}

// ================================================================ train/val 划分

/// Python 3 的 `round()`：五舍六入、**恰好 .5 时取偶**。Rust 的 `f64::round` 是
/// 四舍五入远离零，`round(2.5)` 给 3 而 Python 给 2——`n=5, val_split=0.5` 正好
/// 落在这个分歧上，且不会报错。
fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let frac = x - floor;
    if frac > 0.5 || (frac == 0.5 && (floor as i64) % 2 != 0) {
        floor + 1.0
    } else {
        floor
    }
}

/// 固定 seed 的 train/val 下标划分。`val_split <= 0` 时验证集为空。
///
/// 返回的两个数组都已升序排序且互不重叠，与 `data_common.split_indices` 逐位一致。
pub fn split_indices(n_samples: usize, val_split: f64, seed: u32) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n_samples).collect();
    if val_split <= 0.0 {
        return (idx, Vec::new());
    }
    Mt19937::new(seed).shuffle(&mut idx);

    let n_val = if n_samples > 1 {
        let raw = round_half_even(n_samples as f64 * val_split) as usize;
        raw.clamp(1, n_samples - 1)
    } else {
        0
    };
    let (val, train) = idx.split_at(n_val);
    let (mut train, mut val) = (train.to_vec(), val.to_vec());
    train.sort_unstable();
    val.sort_unstable();
    (train, val)
}

/// 批次切分，对应 `DataLoader(shuffle=..., drop_last=...)`。
///
/// **打乱用的是本文件的 MT19937，不是 PyTorch 的采样器**——两者的随机序列无法
/// 逐位对齐，黄金数据也没有约束它。这里只保证同 seed 可复现；端到端对照（§4.3）
/// 比的是 loss 趋势，不是逐 batch 的样本顺序。
pub fn batch_indices(
    n: usize,
    batch_size: usize,
    shuffle_seed: Option<u32>,
    drop_last: bool,
) -> Vec<Vec<usize>> {
    assert!(batch_size > 0, "batch_size 必须为正");
    let mut order: Vec<usize> = (0..n).collect();
    if let Some(seed) = shuffle_seed {
        Mt19937::new(seed).shuffle(&mut order);
    }
    order
        .chunks(batch_size)
        .filter(|c| !drop_last || c.len() == batch_size)
        .map(|c| c.to_vec())
        .collect()
}

// ================================================================ 2D 惰性数据集

/// [`CfdDataset::open`] 的选项。Rust 没有关键字参数，用结构体 +
/// `..Default::default()` 对齐 Python 那一长串默认值。
#[derive(Debug, Clone)]
pub struct DatasetOptions {
    pub input_key: Option<String>,
    pub prefer_sensor_input: bool,
    pub align_input_to_output: bool,
    /// 空间尺寸下限，供小图放大到网络能吃的最小尺寸。
    pub min_spatial_size: usize,
    /// 只取这些样本（train/val 划分用）。`None` = 全量。
    pub indices: Option<Vec<usize>>,
    /// 外部注入的统计量。测试集必须复用训练集的，否则两边被映射到不同标度，
    /// R² 不可比。
    pub stats: Option<OutputStats>,
    pub stats_chunk: usize,
}

impl Default for DatasetOptions {
    fn default() -> Self {
        Self {
            input_key: None,
            prefer_sensor_input: false,
            align_input_to_output: true,
            min_spatial_size: 7,
            indices: None,
            stats: None,
            stats_chunk: 256,
        }
    }
}

/// 一批样本的原始读取结果：`(x 扁平数据, x 形状, y 扁平数据, y 形状)`。
type RawBatch = (Vec<f32>, Vec<usize>, Vec<f32>, Vec<usize>);

/// 惰性读取的 HDF5 数据集，target 逐通道归一化到 [0, 1]。
/// 对应 `load_data.CFDHDF5Dataset`。
///
/// 句柄在 `open` 时拿一次、整个生命周期复用——Python 旧版是每取一个样本
/// open/close 一次文件。
pub struct CfdDataset {
    _file: H5File,
    x: H5Dataset,
    y: H5Dataset,
    x_shape: Vec<usize>,
    y_shape: Vec<usize>,
    indices: Vec<usize>,
    align_input_to_output: bool,
    min_spatial_size: usize,
    pub stats: OutputStats,
}

impl CfdDataset {
    pub fn open(path: &Path, opts: &DatasetOptions) -> Result<Self, String> {
        let file = H5File::open(path).map_err(|e| format!("打不开 {:?}: {}", path, e))?;
        let input_key =
            resolve_input_key(&file, opts.input_key.as_deref(), opts.prefer_sensor_input)?;
        let x = file
            .dataset(&input_key)
            .map_err(|e| format!("打不开数据集 {:?}: {}", input_key, e))?;
        let y = file
            .dataset("output")
            .map_err(|e| format!("{:?} 里没有 output 数据集: {}", path, e))?;
        let (x_shape, y_shape) = (x.shape(), y.shape());
        if y_shape.len() < 3 {
            return Err(format!(
                "output 至少要有 (N, C, ...) 三维，实得 {:?}",
                y_shape
            ));
        }
        if x_shape[0] != y_shape[0] {
            return Err(format!(
                "输入样本数 {} 与输出样本数 {} 不一致",
                x_shape[0], y_shape[0]
            ));
        }

        let stats = match &opts.stats {
            Some(s) => s.clone(),
            None => compute_output_stats(path, "output", opts.stats_chunk)?,
        };
        if stats.y_min.len() != y_shape[1] {
            return Err(format!(
                "统计量有 {} 个通道，output 有 {} 个——多半是另一个文件的统计量",
                stats.y_min.len(),
                y_shape[1]
            ));
        }

        let n_total = y_shape[0];
        let indices = match &opts.indices {
            Some(v) => {
                if let Some(bad) = v.iter().find(|i| **i >= n_total) {
                    return Err(format!("下标 {} 越界，样本总数 {}", bad, n_total));
                }
                v.clone()
            }
            None => (0..n_total).collect(),
        };

        Ok(Self {
            _file: file,
            x,
            y,
            x_shape,
            y_shape,
            indices,
            align_input_to_output: opts.align_input_to_output,
            min_spatial_size: opts.min_spatial_size,
            stats,
        })
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// 本数据集使用的归一化统计量。训练循环存 checkpoint 时要把它一并写进去——
    /// **推理侧靠它把网络输出还原成物理量纲**，缺了就只能靠猜，而猜错不报错。
    pub fn stats(&self) -> &OutputStats {
        &self.stats
    }

    /// output 的完整形状 `(N, C, ...)`。
    pub fn y_shape(&self) -> &[usize] {
        &self.y_shape
    }

    pub fn x_shape(&self) -> &[usize] {
        &self.x_shape
    }

    /// 读一批样本的原始数据。`sel` 是 [`Self::indices`] 里的位置，不是文件里的下标。
    fn read_pair(&self, sel: &[usize]) -> Result<RawBatch, String> {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for &s in sel {
            let idx = *self
                .indices
                .get(s)
                .ok_or_else(|| format!("批次位置 {} 越界，本子集共 {} 个样本", s, self.len()))?;
            xs.extend(read_block_f32(
                &self.x,
                &sample_starts(&self.x_shape, idx),
                &sample_counts(&self.x_shape),
            )?);
            ys.extend(read_block_f32(
                &self.y,
                &sample_starts(&self.y_shape, idx),
                &sample_counts(&self.y_shape),
            )?);
        }
        let mut x_dims = self.x_shape.clone();
        let mut y_dims = self.y_shape.clone();
        x_dims[0] = sel.len();
        y_dims[0] = sel.len();
        // 输入缺通道维时补一维，对应 Python 的 `x.unsqueeze(0)`
        if x_dims.len() == y_dims.len() - 1 {
            x_dims.insert(1, 1);
        }
        Ok((xs, x_dims, ys, y_dims))
    }

    /// 取一批 4D 样本：`(x, y)`，`y` 已归一化，`x` 已按需对齐到 `y` 的空间尺寸。
    pub fn batch4<B: Backend>(
        &self,
        sel: &[usize],
        device: &B::Device,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>), String> {
        let (xv, xd, yv, yd) = self.read_pair(sel)?;
        let x = to_tensor::<B, 4>(&xv, &xd, device)?;
        let y = normalize(
            to_tensor::<B, 4>(&yv, &yd, device)?,
            &self.stats.y_min,
            &self.stats.y_max,
        );
        if !self.align_input_to_output {
            return Ok((x, y));
        }
        let [_, _, h, w] = y.dims();
        let target = [h.max(self.min_spatial_size), w.max(self.min_spatial_size)];
        let x = if x.dims()[2..] == target {
            x
        } else {
            resize_bilinear(x, target)
        };
        Ok((x, y))
    }

    /// 取一批 5D 样本（文件本身是 `(N, C, D, H, W)`）。
    pub fn batch5<B: Backend>(
        &self,
        sel: &[usize],
        device: &B::Device,
    ) -> Result<(Tensor<B, 5>, Tensor<B, 5>), String> {
        let (xv, xd, yv, yd) = self.read_pair(sel)?;
        let x = to_tensor::<B, 5>(&xv, &xd, device)?;
        let y = normalize(
            to_tensor::<B, 5>(&yv, &yd, device)?,
            &self.stats.y_min,
            &self.stats.y_max,
        );
        if !self.align_input_to_output {
            return Ok((x, y));
        }
        let [_, _, d, h, w] = y.dims();
        let m = self.min_spatial_size;
        let target = [d.max(m), h.max(m), w.max(m)];
        let x = if x.dims()[2..] == target {
            x
        } else {
            resize_trilinear(x, target)
        };
        Ok((x, y))
    }
}

fn sample_starts(shape: &[usize], idx: usize) -> Vec<usize> {
    let mut s = vec![0usize; shape.len()];
    s[0] = idx;
    s
}

fn sample_counts(shape: &[usize]) -> Vec<usize> {
    let mut c = shape.to_vec();
    c[0] = 1;
    c
}

fn to_tensor<B: Backend, const D: usize>(
    data: &[f32],
    dims: &[usize],
    device: &B::Device,
) -> Result<Tensor<B, D>, String> {
    if dims.len() != D {
        return Err(format!("期望 {} 维，实得形状 {:?}", D, dims));
    }
    let n: usize = dims.iter().product();
    if n != data.len() {
        return Err(format!(
            "形状 {:?} 需要 {} 个元素，实得 {}",
            dims,
            n,
            data.len()
        ));
    }
    let mut shape = [1usize; D];
    shape.copy_from_slice(dims);
    Ok(Tensor::<B, 1>::from_data(TensorData::from(data), device).reshape(shape))
}

/// 从同一个文件切出互不重叠的训练集 / 验证集。对应 `load_data.load_train_val`。
///
/// **统计量只算一次并共用**：两边各自算极值会把它们映射到不同标度，R² 不可比。
pub fn load_train_val(
    path: &Path,
    val_split: f64,
    seed: u32,
    opts: &DatasetOptions,
) -> Result<(CfdDataset, Option<CfdDataset>), String> {
    let file = H5File::open(path).map_err(|e| format!("打不开 {:?}: {}", path, e))?;
    let n_samples = file
        .dataset("output")
        .map_err(|e| format!("{:?} 里没有 output 数据集: {}", path, e))?
        .shape()[0];
    drop(file);

    let (train_idx, val_idx) = split_indices(n_samples, val_split, seed);
    let stats = match &opts.stats {
        Some(s) => s.clone(),
        None => compute_output_stats(path, "output", opts.stats_chunk)?,
    };

    let mut train_opts = opts.clone();
    train_opts.stats = Some(stats.clone());
    train_opts.indices = Some(train_idx);
    let train = CfdDataset::open(path, &train_opts)?;

    if val_idx.is_empty() {
        return Ok((train, None));
    }
    let mut val_opts = opts.clone();
    val_opts.stats = Some(stats);
    val_opts.indices = Some(val_idx);
    Ok((train, Some(CfdDataset::open(path, &val_opts)?)))
}

// ================================================================ 3D 加载器

/// [`load_3d`] 的选项，对应 `load_data_3d.load_data` 的关键字参数。
#[derive(Debug, Clone)]
pub struct Options3d {
    pub input_key: Option<String>,
    pub prefer_sensor_input: bool,
    pub align_input_to_output: bool,
    /// `spatial_input` 存在时，把 `input` / `sensor_input` 的标量向量平铺成体素通道
    /// 并拼到空间输入后面。
    pub combine_inputs: bool,
    pub indices: Option<Vec<usize>>,
    pub stats: Option<OutputStats>,
    pub stats_chunk: usize,
}

impl Default for Options3d {
    fn default() -> Self {
        Self {
            input_key: None,
            prefer_sensor_input: false,
            align_input_to_output: true,
            combine_inputs: true,
            indices: None,
            stats: None,
            stats_chunk: 256,
        }
    }
}

/// 把任意维度的输入补成 5D `(N, C, D, H, W)`，只改形状不动数据。
fn reshape_to_5d(shape: &[usize]) -> Result<[usize; 5], String> {
    match shape {
        [n, h, w] => Ok([*n, 1, 1, *h, *w]),
        [n, c, h, w] => Ok([*n, *c, 1, *h, *w]),
        [n, c, d, h, w] => Ok([*n, *c, *d, *h, *w]),
        other => Err(format!(
            "不支持的输入维度 {}（形状 {:?}）",
            other.len(),
            other
        )),
    }
}

/// 把每个样本的标量特征向量平铺成体素通道，对应 `_vector_to_5d` 的 `np.tile`。
fn vector_to_5d(data: &[f32], n: usize, d: usize, h: usize, w: usize) -> (Vec<f32>, [usize; 5]) {
    let f = data.len() / n.max(1);
    let voxels = d * h * w;
    let mut out = Vec::with_capacity(n * f * voxels);
    for i in 0..n {
        for j in 0..f {
            out.extend(std::iter::repeat_n(data[i * f + j], voxels));
        }
    }
    (out, [n, f, d, h, w])
}

/// 3D 体素加载器，对应 `load_data_3d.load_data`。**一次性读进内存**，与 Python 版
/// 一致——3D 数据本来就要整块喂给网络，惰性读没意义。
///
/// 返回 `(x, y, stats)`，`y` 已逐通道归一化。
pub fn load_3d<B: Backend>(
    path: &Path,
    opts: &Options3d,
    device: &B::Device,
) -> Result<(Tensor<B, 5>, Tensor<B, 5>, OutputStats), String> {
    let file = H5File::open(path).map_err(|e| format!("打不开 {:?}: {}", path, e))?;
    let yds = file
        .dataset("output")
        .map_err(|e| format!("{:?} 里没有 output 数据集: {}", path, e))?;
    let (y_data, y_shape) = read_all_f32(&yds)?;
    let y_dims: [usize; 5] = match y_shape.as_slice() {
        [n, c, h, w] => [*n, *c, 1, *h, *w],
        [n, c, d, h, w] => [*n, *c, *d, *h, *w],
        other => {
            return Err(format!(
                "不支持的输出维度 {}（形状 {:?}）",
                other.len(),
                other
            ))
        }
    };
    let n_samples = y_dims[0];

    let x = read_inputs_3d::<B>(&file, opts, n_samples, device)?;

    let stats = match &opts.stats {
        Some(s) => s.clone(),
        None => compute_output_stats(path, "output", opts.stats_chunk)?,
    };
    // 文件里 output 是 4D 时，补深度维不改变通道轴，统计量可直接用
    if stats.y_min.len() != y_dims[1] {
        return Err(format!(
            "统计量有 {} 个通道，output 有 {} 个——多半是另一个文件的统计量",
            stats.y_min.len(),
            y_dims[1]
        ));
    }
    let y = normalize(
        to_tensor::<B, 5>(&y_data, &y_dims, device)?,
        &stats.y_min,
        &stats.y_max,
    );

    let x = if opts.align_input_to_output && x.dims()[2..] != y.dims()[2..] {
        let [_, _, d, h, w] = y.dims();
        resize_trilinear(x, [d, h, w])
    } else {
        x
    };

    let (x, y) = match &opts.indices {
        None => (x, y),
        Some(sel) => {
            if let Some(bad) = sel.iter().find(|i| **i >= n_samples) {
                return Err(format!("下标 {} 越界，样本总数 {}", bad, n_samples));
            }
            let ids: Vec<i32> = sel.iter().map(|i| *i as i32).collect();
            let ids = Tensor::<B, 1, Int>::from_data(TensorData::from(ids.as_slice()), device);
            (x.select(0, ids.clone()), y.select(0, ids))
        }
    };
    Ok((x, y, stats))
}

fn read_inputs_3d<B: Backend>(
    file: &H5File,
    opts: &Options3d,
    n_samples: usize,
    device: &B::Device,
) -> Result<Tensor<B, 5>, String> {
    let names = file.dataset_names();
    let read5 = |key: &str| -> Result<Tensor<B, 5>, String> {
        let ds = file
            .dataset(key)
            .map_err(|e| format!("打不开数据集 {:?}: {}", key, e))?;
        let (data, shape) = read_all_f32(&ds)?;
        // (N, F) 的纯特征向量得先摆成方形网格，否则根本没有空间结构可卷
        if let [n, feats] = shape.as_slice() {
            let side = (*feats as f64).sqrt().round() as usize;
            if side * side != *feats {
                return Err(format!(
                    "二维输入 {:?} 无法重排成方形网格，请提供 spatial_input 或用 input_key 指定带空间结构的数据集",
                    shape
                ));
            }
            return to_tensor::<B, 5>(&data, &[*n, 1, 1, side, side], device);
        }
        let dims = reshape_to_5d(&shape)?;
        to_tensor::<B, 5>(&data, &dims, device)
    };

    if let Some(key) = &opts.input_key {
        if !names.iter().any(|n| n == key) {
            return Err(format!("HDF5 中不存在数据集 {:?}", key));
        }
        return read5(key);
    }

    if opts.combine_inputs && names.iter().any(|n| n == "spatial_input") {
        let spatial = read5("spatial_input")?;
        let [_, _, d, h, w] = spatial.dims();
        let mut parts = vec![spatial];
        for extra in ["input", "sensor_input"] {
            if !names.iter().any(|n| n == extra) {
                continue;
            }
            let ds = file
                .dataset(extra)
                .map_err(|e| format!("打不开数据集 {:?}: {}", extra, e))?;
            let (data, _) = read_all_f32(&ds)?;
            let (tiled, dims) = vector_to_5d(&data, n_samples, d, h, w);
            parts.push(to_tensor::<B, 5>(&tiled, &dims, device)?);
        }
        return Ok(Tensor::cat(parts, 1));
    }

    read5(&resolve_input_key(file, None, opts.prefer_sensor_input)?)
}

// ================================================================ R² / RMSE 累计器

/// 按掩码累计 SSE / SST，支持整体与分通道 R²。对应 `utils/misc.py` 的
/// `RunningMetrics`。
///
/// 分通道是必需的：三个通道池化成一个 R² 时，压力场好、速度场差会被平均掩盖。
#[derive(Debug, Clone)]
pub struct RunningMetrics {
    n_channels: usize,
    sse: Vec<f64>,
    sum_y: Vec<f64>,
    sum_y2: Vec<f64>,
    count: Vec<f64>,
}

impl RunningMetrics {
    pub fn new(n_channels: usize) -> Self {
        Self {
            n_channels,
            sse: vec![0.0; n_channels],
            sum_y: vec![0.0; n_channels],
            sum_y2: vec![0.0; n_channels],
            count: vec![0.0; n_channels],
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.n_channels);
    }

    /// `output` / `target` 形状 `(B, C, ...)`；`mask` 可广播（形如 `(1, 1, H, W)`）。
    pub fn update<B: Backend, const D: usize>(
        &mut self,
        output: Tensor<B, D>,
        target: Tensor<B, D>,
        mask: Option<Tensor<B, D>>,
    ) {
        assert!(D >= 2, "指标累计器需要 (B, C, ...) 至少两维");
        let dims = target.dims();
        assert_eq!(
            dims[1], self.n_channels,
            "通道数 {} 与累计器的 {} 不符",
            dims[1], self.n_channels
        );
        // 掩码要先铺开到 target 的形状：count 的分母是掩码内像素数，
        // 用未展开的掩码求和会把分母算成 H*W 而不是 B*H*W
        let weight = match mask {
            Some(m) => m * target.clone().ones_like(),
            None => target.clone().ones_like(),
        };
        let err2 = (output - target.clone()).powi_scalar(2);
        accumulate(&mut self.sse, channel_sums(err2 * weight.clone()));
        accumulate(
            &mut self.sum_y,
            channel_sums(target.clone() * weight.clone()),
        );
        accumulate(
            &mut self.sum_y2,
            channel_sums(target.powi_scalar(2) * weight.clone()),
        );
        accumulate(&mut self.count, channel_sums(weight));
    }

    fn sst(&self) -> Vec<f64> {
        self.sum_y
            .iter()
            .zip(&self.sum_y2)
            .zip(&self.count)
            .map(|((sy, sy2), n)| (sy2 - sy * sy / n.max(1.0)).max(0.0))
            .collect()
    }

    /// 分通道 R²。SST 退化（常数场）时给 `NaN`，而不是拿 0 顶上——那会让报表上
    /// 出现一个看起来「不算差」的 0.0。
    pub fn r2_per_channel(&self) -> Vec<f64> {
        self.sst()
            .iter()
            .zip(&self.sse)
            .map(|(sst, sse)| {
                if *sst > 1e-12 {
                    1.0 - sse / sst.max(1e-12)
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// 全局 R²（各通道 SSE / SST 汇总）。
    pub fn r2(&self) -> f64 {
        let sst: f64 = self.sst().iter().sum();
        1.0 - self.sse.iter().sum::<f64>() / sst.max(1e-12)
    }

    pub fn rmse(&self) -> f64 {
        (self.sse.iter().sum::<f64>() / self.count.iter().sum::<f64>().max(1.0)).sqrt()
    }

    pub fn rmse_per_channel(&self) -> Vec<f64> {
        self.sse
            .iter()
            .zip(&self.count)
            .map(|(sse, n)| (sse / n.max(1.0)).sqrt())
            .collect()
    }
}

fn accumulate(dst: &mut [f64], src: Vec<f64>) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// 除通道轴以外全部归约。**必须先换轴再压平**：直接 `reshape([c, -1])` 会把 batch
/// 维和通道维搅在一起，得到的「分通道」指标其实是别的通道的数。
fn channel_sums<B: Backend, const D: usize>(x: Tensor<B, D>) -> Vec<f64> {
    let dims = x.dims();
    let c = dims[1];
    let rest: usize = dims.iter().product::<usize>() / c;
    x.swap_dims(0, 1)
        .reshape([c, rest])
        .sum_dim(1)
        .into_data()
        .to_vec::<f32>()
        .expect("通道求和结果读不出来")
        .into_iter()
        .map(|v| v as f64)
        .collect()
}

// ================================================================ 单元测试

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn mt19937_matches_numpy_first_draws() {
        // 参考值来自 numpy：np.random.RandomState(1).tomaxint 之前的原始 32 位抽取，
        // 等价于 RandomState(1).randint(0, 2**32, 5, dtype='uint32')
        let mut r = Mt19937::new(1);
        let got: Vec<u32> = (0..5).map(|_| r.next_u32()).collect();
        assert_eq!(
            got,
            vec![1791095845, 4282876139, 3093770124, 4005303368, 491263]
        );
    }

    #[test]
    fn rand_matches_numpy_random_sample() {
        // 等价于 RandomState(0).rand() 连抽 5 次；f64 逐位相等，不是「接近」
        let mut r = Mt19937::new(0);
        let got: Vec<f64> = (0..5).map(|_| r.rand()).collect();
        assert_eq!(
            got,
            vec![
                0.5488135039273248,
                0.7151893663724195,
                0.6027633760716439,
                0.5448831829968969,
                0.4236547993389047
            ]
        );
    }

    #[test]
    fn randn_shares_the_stream_with_rand() {
        // 等价于 RandomState(13) 下交替 rand()/randn() 各 4 次。
        // 交替抽是关键：randn 的缓存若丢了，后续 rand 会读到错位的值。
        let mut r = Mt19937::new(13);
        let got: Vec<f64> = (0..4).flat_map(|_| [r.rand(), r.randn()]).collect();
        assert_eq!(
            got,
            vec![
                0.7777024105738202,
                0.6615502260395926,
                0.9657491980429997,
                -0.5354337328067726,
                0.9726011139048933,
                2.2066354572659455,
                0.7755265146048467,
                -0.9420233055103951
            ]
        );
    }

    #[test]
    fn interval_matches_numpy_randint() {
        // 等价于 RandomState(11) 下循环 rand() / randint(2) / randint(4)。
        // numpy 的 randint 在上界不超过 2^32 时走 32 位掩码拒绝采样，正是 interval。
        let mut r = Mt19937::new(11);
        let got: Vec<(f64, u32, u32)> = (0..4)
            .map(|_| (r.rand(), r.interval(1), r.interval(3)))
            .collect();
        assert_eq!(
            got,
            vec![
                (0.1802696888767692, 0, 3),
                (0.4632185264983446, 1, 0),
                (0.42020360458772743, 0, 0),
                (0.012780814590608647, 0, 1)
            ]
        );
    }

    #[test]
    fn round_half_even_differs_from_rust_round() {
        assert_eq!(round_half_even(2.5), 2.0); // Rust 的 f64::round 给 3
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(round_half_even(11.1), 11.0);
        assert_eq!(round_half_even(20.0), 20.0);
    }

    #[test]
    fn split_is_disjoint_and_covers_everything() {
        let (tr, va) = split_indices(37, 0.3, 7);
        assert_eq!(tr.len() + va.len(), 37);
        let mut all: Vec<usize> = tr.iter().chain(&va).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..37).collect::<Vec<_>>());
        assert!(tr.windows(2).all(|w| w[0] < w[1]), "训练下标应升序且不重复");
    }

    #[test]
    fn split_with_no_val_keeps_original_order() {
        let (tr, va) = split_indices(10, 0.0, 1);
        assert_eq!(tr, (0..10).collect::<Vec<_>>());
        assert!(va.is_empty());
    }

    #[test]
    fn val_size_never_swallows_the_whole_set() {
        // val_split=0.99 时验证集最多留 n-1 个，训练集不能空
        let (tr, va) = split_indices(4, 0.99, 1);
        assert_eq!(tr.len(), 1);
        assert_eq!(va.len(), 3);
        // val_split 极小也至少给一个验证样本，否则「有验证集」是假的
        let (_, va) = split_indices(100, 0.001, 1);
        assert_eq!(va.len(), 1);
    }

    #[test]
    fn stats_are_chunk_size_independent() {
        // 分块扫描不能改变结果——「只扫前 N 个样本」正是旧版的缺陷
        let data: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.5 - 7.0).collect();
        let mut whole = StatsAccumulator::new(3);
        whole.push(&data, 2).unwrap();
        let mut split = StatsAccumulator::new(3);
        split.push(&data[..12], 1).unwrap();
        split.push(&data[12..], 1).unwrap();
        assert_eq!(whole.finish(vec![2, 3, 4]), split.finish(vec![2, 3, 4]));
    }

    #[test]
    fn normalize_roundtrip_on_3d_uses_axis_0() {
        // 单样本 (C, H, W) 的通道轴是 0，用错轴会拿别的通道的极值去缩放
        let device = Default::default();
        let x = Tensor::<B, 1>::from_data(
            TensorData::from([0.0f32, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0]),
            &device,
        )
        .reshape([2, 2, 2]);
        let (lo, hi) = ([0.0f32, 100.0], [3.0f32, 103.0]);
        let n = normalize(x.clone(), &lo, &hi);
        let v: Vec<f32> = n.clone().into_data().to_vec().unwrap();
        assert!(
            v.iter().all(|e| (-1e-6..=1.0 + 1e-6).contains(e)),
            "{:?}",
            v
        );
        let back: Vec<f32> = denormalize(n, &lo, &hi).into_data().to_vec().unwrap();
        let orig: Vec<f32> = x.into_data().to_vec().unwrap();
        for (b, o) in back.iter().zip(&orig) {
            assert!((b - o).abs() < 1e-3, "往返失真 {} vs {}", b, o);
        }
    }

    #[test]
    fn constant_channel_is_flagged_not_nan() {
        let s = OutputStats {
            y_min: vec![1.0, 0.0],
            y_max: vec![1.0, 5.0],
            y_mean: vec![1.0, 2.5],
            y_std: vec![0.0, 1.0],
            n_samples: 1,
            shape: vec![1, 2, 1, 1],
        };
        assert_eq!(s.degenerate_channels(), vec![0]);
        let device = Default::default();
        let x =
            Tensor::<B, 1>::from_data(TensorData::from([1.0f32, 2.5]), &device).reshape([2, 1, 1]);
        let v: Vec<f32> = normalize(x, &s.y_min, &s.y_max)
            .into_data()
            .to_vec()
            .unwrap();
        assert!(
            v.iter().all(|e| e.is_finite()),
            "常数通道不应出 NaN: {:?}",
            v
        );
    }

    #[test]
    fn batches_cover_every_sample_exactly_once() {
        let batches = batch_indices(10, 3, Some(1), false);
        assert_eq!(
            batches.iter().map(|b| b.len()).collect::<Vec<_>>(),
            vec![3, 3, 3, 1]
        );
        let mut all: Vec<usize> = batches.into_iter().flatten().collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
        // drop_last 丢掉不满一批的尾巴
        assert_eq!(batch_indices(10, 3, None, true).len(), 3);
        // 同 seed 必须复现
        assert_eq!(
            batch_indices(10, 3, Some(5), false),
            batch_indices(10, 3, Some(5), false)
        );
    }

    #[test]
    fn metrics_perfect_prediction_gives_r2_one() {
        let device = Default::default();
        let t = Tensor::<B, 1>::from_data(
            TensorData::from([1.0f32, 2.0, 3.0, 4.0, 9.0, 8.0, 7.0, 6.0]),
            &device,
        )
        .reshape([1, 2, 2, 2]);
        let mut m = RunningMetrics::new(2);
        m.update(t.clone(), t.clone(), None);
        assert!((m.r2() - 1.0).abs() < 1e-6);
        assert!(m.rmse() < 1e-6);
    }

    #[test]
    fn metrics_mask_shrinks_the_denominator() {
        // 掩码外的像素既不进 SSE 也不进 count，否则 RMSE 被域外的零值稀释
        let device = Default::default();
        let out = Tensor::<B, 1>::from_data(TensorData::from([0.0f32, 0.0, 0.0, 0.0]), &device)
            .reshape([1, 1, 2, 2]);
        let tgt = Tensor::<B, 1>::from_data(TensorData::from([2.0f32, 0.0, 0.0, 0.0]), &device)
            .reshape([1, 1, 2, 2]);
        let mask = Tensor::<B, 1>::from_data(TensorData::from([1.0f32, 1.0, 0.0, 0.0]), &device)
            .reshape([1, 1, 2, 2]);
        let mut m = RunningMetrics::new(1);
        m.update(out, tgt, Some(mask));
        // SSE=4，掩码内 2 个像素 -> RMSE = sqrt(4/2)
        assert!((m.rmse() - 2f64.sqrt()).abs() < 1e-6, "rmse={}", m.rmse());
    }

    #[test]
    fn metrics_constant_target_gives_nan_not_zero() {
        let device = Default::default();
        let t = Tensor::<B, 1>::from_data(TensorData::from([3.0f32, 3.0, 3.0, 3.0]), &device)
            .reshape([1, 1, 2, 2]);
        let o = Tensor::<B, 1>::from_data(TensorData::from([3.0f32, 3.1, 2.9, 3.0]), &device)
            .reshape([1, 1, 2, 2]);
        let mut m = RunningMetrics::new(1);
        m.update(o, t, None);
        assert!(m.r2_per_channel()[0].is_nan());
    }

    #[test]
    fn vector_tile_repeats_per_sample_not_globally() {
        let (out, dims) = vector_to_5d(&[1.0, 2.0, 3.0, 4.0], 2, 1, 2, 1);
        assert_eq!(dims, [2, 2, 1, 2, 1]);
        assert_eq!(out, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
    }

    #[test]
    fn reshape_to_5d_inserts_depth_not_channel() {
        assert_eq!(reshape_to_5d(&[7, 3, 8, 9]).unwrap(), [7, 3, 1, 8, 9]);
        assert_eq!(reshape_to_5d(&[7, 8, 9]).unwrap(), [7, 1, 1, 8, 9]);
        assert!(reshape_to_5d(&[7, 8]).is_err());
    }
}
