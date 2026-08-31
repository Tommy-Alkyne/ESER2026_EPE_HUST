//! P4 数据管线的黄金数据验收：统计量、归一化往返、train/val 划分、R²/RMSE 累计器。
//!
//! 逐元素对 `tests/golden/golden.npz` 的 `data/*` 与 `metrics/*`，容差 1e-5。
//! 另有两组 HDF5 通路测试，验证 `rust-hdf5` 真能读出本项目的数据文件。

mod common;

use std::path::PathBuf;

use common::{assert_close4, assert_close_f64, open_npz, read, read_vec, B, TOL};
use eser::data::{
    batch_indices, compute_output_stats, denormalize, load_3d, load_train_val, normalize,
    split_indices, CfdDataset, DatasetOptions, Options3d, RunningMetrics, StatsAccumulator,
};
use rust_hdf5::H5File;

// ================================================================ data/*

/// 全量分块扫描的统计量。**故意用 chunk=2 分三块喂**：旧 Python 版只扫前 N 个样本
/// 就定极值，分块结果必须与整块一致才说明扫全了。
#[test]
fn stats_match_golden() {
    let mut npz = open_npz();
    let raw = read(&mut npz, "data/raw");
    let dims = raw.shape().to_vec();
    let (n, c) = (dims[0], dims[1]);
    let per_sample: usize = dims[1..].iter().product();
    let flat: Vec<f32> = raw.iter().copied().collect();

    let mut acc = StatsAccumulator::new(c);
    for start in (0..n).step_by(2) {
        let k = 2.min(n - start);
        acc.push(&flat[start * per_sample..(start + k) * per_sample], k)
            .unwrap();
    }
    let stats = acc.finish(dims);

    for (i, want) in read_vec(&mut npz, "data/y_min").iter().enumerate() {
        assert_close_f64(
            &format!("data/y_min[{}]", i),
            stats.y_min[i] as f64,
            *want as f64,
        );
    }
    for (i, want) in read_vec(&mut npz, "data/y_max").iter().enumerate() {
        assert_close_f64(
            &format!("data/y_max[{}]", i),
            stats.y_max[i] as f64,
            *want as f64,
        );
    }
    assert_eq!(stats.n_samples, n);
}

#[test]
fn normalize_and_roundtrip_match_golden() {
    let mut npz = open_npz();
    let raw = read(&mut npz, "data/raw");
    let y_min = read_vec(&mut npz, "data/y_min");
    let y_max = read_vec(&mut npz, "data/y_max");
    let x = common::to_tensor4(&raw);

    let norm = normalize(x, &y_min, &y_max);
    assert_close4(
        "data/normalized",
        norm.clone(),
        &read(&mut npz, "data/normalized"),
    );
    assert_close4(
        "data/roundtrip",
        denormalize(norm, &y_min, &y_max),
        &read(&mut npz, "data/roundtrip"),
    );
}

/// 划分下标必须**逐位**一致。同 seed 不同序列不会报错，只会让两个实现的验证集
/// 悄悄换成另一批样本，此后所有 R² 对照都是假的。
#[test]
fn split_indices_match_golden_bit_for_bit() {
    let mut npz = open_npz();
    for (n, tag, vs, seed) in [
        (100usize, "100_0.2_1", 0.2, 1u32),
        (37, "37_0.3_7", 0.3, 7),
        (5, "5_0.5_3", 0.5, 3),
    ] {
        let (train, val) = split_indices(n, vs, seed);
        for (name, got) in [("train", &train), ("val", &val)] {
            let want: Vec<usize> = read_vec(&mut npz, &format!("data/split_{}_{}", tag, name))
                .iter()
                .map(|v| *v as usize)
                .collect();
            assert_eq!(
                got, &want,
                "data/split_{}_{}: 下标不一致\ngot ={:?}\nwant={:?}",
                tag, name, got, want
            );
        }
        // 训练/验证不重叠，且合起来是全集
        let mut all = train.clone();
        all.extend(&val);
        all.sort_unstable();
        assert_eq!(
            all,
            (0..n).collect::<Vec<_>>(),
            "split_{} 有重叠或漏样本",
            tag
        );
    }
}

// ================================================================ metrics/*

#[test]
fn running_metrics_match_golden() {
    let mut npz = open_npz();
    let out = common::tensor4(&mut npz, "metrics/out");
    let tgt = common::tensor4(&mut npz, "metrics/tgt");
    let mask = common::tensor4(&mut npz, "metrics/mask");

    let mut m = RunningMetrics::new(3);
    m.update(out, tgt, Some(mask));

    assert_close_f64(
        "metrics/r2",
        m.r2(),
        common::read_scalar(&mut npz, "metrics/r2"),
    );
    assert_close_f64(
        "metrics/rmse",
        m.rmse(),
        common::read_scalar(&mut npz, "metrics/rmse"),
    );
    for (i, want) in read_vec(&mut npz, "metrics/r2_per_channel")
        .iter()
        .enumerate()
    {
        assert_close_f64(
            &format!("metrics/r2_per_channel[{}]", i),
            m.r2_per_channel()[i],
            *want as f64,
        );
    }
}

/// 分批累计与一次性累计必须给同一个结果——训练时指标是逐 batch 累加出来的。
#[test]
fn metrics_accumulate_across_batches() {
    let mut npz = open_npz();
    let out = common::tensor4(&mut npz, "metrics/out");
    let tgt = common::tensor4(&mut npz, "metrics/tgt");
    let mask = common::tensor4(&mut npz, "metrics/mask");
    let dims = out.dims();

    let mut split = RunningMetrics::new(3);
    for b in 0..dims[0] {
        let range = [b..b + 1, 0..dims[1], 0..dims[2], 0..dims[3]];
        split.update(
            out.clone().slice(range.clone()),
            tgt.clone().slice(range),
            Some(mask.clone()),
        );
    }
    assert_close_f64(
        "metrics/r2 分批累计",
        split.r2(),
        common::read_scalar(&mut npz, "metrics/r2"),
    );
}

// ================================================================ HDF5 通路

fn temp_h5(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("eser_p4_{}_{}.h5", name, std::process::id()))
}

/// 把黄金数据写成 **gzip 压缩 + 分块** 的 HDF5，再走完整的
/// `compute_output_stats`，结果必须还是黄金数据里的极值。
///
/// 覆盖的是压缩块的 hyperslab 读取——本项目真实数据集就是这么存的，读错了不会报错，
/// 只会让统计量偏掉。
#[test]
fn compute_output_stats_reads_compressed_hdf5() {
    let mut npz = open_npz();
    let raw = read(&mut npz, "data/raw");
    let dims = raw.shape().to_vec();
    let flat: Vec<f32> = raw.iter().copied().collect();

    let path = temp_h5("gzip_stats");
    {
        let file = H5File::create(&path).expect("建不了临时 HDF5");
        let ds = file
            .new_dataset::<f32>()
            .shape(&dims)
            .chunk(&[2, dims[1], dims[2], dims[3]])
            .deflate(6)
            .create("output")
            .expect("建不了压缩数据集");
        ds.write_raw(&flat).expect("写不进数据");
        file.close().expect("关不掉文件");
    }

    let stats = compute_output_stats(&path, "output", 2).expect("统计失败");
    let _ = std::fs::remove_file(&path);

    assert_eq!(stats.shape, dims);
    assert_eq!(stats.n_samples, dims[0]);
    for (i, want) in read_vec(&mut npz, "data/y_min").iter().enumerate() {
        assert_close_f64(
            &format!("gzip y_min[{}]", i),
            stats.y_min[i] as f64,
            *want as f64,
        );
    }
    for (i, want) in read_vec(&mut npz, "data/y_max").iter().enumerate() {
        assert_close_f64(
            &format!("gzip y_max[{}]", i),
            stats.y_max[i] as f64,
            *want as f64,
        );
    }
}

/// 取合成数据 fixture 的路径。**缺文件直接失败，不静默跳过**。
///
/// `dataset/` 在 .gitignore 里，新克隆的仓库确实没有这些文件——但「跳过」和
/// 「通过」在测试输出里长得一模一样，跨实现对照就这么悄悄没了。整个移植的原则
/// 是宁可报错也不静默，这里不能开例外。缺文件时报的是可直接执行的修复命令。
fn fixture(name: &str) -> PathBuf {
    let p: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "..", "dataset", name]
        .iter()
        .collect();
    assert!(
        p.exists(),
        "缺少合成数据 {:?}。先在仓库根目录跑：python -m tests.make_fake_data",
        p
    );
    p
}

/// 读 **h5py 亲手写出来的** 文件，统计量对上 Python 版 `compute_output_stats`。
///
/// 参考值由 `dataset/fake2d.hdf5`（`python -m tests.make_fake_data`，seed 固定）算出，
/// 是 rust-hdf5 与 h5py 之间唯一的跨实现对照。
#[test]
fn stats_on_h5py_file_match_python() {
    let path = fixture("fake2d.hdf5");
    let stats = compute_output_stats(&path, "output", 8).expect("统计失败");
    assert_eq!(stats.shape, vec![24, 3, 137, 137]);
    assert_eq!(stats.n_samples, 24);

    let want_min = [
        -147.86183166503906f64,
        -2.9572367668151855,
        -0.44358551502227783,
    ];
    let want_max = [
        147.86183166503906f64,
        2.9572367668151855,
        0.4434671700000763,
    ];
    let want_std = [57.14970397949219f64, 1.622288703918457, 0.23117661476135254];
    for i in 0..3 {
        assert_close_f64(
            &format!("fake2d y_min[{}]", i),
            stats.y_min[i] as f64,
            want_min[i],
        );
        assert_close_f64(
            &format!("fake2d y_max[{}]", i),
            stats.y_max[i] as f64,
            want_max[i],
        );
        assert_close_f64(
            &format!("fake2d y_std[{}]", i),
            stats.y_std[i] as f64,
            want_std[i],
        );
    }
}

/// 惰性取样：批次里第 k 个样本必须是 `indices[sel[k]]`，而不是文件里的第 k 个。
/// 取错样本不会报错，只会让训练集里混进验证集的样本。
#[test]
fn lazy_batch_picks_the_requested_samples() {
    let path = fixture("fake2d.hdf5");
    let opts = DatasetOptions {
        indices: Some(vec![7, 5, 0]),
        ..Default::default()
    };
    let ds = CfdDataset::open(&path, &opts).expect("打不开数据集");
    assert_eq!(ds.len(), 3);

    let device = Default::default();
    let (x, y) = ds.batch4::<B>(&[0, 1, 2], &device).expect("取批次失败");
    assert_eq!(x.dims(), [3, 1, 137, 137]);
    assert_eq!(y.dims(), [3, 3, 137, 137]);

    // make_fake_data 把每个样本的输入填成常数 a，正好当样本身份的指纹
    let xv: Vec<f32> = x.into_data().to_vec().unwrap();
    let stride = 137 * 137;
    let want_a = [1.391772985458374f64, 1.145894169807434, 1.0488134622573853];
    for (k, a) in want_a.iter().enumerate() {
        assert!(
            (xv[k * stride] as f64 - a).abs() < 1e-6,
            "第 {} 个样本取错了：x={} 期望 {}",
            k,
            xv[k * stride],
            a
        );
    }

    // 归一化后的逐样本和，对上 Python 的 normalize_array
    let yv: Vec<f32> = y.into_data().to_vec().unwrap();
    let ystride = 3 * stride;
    let want_sum = [30079.53674119152f64, 29739.49277739972, 29605.23282930255];
    for (k, want) in want_sum.iter().enumerate() {
        let got: f64 = yv[k * ystride..(k + 1) * ystride]
            .iter()
            .map(|v| *v as f64)
            .sum();
        assert!(
            (got - want).abs() <= 1e-3 * want.abs(),
            "第 {} 个样本归一化和 {} 与 Python 的 {} 不符",
            k,
            got,
            want
        );
    }
}

/// 训练集与验证集共用一份统计量，且样本互不重叠。
/// 两边各自算极值时 R² 不可比——这是旧版真实踩过的坑。
#[test]
fn train_val_share_stats_and_do_not_overlap() {
    let path = fixture("fake2d.hdf5");
    let (train, val) =
        load_train_val(&path, 0.25, 1, &DatasetOptions::default()).expect("划分失败");
    let val = val.expect("val_split>0 应该有验证集");
    assert_eq!(train.len() + val.len(), 24);
    assert_eq!(train.stats.y_min, val.stats.y_min);
    assert_eq!(train.stats.y_max, val.stats.y_max);

    let (want_train, want_val) = split_indices(24, 0.25, 1);
    assert!(!want_val.is_empty());
    // 用输入常数当指纹，确认两个子集真的取到了各自那批样本
    let device = Default::default();
    let fingerprint = |ds: &CfdDataset, n: usize| -> Vec<f32> {
        let sel: Vec<usize> = (0..n).collect();
        let x = ds.batch4::<B>(&sel, &device).unwrap().0;
        let v: Vec<f32> = x.into_data().to_vec().unwrap();
        (0..n).map(|k| v[k * 137 * 137]).collect()
    };
    let a_train = fingerprint(&train, want_train.len());
    let a_val = fingerprint(&val, want_val.len());
    for a in &a_val {
        assert!(
            !a_train.iter().any(|t| (t - a).abs() < 1e-9),
            "验证样本 {} 也出现在训练集里",
            a
        );
    }
}

/// 3D 加载器：4D 的 output 会被补上深度维，输入按需重采样到输出的空间尺寸。
#[test]
fn load_3d_reshapes_and_aligns() {
    let path = fixture("fake3d.hdf5");
    let device = Default::default();
    let opts = Options3d {
        indices: Some(vec![0, 3]),
        ..Default::default()
    };
    let (x, y, stats) = load_3d::<B>(&path, &opts, &device).expect("加载失败");
    assert_eq!(y.dims(), [2, 3, 2, 65, 65]);
    assert_eq!(x.dims(), [2, 1, 2, 65, 65]);
    assert_eq!(stats.n_samples, 12);

    // 归一化后必须落在 [0,1]（统计量就是这份数据自己算的，不该越界）
    let yv: Vec<f32> = y.into_data().to_vec().unwrap();
    let (lo, hi) = yv
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    assert!(
        lo >= -TOL as f32 && hi <= 1.0 + TOL as f32,
        "越界 [{}, {}]",
        lo,
        hi
    );
}

/// 批次切分覆盖全集，且不同 seed 给出不同顺序（否则 shuffle 是摆设）。
#[test]
fn batch_indices_shuffle_actually_shuffles() {
    let a = batch_indices(50, 8, Some(1), false);
    let b = batch_indices(50, 8, Some(2), false);
    assert_ne!(a, b, "换了 seed 顺序却一样，shuffle 没生效");
    let ordered = batch_indices(50, 8, None, false);
    assert_ne!(a, ordered, "shuffle 后与原序一致");
    let mut flat: Vec<usize> = a.into_iter().flatten().collect();
    flat.sort_unstable();
    assert_eq!(flat, (0..50).collect::<Vec<_>>());
}
