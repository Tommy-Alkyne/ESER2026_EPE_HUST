//! P6 验收：推理与诊断逐元素 / 逐行对上 Python 版。
//!
//! 参照数据**不进版本库**——它依赖 `dataset/` 里的合成数据（本身也不在库里），
//! 而且体量比黄金数据大一个量级。本文件内嵌了生成脚本，缺文件时自动用仓库的
//! `.venv` 跑一遍，跑不了就报出可直接执行的命令，**不静默跳过**。
//!
//! 对照的判据与黄金数据一致：`|a-b| <= tol + tol*|b|`，tol = 1e-5。参考量级跨好几
//! 个数量级（归一化输出 O(1)、反归一化后的压力场 O(100)），绝对容差要么放过真错，
//! 要么把浮点噪声报成错。
//!
//! # 覆盖了什么
//!
//! | 组 | 覆盖 |
//! |---|---|
//! | `p2d` | 2D 全流程：分批前向 + 拼接顺序 + `limit` 截断 + 反归一化 + 默认 T 型掩码 |
//! | `pstation` | `.npy` 栅格几何（车站平面）+ 4 通道 + 非整除批次 |
//! | `p3d` | 3D 路径：5D 场、掩码在深度轴上的广播 |
//! | `diag_exp_*` | 诊断报告逐行文本 + 退出码 |

mod common;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use burn::tensor::{Tensor, TensorData};
use common::{assert_close_slice, read, state_dict, B};
use eser::data::{CfdDataset, DatasetOptions, Options3d};
use eser::diagnose;
use eser::model::DenseEd;
use eser::model3d::DenseEd3d;
use eser::predict::json::Json;
use eser::predict::{
    default_out_file, infer_channels, merge_ckpt_args, predict_2d, predict_3d,
    write_prediction_hdf5, GeometrySpec, Hyperparams, MaskChoice, MaskSource, OutputMeta,
    PredictOptions,
};
use eser::store::LoadState;
use ndarray_npy::NpzReader;

/// 冻结在版本库里的 Python 参照数据。
///
/// **这批数据当初由 Python 版跑出来，现在 Python 已经退役**，所以它是冻结的：
/// 既不能再生成，也不需要再生成——它记录的是「Rust 侧当年逐元素对上了 PyTorch」
/// 这个事实。改了 Rust 侧的实现，这些断言仍然是有效的回归防线。
///
/// 早先这里是「测试运行时 shell 出去跑一段内嵌 Python 生成参照」，失效判据一路从
/// 手写版本号升级到内容指纹；Python 退役后那条路整个不成立了，改为直接读版本库。
fn fixture() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir: PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            "..",
            "..",
            "tests",
            "golden",
            "predict_oracle",
        ]
        .iter()
        .collect();
        assert!(
            dir.join("oracle.npz").exists(),
            "缺少冻结的参照数据 {:?}。它应当在版本库里，检查是不是被误删了",
            dir.join("oracle.npz")
        );
        dir
    })
    .as_path()
}

fn npz() -> NpzReader<File> {
    let path = fixture().join("oracle.npz");
    NpzReader::new(File::open(&path).unwrap_or_else(|e| panic!("打不开 {:?}: {}", path, e)))
        .expect("oracle.npz 不是合法的 npz")
}

fn scalar(name: &str) -> f64 {
    let mut z = npz();
    let a = read(&mut z, name);
    assert_eq!(a.len(), 1, "{} 不是标量", name);
    a.iter().copied().next().unwrap() as f64
}

fn vec1(name: &str) -> Vec<f32> {
    read(&mut npz(), name).iter().copied().collect()
}

/// 仓库根目录（本文件在 ，往上两级）。
fn repo_root() -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "..", ".."].iter().collect()
}

fn dataset(name: &str) -> PathBuf {
    repo_root().join("dataset").join(name)
}

fn station_npy() -> String {
    dataset("station_mask.npy").to_string_lossy().to_string()
}

// ================================================================ 2D 推理

/// 造一份与 Python 侧同构的超参，**走 checkpoint 合并这条路**——这样白名单漏项
/// 会直接体现在预测结果上，而不只是一条单元测试。
fn hp_from_ckpt(channel_order: &str, geometry: Option<&str>) -> Hyperparams {
    let geom = match geometry {
        Some(g) => format!("{:?}", g),
        None => "null".to_string(),
    };
    let args = Json::parse(&format!(
        r#"{{"blocks": [1, 1, 1], "growth_rate": 4, "init_features": 8,
             "channel_order": {:?}, "t_geometry": {}, "no_mask": false,
             "epochs": 3, "lr": 0.001}}"#,
        channel_order, geom
    ))
    .expect("测试用的 checkpoint args 不是合法 JSON");
    let mut hp = Hyperparams::default();
    let dropped = merge_ckpt_args(&mut hp, &args).expect("合并超参失败");
    assert_eq!(
        dropped,
        vec!["epochs".to_string(), "lr".to_string()],
        "白名单外的键要被报出来"
    );
    hp
}

fn open_2d(file: &str) -> CfdDataset {
    let path = dataset(file);
    CfdDataset::open(&path, &DatasetOptions::default())
        .unwrap_or_else(|e| panic!("打不开 {:?}: {}", path, e))
}

fn build_2d(tag: &str, hp: &Hyperparams, ds: &CfdDataset) -> DenseEd<B> {
    let mut z = npz();
    let sd = state_dict(&mut z, tag);
    let (nic, noc) = infer_channels::<B>(&sd, 2, None, None).expect("推不出通道数");
    assert_eq!(noc, ds.y_shape()[1], "{}: 输出通道数与数据不符", tag);
    let output_size = [ds.y_shape()[2], ds.y_shape()[3]];
    let mut model = hp
        .model_config_2d(nic, noc, output_size)
        .init::<B>(&Default::default())
        .expect("建模失败");
    model
        .load_state(&sd, "")
        .unwrap_or_else(|e| panic!("{} 加载权重失败：{}", tag, e));
    // 有键没被取走 = 有一层还留着随机初始化，模型照样前向、照样出数
    sd.ensure_fully_used()
        .unwrap_or_else(|e| panic!("{}：{}", tag, e));
    model
}

fn opts_2d(batch: usize, limit: usize) -> PredictOptions {
    PredictOptions {
        batch_size: batch,
        limit: Some(limit),
        ..Default::default()
    }
}

/// 全流程：分批前向 -> 反归一化 -> 默认 T 型几何裁剪，逐元素对上 Python。
#[test]
fn predict_2d_matches_python_end_to_end() {
    let hp = hp_from_ckpt("pvv", None);
    let ds = open_2d("fake2d.hdf5");
    let model = build_2d("p2d", &hp, &ds);
    let pred = predict_2d(&model, &ds, &hp, &opts_2d(5, 6), &Default::default()).expect("推理失败");

    let want = read(&mut npz(), "p2d/pred_masked");
    assert_eq!(&pred.dims[..], want.shape(), "预测场形状不符");
    assert_close_slice("p2d/pred_masked", &pred.data, &want);

    assert!(matches!(
        pred.mask,
        MaskChoice::Applied {
            source: MaskSource::Checkpoint,
            ..
        }
    ));
    let cov = pred.coverage.expect("用了掩码就该有覆盖率");
    assert!(
        (cov - scalar("p2d/coverage")).abs() < 1e-6,
        "覆盖率 {} != Python 的 {}",
        cov,
        scalar("p2d/coverage")
    );
    assert_eq!(
        pred.y_min,
        vec1("p2d/y_min"),
        "y_min 与 Python 的统计量不符"
    );
    assert_eq!(
        pred.y_max,
        vec1("p2d/y_max"),
        "y_max 与 Python 的统计量不符"
    );
}

/// 逐阶段拆开对：归一化输出 -> 反归一化 -> 掩码。
///
/// 只比最终结果的话，「忘了反归一化」和「掩码没生效」会互相掩盖；分阶段对照才能
/// 定位到是哪一步错了。
#[test]
fn predict_2d_stages_match_python() {
    let hp = hp_from_ckpt("pvv", None);
    let ds = open_2d("fake2d.hdf5");
    let model = build_2d("p2d", &hp, &ds);

    let raw = predict_2d(
        &model,
        &ds,
        &hp,
        &PredictOptions {
            denormalize: false,
            apply_mask: false,
            ..opts_2d(5, 6)
        },
        &Default::default(),
    )
    .expect("推理失败");
    assert_close_slice(
        "p2d/pred_norm",
        &raw.data,
        &read(&mut npz(), "p2d/pred_norm"),
    );
    assert!(!raw.denormalized);
    assert_eq!(raw.mask, MaskChoice::Disabled);

    let dn = predict_2d(
        &model,
        &ds,
        &hp,
        &PredictOptions {
            apply_mask: false,
            ..opts_2d(5, 6)
        },
        &Default::default(),
    )
    .expect("推理失败");
    assert_close_slice(
        "p2d/pred_denorm",
        &dn.data,
        &read(&mut npz(), "p2d/pred_denorm"),
    );
}

/// 批次大小不影响结果：拼接顺序错了、或最后一个不满批被丢掉，这里就会红。
#[test]
fn batch_size_does_not_change_the_prediction() {
    let hp = hp_from_ckpt("pvv", None);
    let ds = open_2d("fake2d.hdf5");
    let model = build_2d("p2d", &hp, &ds);
    let a = predict_2d(&model, &ds, &hp, &opts_2d(5, 6), &Default::default()).unwrap();
    let b = predict_2d(&model, &ds, &hp, &opts_2d(3, 6), &Default::default()).unwrap();
    let c = predict_2d(&model, &ds, &hp, &opts_2d(64, 6), &Default::default()).unwrap();
    assert_eq!(a.dims, b.dims);
    assert_eq!(a.data, b.data, "batch=5 与 batch=3 的结果不一致");
    assert_eq!(a.data, c.data, "batch=5 与整批的结果不一致");
}

// ================================================================ 栅格几何

/// 车站平面（`.npy` 栅格）这条路径。**几何从 checkpoint 里读回来**——
/// `t_geometry` 一旦掉出白名单，这里就会退回默认 T 型几何，逐元素对照立刻红。
#[test]
fn predict_with_station_raster_matches_python() {
    let hp = hp_from_ckpt("pvvt", Some(&station_npy()));
    let ds = open_2d("fake_stage1.hdf5");
    let model = build_2d("pstation", &hp, &ds);
    let pred = predict_2d(&model, &ds, &hp, &opts_2d(4, 7), &Default::default()).expect("推理失败");

    let want = read(&mut npz(), "pstation/pred_masked");
    assert_eq!(&pred.dims[..], want.shape());
    assert_close_slice("pstation/pred_masked", &pred.data, &want);
    assert_eq!(pred.dims[1], 4, "stage1 是 4 通道 (P, Vx, Vy, T)");
}

/// 掩码本身逐格对上，且**两份几何真的不一样**——参照数据得有鉴别力，
/// 否则「用错几何」根本验不出来。
#[test]
fn station_mask_differs_from_the_default_t_pipe() {
    let station = GeometrySpec::parse(Some(&station_npy())).expect("读不了栅格掩码");
    let got = station.mask(96, 96).expect("生成掩码失败");
    let want = read(&mut npz(), "pstation/mask");
    let want_v: Vec<f32> = want.iter().copied().collect();
    assert_eq!(got, want_v, "栅格掩码与 Python 的 t_mask_numpy 不符");

    let t_pipe = GeometrySpec::Default
        .mask(96, 96)
        .expect("生成 T 型掩码失败");
    let cov_station = eser::geometry::coverage(&got);
    let cov_t = eser::geometry::coverage(&t_pipe);
    assert!(
        (cov_station - cov_t).abs() > 0.05,
        "车站掩码覆盖率 {:.3} 与 T 型 {:.3} 太接近，这组参照数据验不出用错几何",
        cov_station,
        cov_t
    );
}

// ================================================================ 3D 推理

#[test]
fn predict_3d_matches_python() {
    let hp = hp_from_ckpt("pvv", None);
    let path = dataset("fake3d.hdf5");
    let (x, _, stats) = eser::data::load_3d::<B>(&path, &Options3d::default(), &Default::default())
        .expect("3D 加载失败");

    let mut z = npz();
    let sd = state_dict(&mut z, "p3d");
    let (nic, noc) = infer_channels::<B>(&sd, 3, None, None).expect("推不出通道数");
    assert_eq!((nic, noc), (1, 3));
    let mut model: DenseEd3d<B> = hp
        .model_config_3d(nic, noc, [2, 65, 65])
        .init(&Default::default())
        .expect("建模失败");
    model.load_state(&sd, "").expect("加载 3D 权重失败");
    sd.ensure_fully_used().expect("有 3D 权重没被加载");

    let pred = predict_3d(
        &model,
        x,
        &stats,
        &hp,
        &PredictOptions {
            batch_size: 2,
            limit: Some(3),
            ..Default::default()
        },
    )
    .expect("推理失败");

    let want = read(&mut npz(), "p3d/pred_masked");
    assert_eq!(&pred.dims[..], want.shape(), "5D 预测场形状不符");
    assert_close_slice("p3d/pred_masked", &pred.data, &want);
}

// ================================================================ 写盘

#[test]
fn prediction_file_round_trips() {
    use rust_hdf5::H5File;

    let hp = hp_from_ckpt("pvv", None);
    let ds = open_2d("fake2d.hdf5");
    let model = build_2d("p2d", &hp, &ds);
    let pred = predict_2d(&model, &ds, &hp, &opts_2d(5, 6), &Default::default()).unwrap();

    let out = std::env::temp_dir().join(format!("eser_p6_out_{}.pred.hdf5", std::process::id()));
    let meta = OutputMeta::new(&hp, "fake2d.hdf5", "model_best.mpk", 2, 1, 3);
    write_prediction_hdf5(&out, &pred, &meta).expect("写不出预测文件");

    let file = H5File::open(&out).expect("读不回预测文件");
    let dsx = file
        .dataset("predictions")
        .expect("没有 predictions 数据集");
    assert_eq!(dsx.shape(), pred.dims);
    let back: Vec<f32> = dsx.read_raw().expect("读不出 predictions");
    assert_eq!(back, pred.data, "gzip 往返后数值不一致");

    let y_min: Vec<f32> = file.dataset("y_min").unwrap().read_raw().unwrap();
    assert_eq!(y_min, pred.y_min);

    let names = file.attr_names().expect("读不出属性名");
    for want in [
        "source_data",
        "ckpt_file",
        "ndim",
        "nic",
        "noc",
        "denormalized",
        "masked",
        "channel_order",
        "hyperparams",
    ] {
        assert!(
            names.iter().any(|n| n == want),
            "缺属性 {}（现有 {:?}）",
            want,
            names
        );
    }
    assert_eq!(file.attr_string("channel_order").unwrap(), "pvv");
    // 属性里的超参要能原样解析回来，下游不必再猜
    let hyper = file.attr_string("hyperparams").unwrap();
    let parsed = Json::parse(&hyper).unwrap_or_else(|e| panic!("{} -> {}", hyper, e));
    let mut restored = Hyperparams::default();
    merge_ckpt_args(&mut restored, &parsed).unwrap();
    assert_eq!(restored, hp);

    drop(file);
    let _ = std::fs::remove_file(&out);
    assert_eq!(
        default_out_file(&dataset("fake2d.hdf5")),
        repo_root().join("dataset").join("fake2d.pred.hdf5")
    );
}

// ================================================================ 诊断

/// 与 Python 的 `diagnose.py` **逐行**对照。
fn compare_diagnose(name: &str) {
    let dir = fixture();
    // 见 normalize_run_path：参照文本里带着生成当时的绝对路径
    let path = dir.join(format!("diag_{}.txt", name));
    let want =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("读不了 {:?}: {}", path, e));
    let (code, body) = want.split_once('\n').expect("参照文件第一行是退出码");

    let report = diagnose::run(&dir.join(name));
    assert_eq!(
        report.exit_code,
        code.trim().parse::<i32>().expect("退出码不是整数"),
        "{} 的退出码不符",
        name
    );

    // 两边各自把「自己那份文本里的实验根目录」替换成占位符，见 strip_run_root
    let got_text = strip_run_root(&report.text());
    let want_text = strip_run_root(body);
    let got_lines: Vec<&str> = got_text.split('\n').collect();
    let want_lines: Vec<&str> = want_text.split('\n').collect();
    for (i, (g, w)) in got_lines.iter().zip(want_lines.iter()).enumerate() {
        assert_eq!(
            g,
            w,
            "{} 第 {} 行不符\n  Rust  : {:?}\n  Python: {:?}",
            name,
            i + 1,
            g,
            w
        );
    }
    assert_eq!(
        got_lines.len(),
        want_lines.len(),
        "{} 行数不符：Rust {} 行，Python {} 行",
        name,
        got_lines.len(),
        want_lines.len()
    );
}

/// 多次实验里挑最新的那次、dispersion 任务、分通道 R²、过拟合回落告警。
#[test]
fn diagnose_matches_python_dispersion_run() {
    compare_diagnose("exp_a");
}

/// 配置硬伤（同文件 + val_split=0、缺 channel_order）、单列分通道序列、物理残差块。
#[test]
fn diagnose_matches_python_flow_run_with_problems() {
    compare_diagnose("exp_b");
}

/// 只有 args.txt、没有任何曲线数值的实验。
#[test]
fn diagnose_matches_python_bare_run() {
    compare_diagnose("exp_c");
}

/// 硬伤要作为结构化结果暴露出来，而不只是日志里的一行字。
#[test]
fn diagnose_reports_problems_structurally() {
    let report = diagnose::run(&fixture().join("exp_b"));
    assert_eq!(report.problems.len(), 2, "{:?}", report.problems);
    assert!(report.problems[0].contains("训练集与测试集是同一个文件"));
    assert!(report.problems[1].contains("没有 channel_order"));

    let clean = diagnose::run(&fixture().join("exp_a"));
    assert!(clean.problems.is_empty(), "{:?}", clean.problems);
}

// ================================================================ 权重表

/// 通道数从权重形状反推，且不会把权重表标成「用过了」而让漏加载蒙混过关。
#[test]
fn infer_channels_reads_the_weight_shapes() {
    let mut z = npz();
    let sd = state_dict(&mut z, "pstation");
    assert_eq!(infer_channels::<B>(&sd, 2, None, None).unwrap(), (3, 4));
    // checkpoint 里显式给了就以它为准
    assert_eq!(
        infer_channels::<B>(&sd, 2, Some(9), Some(7)).unwrap(),
        (9, 7)
    );

    let mut empty = eser::store::StateDict::new();
    empty.insert("in_conv.weight", vec![8, 3, 7, 7], vec![0.0; 8 * 3 * 49]);
    assert!(
        infer_channels::<B>(&empty, 2, None, None).is_err(),
        "缺输出层权重时必须报错，而不是猜一个通道数"
    );
}

/// 掩码是「置零」不是「相乘」：预测场出了 NaN 时两者不是一回事。
#[test]
fn mask_zeroes_instead_of_multiplying() {
    let hp = hp_from_ckpt("pvv", None);
    let ds = open_2d("fake2d.hdf5");
    let model = build_2d("p2d", &hp, &ds);
    let pred = predict_2d(&model, &ds, &hp, &opts_2d(5, 6), &Default::default()).unwrap();

    let mask = GeometrySpec::Default.mask(137, 137).unwrap();
    let plane = 137 * 137;
    let outside_nonzero = pred
        .data
        .chunks(plane)
        .flat_map(|c| c.iter().zip(mask.iter()))
        .filter(|(v, m)| **m == 0.0 && **v != 0.0)
        .count();
    assert_eq!(
        outside_nonzero, 0,
        "计算域外还有 {} 个非零值",
        outside_nonzero
    );

    // 域内不能被顺手抹掉：掩码写反的话这里会变成 0
    let inside_nonzero = pred
        .data
        .chunks(plane)
        .flat_map(|c| c.iter().zip(mask.iter()))
        .filter(|(v, m)| **m == 1.0 && **v != 0.0)
        .count();
    assert!(inside_nonzero > plane / 2, "域内几乎全是 0，掩码可能写反了");
}

/// 张量层的健全性：`Tensor::cat` 的顺序就是喂进去的顺序。
/// （拼接顺序错了 `batch_size_does_not_change_the_prediction` 未必抓得到——
/// 两个批次都反了的话结果仍然自洽。）
#[test]
fn concatenation_keeps_sample_order() {
    let dev = Default::default();
    let a = Tensor::<B, 4>::from_data(TensorData::from([[[[1.0f32]]]]), &dev);
    let b = Tensor::<B, 4>::from_data(TensorData::from([[[[2.0f32]]]]), &dev);
    let c = Tensor::cat(vec![a, b], 0);
    assert_eq!(c.into_data().to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}

/// 把文本里的实验根目录换成占位符。
///
/// 参照文本是 Python 版当年跑出来的，**路径写死在里面**（临时目录），而参照数据后来
/// 被冻进了版本库——两边的绝对路径必然不同，那是环境差异不是行为差异。
/// 出现路径的行不止一处（实验目录、曲线图清单、checkpoint 清单……），逐行打补丁是
/// 打地鼠；这里改成从文本里**自己推出根目录**再整篇替换。
///
/// 根目录的来源是 `[INFO] 实验目录: <root>/<exp>/<run>` 这一行——去掉末两级就是根。
/// 推不出来（比如报告压根没走到那一步）就原样返回，让断言照常报差异。
fn strip_run_root(text: &str) -> String {
    const KEY: &str = "[INFO] 实验目录: ";
    let Some(line) = text.lines().find(|l| l.starts_with(KEY)) else {
        return text.to_string();
    };
    let full = &line[KEY.len()..];
    // 去掉末两级（<exp>/<run>），剩下的就是根
    let mut root = full.to_string();
    for _ in 0..2 {
        match root.rfind(['/', '\\']) {
            Some(i) => root.truncate(i),
            None => return text.to_string(),
        }
    }
    if root.is_empty() {
        return text.to_string();
    }
    // 路径分隔符两种写法都要换：报告里 join 出来的既有 `\` 也有 `/`
    let alt = if root.contains('\\') {
        root.replace('\\', "/")
    } else {
        root.replace('/', "\\")
    };
    text.replace(&root, "<RUN_ROOT>")
        .replace(&alt, "<RUN_ROOT>")
}
