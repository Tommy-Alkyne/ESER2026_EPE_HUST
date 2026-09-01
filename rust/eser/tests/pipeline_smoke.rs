//! 端到端冒烟：训练 → 落盘 → 诊断读得出来。
//!
//! **单测各自绿着也会漏掉这一类问题**：训练器把指标写在 `<run>/`，而诊断从
//! `<run>/training/` 读——两边的单测都是绿的，连起来才发现文件明明生成了、
//! 诊断却报「没有找到 r2_test.txt，训练可能未正常结束」。
//!
//! 这条测试直接跑构建出来的二进制，验的就是「用户实际会走的那条路」。

use std::path::{Path, PathBuf};
use std::process::Command;

use rust_hdf5::H5File;

/// 找到与测试可执行文件同目录的二进制。
fn bin(name: &str) -> PathBuf {
    let exe_name = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let me = std::env::current_exe().expect("拿不到测试可执行文件路径");
    // 测试二进制在 target/<profile>/deps/，产物在它的上一级
    for dir in me.ancestors().skip(1).take(3) {
        let p = dir.join(&exe_name);
        if p.is_file() {
            return p;
        }
    }
    panic!("找不到 {}，先跑 cargo build -p eser --bins", exe_name);
}

/// 就地造一个**极小**数据集：8 个样本、1 进 3 出、16x16。
///
/// 不用 `dataset/` 里那份 137x137 的合成数据：debug 构建下跑两轮要 5 分半，
/// 会把日常 `cargo test` 拖垮。这条测试验的是**接线**（落盘路径、进度输出、
/// checkpoint 内容），跟网格多大无关。
fn tiny_dataset(path: &Path) {
    let (n, ci, co, h, w) = (8usize, 1usize, 3usize, 16usize, 16usize);
    let x: Vec<f32> = (0..n * ci * h * w)
        .map(|i| ((i % 23) as f32 / 23.0) - 0.5)
        .collect();
    // 通道量级刻意拉开，和 fake2d 一致：顺带验「逐通道归一化」没把它们压成一样
    let y: Vec<f32> = (0..n * co * h * w)
        .map(|i| {
            let c = (i / (h * w)) % co;
            let v = ((i % 31) as f32 / 31.0) - 0.5;
            v * [100.0f32, 2.0, 0.3][c]
        })
        .collect();
    let f = H5File::create(path).expect("建不了临时 HDF5");
    for (name, dims, data) in [
        ("input", vec![n, ci, h, w], &x),
        ("output", vec![n, co, h, w], &y),
    ] {
        f.new_dataset::<f32>()
            .shape(&dims)
            .create(name)
            .unwrap()
            .write_raw(data)
            .unwrap();
    }
}

#[test]
fn train_then_diagnose_reads_what_training_wrote() {
    let exp_dir = std::env::temp_dir().join(format!("eser_pipeline_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&exp_dir);
    std::fs::create_dir_all(&exp_dir).unwrap();
    let dataset = exp_dir.join("tiny.hdf5");
    tiny_dataset(&dataset);

    // ---- 训练 ----
    let out = Command::new(bin("eser-train"))
        .args([
            "--train-file",
            dataset.to_str().unwrap(),
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--blocks",
            "2,2,2",
            "--growth-rate",
            "4",
            "--init-features",
            "8",
            "--exp-dir",
            exp_dir.to_str().unwrap(),
            "--exp-name",
            "smoke",
        ])
        .output()
        .expect("起不了 eser-train");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "训练失败（退出码 {:?}）\n--- stdout ---\n{}\n--- stderr ---\n{}",
        out.status.code(),
        stdout,
        String::from_utf8_lossy(&out.stderr)
    );
    // 每轮都要有一行进度：GUI 靠 stdout 实时滚日志
    let epochs = stdout.lines().filter(|l| l.starts_with("epoch ")).count();
    assert_eq!(epochs, 2, "应当每轮打一行进度，实得 {} 行\n{}", epochs, stdout);

    // ---- 诊断必须读得到训练写的东西 ----
    let out = Command::new(bin("eser-diagnose"))
        .arg(exp_dir.to_str().unwrap())
        .output()
        .expect("起不了 eser-diagnose");
    let report = String::from_utf8_lossy(&out.stdout).to_string();

    assert!(
        !report.contains("没有找到 r2_test.txt"),
        "**诊断读不到训练写的指标**——训练器与诊断的落盘路径对不上。\n{}",
        report
    );
    for key in ["R²", "RMSE"] {
        assert!(
            report.contains(key),
            "诊断报告里没有 {}，说明指标没被读进去\n{}",
            key,
            report
        );
    }

    // ---- checkpoint 要能被推理侧消费 ----
    let ck = exp_dir.join("smoke").join("checkpoints");
    let meta = std::fs::read_to_string(ck.join("best.meta")).expect("没写出 best checkpoint");
    let meta = eser::train::CheckpointMeta::from_text(&meta).expect("元信息解析失败");
    for key in ["nic", "noc", "ndim", "channel_order"] {
        assert!(
            meta.hp(key).is_some(),
            "checkpoint 缺超参 {}——推理侧没法重建同构的网络",
            key
        );
    }
    assert!(
        meta.stats().is_some(),
        "checkpoint 里的归一化统计量不完整，推理侧没法还原物理量纲"
    );

    // ---- --post：只评估不训练 ----
    let before = std::fs::read_to_string(exp_dir.join("smoke/training/r2_test.txt")).unwrap();
    let out = Command::new(bin("eser-train"))
        .args([
            "--train-file",
            dataset.to_str().unwrap(),
            "--post",
            "--exp-dir",
            exp_dir.to_str().unwrap(),
            "--exp-name",
            "smoke",
            "--blocks",
            "2,2,2",
            "--growth-rate",
            "4",
            "--init-features",
            "8",
        ])
        .output()
        .expect("起不了 eser-train --post");
    let log = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "--post 失败\n--- stdout ---\n{}\n--- stderr ---\n{}",
        log,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(log.contains("[post]"), "--post 没打指标\n{}", log);
    assert!(
        !log.starts_with("epoch ") && !log.contains("\nepoch "),
        "--post 不该训练，却打了逐轮进度\n{}",
        log
    );
    // 评估结果要落盘，而且**只有一行**——它评的是单个 checkpoint，不是一条曲线
    let after = std::fs::read_to_string(exp_dir.join("smoke/training/r2_test.txt")).unwrap();
    assert_eq!(
        after.lines().count(),
        1,
        "--post 应当写单行 r2_test，实得 {} 行",
        after.lines().count()
    );
    assert_ne!(before, after, "--post 没覆盖 r2_test.txt");
    // 训练侧曲线不该被伪造出来：--post 没跑训练，写一列 0 会被诊断当成「训练完 R²=0」
    let train_r2 = std::fs::read_to_string(exp_dir.join("smoke/training/r2_train.txt")).unwrap();
    assert_eq!(
        train_r2.lines().count(),
        2,
        "--post 覆盖了训练侧曲线，那些数这次根本没算"
    );

    // 找不到 checkpoint 时必须报错，不能拿随机权重评出一个像样的 R²
    let out = Command::new(bin("eser-train"))
        .args([
            "--train-file",
            dataset.to_str().unwrap(),
            "--post",
            "--ckpt-epoch",
            "999",
            "--exp-dir",
            exp_dir.to_str().unwrap(),
            "--exp-name",
            "smoke",
        ])
        .output()
        .expect("起不了 eser-train");
    assert!(
        !out.status.success(),
        "checkpoint 不存在时 --post 居然成功了——那评的是随机初始化的权重"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("找不到 checkpoint"),
        "报错信息没说清是 checkpoint 找不到：{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // ---- 推理命令行入口 ----
    let pred_file = exp_dir.join("pred.hdf5");
    let out = Command::new(bin("eser-predict"))
        .args([
            "--ckpt-file",
            ck.join("best").to_str().unwrap(),
            "--data-file",
            dataset.to_str().unwrap(),
            "--out-file",
            pred_file.to_str().unwrap(),
            "--batch-size",
            "4",
        ])
        .output()
        .expect("起不了 eser-predict");
    assert!(
        out.status.success(),
        "推理失败\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let pred = H5File::open(&pred_file).expect("读不回预测文件");
    let ds = pred.dataset("predictions").expect("预测文件里没有 predictions");
    assert_eq!(
        ds.shape(),
        vec![8, 3, 16, 16],
        "预测场形状与输入数据对不上"
    );

    // 前缀写错（带上了 .mpk）时要给出可操作的提示，而不是一句 IO 错误
    let out = Command::new(bin("eser-predict"))
        .args([
            "--ckpt-file",
            ck.join("best.mpk").to_str().unwrap(),
            "--data-file",
            dataset.to_str().unwrap(),
        ])
        .output()
        .expect("起不了 eser-predict");
    assert!(!out.status.success(), "把 .mpk 当前缀传居然成功了");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("前缀"),
        "没提示用户 --ckpt-file 要给前缀：{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let _ = std::fs::remove_dir_all(&exp_dir);
}

/// 造一份带边界分区标签的栅格掩码：外圈墙，左右墙中段各挖一个风口。
///
/// 标签口径见 `eser::geometry::Zone`：0 域外 / 1 流体 / 2 壁 / 3 入口 / 4 出口。
fn zoned_mask(path: &Path, h: usize, w: usize) {
    let mut z = vec![0.0f32; h * w];
    let put = |z: &mut Vec<f32>, y: usize, x: usize, v: f32| z[y * w + x] = v;
    for y in 2..h - 2 {
        for x in 2..w - 2 {
            put(&mut z, y, x, 1.0);
        }
    }
    for x in 1..w - 1 {
        put(&mut z, 1, x, 2.0);
        put(&mut z, h - 2, x, 2.0);
    }
    for y in 1..h - 1 {
        put(&mut z, y, 1, 2.0);
        put(&mut z, y, w - 2, 2.0);
    }
    // 左墙中段 = 送风口，右墙中段 = 排风口
    for y in h / 2 - 1..h / 2 + 1 {
        put(&mut z, y, 1, 3.0);
        put(&mut z, y, w - 2, 4.0);
    }
    let arr = ndarray::Array2::from_shape_vec((h, w), z).unwrap();
    ndarray_npy::write_npy(path, &arr).expect("写不了 .npy 掩码");
}

/// 分区掩码要真的走到损失里：**画了风口不给风速必须报错**，给了就得出现 bc 项。
///
/// 这条盯的是「静默失效」——边界标签解析对了、`Boundary` 也建出来了，但只要
/// `PhysicsLoss` 那头没接上，训练照样跑完、日志照样好看，而边界条件一条都没加。
#[test]
fn zoned_mask_actually_reaches_the_loss() {
    let exp_dir = std::env::temp_dir().join(format!("eser_bc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&exp_dir);
    std::fs::create_dir_all(&exp_dir).unwrap();
    let dataset = exp_dir.join("tiny.hdf5");
    tiny_dataset(&dataset);
    let mask = exp_dir.join("zoned.npy");
    zoned_mask(&mask, 16, 16);

    let run = |extra: &[&str]| {
        let mut args = vec![
            "--train-file",
            dataset.to_str().unwrap(),
            "--task",
            "flow",
            "--channel-order",
            "pvv",
            "--domain-mask",
            mask.to_str().unwrap(),
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--blocks",
            "2,2,2",
            "--growth-rate",
            "4",
            "--init-features",
            "8",
            "--exp-dir",
            exp_dir.to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        let out = Command::new(bin("eser-train"))
            .args(&args)
            .output()
            .expect("起不了 eser-train");
        // 合并 stdout / stderr：报错走的是 stderr，只看 stdout 会漏掉它
        (
            out.status.success(),
            format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ),
        )
    };

    // ---- 画了风口却不给风速：必须报错停下 ----
    let (ok, log) = run(&["--exp-name", "missing"]);
    assert!(
        !ok,
        "掩码里有风口却没给 --inlet-speed，必须报错而不是默默当成 0（等于把风口当墙用）\n{}",
        log
    );
    assert!(
        log.contains("--inlet-speed"),
        "报错信息里要指名缺的是哪个参数\n{}",
        log
    );

    // ---- 给齐了：边界项必须真的出现在损失里 ----
    let (ok, log) = run(&[
        "--exp-name",
        "bc",
        "--inlet-speed",
        "1.5",
        "--outlet-speed",
        "1.5",
    ]);
    assert!(ok, "给齐风速后训练应当成功\n{}", log);
    assert!(
        log.contains("分区格点数：") && log.contains("贴边流体格点："),
        "启动自检要报出各分区与贴边格点数，否则掩码画错了没人看得出来\n{}",
        log
    );
    let with_bc = log
        .lines()
        .filter(|l| l.starts_with("epoch ") && l.contains("bc="))
        .count();
    assert_eq!(with_bc, 2, "每轮进度都要带 bc 分项\n{}", log);

    // 落盘的曲线也要有，诊断侧只认这些文件
    let train_dir = exp_dir.join("bc").join("training");
    for f in ["loss_bc.txt", "bc_rms.txt", "bc_rms.svg"] {
        assert!(train_dir.join(f).is_file(), "缺产物 {}", f);
    }
    let lines = std::fs::read_to_string(train_dir.join("bc_rms.txt"))
        .unwrap()
        .lines()
        .count();
    assert_eq!(lines, 2, "bc_rms.txt 应当每轮一行");

    // ---- 对照：同一份数据不给分区掩码，就不该有 bc ----
    let (ok, log) = run(&["--exp-name", "plain", "--no-mask"]);
    assert!(ok, "无掩码跑法应当仍然成功\n{}", log);
    assert!(
        !log.contains("bc="),
        "没有边界标签时不该冒出 bc 分项——那说明它在拿默认值瞎算\n{}",
        log
    );

    let _ = std::fs::remove_dir_all(&exp_dir);
}
