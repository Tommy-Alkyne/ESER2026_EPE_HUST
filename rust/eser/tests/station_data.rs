//! `eser-makestation` 的产物是否符合方案 §4 的规范。
//!
//! 物理性质由 `synth_physics.rs` 管；这里只管**契约**：形状、通道数、附带文件。
//! 契约错了跟物理错了一样致命——通道顺序对不上，散度约束会去算
//! `∂P/∂x + ∂Vx/∂y` 这种无意义的量，而且不报任何错（方案 §1 原话）。

use std::path::PathBuf;
use std::process::Command;

use rust_hdf5::H5File;

fn bin(name: &str) -> PathBuf {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let me = std::env::current_exe().expect("拿不到测试可执行文件路径");
    for dir in me.ancestors().skip(1).take(3) {
        let p = dir.join(&exe);
        if p.is_file() {
            return p;
        }
    }
    panic!("找不到 {}，先跑 cargo build -p eser --bins", exe);
}

#[test]
fn generated_files_match_the_spec() {
    let out = std::env::temp_dir().join(format!("eser_station_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);

    // 最小规模：验的是契约，不是规模。1 档通风 x 2 屏蔽门态 x 4 列车工况 = 8 个工况
    let (h, w, sources, frames) = (32usize, 64usize, 2usize, 3usize);
    let res = Command::new(bin("eser-makestation"))
        .args([
            "--out-dir", out.to_str().unwrap(),
            "--grid", "32", "64",
            "--vent-levels", "1",
            "--sources", "2",
            "--frames", "3",
            "--t-end", "60",
        ])
        .output()
        .expect("起不了 eser-makestation");
    assert!(
        res.status.success(),
        "生成失败\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&res.stdout),
        String::from_utf8_lossy(&res.stderr)
    );
    // 通风量 1 档 x 屏蔽门 2 态 x 列车工况 4 种（§4.1）
    let n_cases = 2 * 4;

    // ---- §4.1 阶段一 ----
    let f = H5File::open(out.join("stage1.hdf5")).expect("打不开 stage1.hdf5");
    assert_eq!(
        f.dataset("input").expect("stage1 缺 input").shape(),
        vec![n_cases, 3, h, w],
        "§4.1 的 input 应当是 (N,3,H,W) = (通风量, 屏蔽门, 列车工况)"
    );
    assert_eq!(
        f.dataset("output").expect("stage1 缺 output").shape(),
        vec![n_cases, 4, h, w],
        "§4.1 的 output 应当是 (N,4,H,W) = (P, Vx, Vy, T)"
    );

    // ---- §4.2 阶段二 ----
    let n2 = n_cases * sources * frames;
    let f = H5File::open(out.join("stage2.hdf5")).expect("打不开 stage2.hdf5");
    assert_eq!(
        f.dataset("input").expect("stage2 缺 input").shape(),
        vec![n2, 4, h, w],
        "§4.2 的 input 应当是 (N,4,H,W) = (Vx, Vy, 源位置图, t)"
    );
    assert_eq!(
        f.dataset("output").expect("stage2 缺 output").shape(),
        vec![n2, 1, h, w],
        "§4.2 的 output 应当是 (N,1,H,W) = C"
    );

    // ---- §4.5 附带文件 ----
    let mask = out.join("station_mask.npy");
    assert!(mask.is_file(), "§4.5 要求随数据给出 .npy 栅格掩码");
    // 训练侧就是用这条路径读掩码的，读不回来等于没给
    let spec = eser::predict::GeometrySpec::parse(Some(mask.to_str().unwrap()))
        .expect("生成的 station_mask.npy 训练侧解析不了");
    let m = spec.mask(h, w).expect("掩码重采样失败");
    let cov = eser::geometry::coverage(&m);
    assert!(
        cov > 0.05 && cov < 0.6,
        "掩码覆盖率 {:.1}% 不像一个车站平面",
        cov * 100.0
    );

    // README 必须带上「没法从数据里推断」的那几个参数（§4.5 原话）
    let readme = std::fs::read_to_string(out.join("README.txt")).expect("没写 README.txt");
    for key in ["扩散系数", "计算域", "rho", "channel-order", "不是 CFD"] {
        assert!(readme.contains(key), "README 里没提到 {:?}：\n{}", key, readme);
    }

    let _ = std::fs::remove_dir_all(&out);
}

/// 参数拼错、越界时必须报错退出，不能按默认值悄悄跑一遍。
#[test]
fn bad_arguments_are_rejected() {
    let out = std::env::temp_dir().join(format!("eser_station_bad_{}", std::process::id()));
    for (args, want) in [
        (vec!["--sources", "999"], "最多"),
        (vec!["--grid", "4", "4"], "至少 16"),
        (vec!["--t-end", "0"], "必须为正"),
        (vec!["--frames", "0"], "都必须大于 0"),
        (vec!["--nonsense"], "不认识的参数"),
        (vec!["--grid", "32"], "缺参数"),
    ] {
        let mut argv = vec!["--out-dir".to_string(), out.to_string_lossy().to_string()];
        argv.extend(args.iter().map(|s| s.to_string()));
        let res = Command::new(bin("eser-makestation"))
            .args(&argv)
            .output()
            .expect("起不了 eser-makestation");
        assert!(!res.status.success(), "{:?} 居然被接受了", args);
        let err = String::from_utf8_lossy(&res.stderr);
        assert!(
            err.contains(want),
            "{:?} 的报错没说清问题（期望包含 {:?}）：{}",
            args, want, err
        );
    }
    let _ = std::fs::remove_dir_all(&out);
}
