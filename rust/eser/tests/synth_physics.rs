//! 合成数据生成器的物理验收。
//!
//! 这批测试的作用不是「代码没崩」，而是**证明产出的数据真的满足损失里量的那两个方程**。
//! 三角函数合成场同样不崩，但物理残差在它上面是噪声——那种数据训出来的模型没有参考
//! 价值。所以这里一律用 `eser::physics` / `eser::transport` 里**损失实际调用的那些函数**
//! 去量，不另写一份判据。

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

use eser::synth::{
    self, advance, bernoulli_pressure, central_diff, log_times, solve_flow, solve_stream,
    source_map, stable_dt, station_mask, velocity_from_stream, FlowCase, Scenario,
};

type B = NdArray<f32>;

// 验的是物理性质，与网格分辨率无关。取小网格是为了让整轮测试留在秒级——
// 96x192 那版一轮要 414 秒，没人会在改代码时反复跑
const H: usize = 48;
const W: usize = 96;

fn dev() -> <B as burn::tensor::backend::Backend>::Device {
    Default::default()
}

fn to_t(v: &[f64], h: usize, w: usize) -> Tensor<B, 4> {
    let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::<B, 1>::from_data(TensorData::from(f.as_slice()), &dev()).reshape([1, 1, h, w])
}

fn to_vec(t: Tensor<B, 4>) -> Vec<f64> {
    t.into_data()
        .into_vec::<f32>()
        .expect("张量取不出 f32")
        .into_iter()
        .map(|x| x as f64)
        .collect()
}

/// 掩码内的 RMS。域外恒为 0，混进来会把指标稀释得看不出问题。
fn masked_rms(v: &[f64], mask: &[f32]) -> f64 {
    let (mut s, mut n) = (0.0, 0usize);
    for (i, &x) in v.iter().enumerate() {
        if mask[i] > 0.5 {
            s += x * x;
            n += 1;
        }
    }
    if n == 0 {
        return f64::NAN;
    }
    (s / n as f64).sqrt()
}

/// 损失实际取平均的那个域：掩码往内腐蚀 1 格。
///
/// 用 `ops::erode_mask`——`physics.rs:377` 调的同一个函数，不自己再写一份。
/// 贴墙那一圈必须排除：生成器把壁面当无通量（域外邻居取中心值），而 `grad_xy` 是
/// replicate padding、把域外看成 0，两者在那一圈**必然**不同。损失早就把这圈剔掉了，
/// 判据也应当在同一个域上量，否则量的是「两种边界处理的差」而不是「方程满足得好不好」。
fn eroded(mask: &[f32], h: usize, w: usize) -> Vec<f32> {
    let t = Tensor::<B, 1>::from_data(TensorData::from(mask), &dev()).reshape([1, 1, h, w]);
    eser::ops::erode_mask(t, 1)
        .into_data()
        .into_vec::<f32>()
        .expect("腐蚀后的掩码取不出 f32")
}

fn masked_max_abs(v: &[f64], mask: &[f32]) -> f64 {
    v.iter()
        .enumerate()
        .filter(|(i, _)| mask[*i] > 0.5)
        .map(|(_, x)| x.abs())
        .fold(0.0f64, f64::max)
}

fn demo_case() -> FlowCase {
    FlowCase { vent: 1.0, psd_open: true, train_phase: 1 }
}

// ================================================================ 核心：div(v) = 0

/// **本模块最重要的一条**：产出的速度场在 `physics::divergence`——损失里同一个
/// 函数——下必须归零到浮点精度，而不是「小」。
///
/// 依据见 `synth.rs` 头部：`v = (∂ψ/∂y, -∂ψ/∂x)` 两次中心差分后混合偏导逐项抵消。
/// 若哪天有人把速度改成 `∇φ`（速度势），这条会立刻变红。
#[test]
fn divergence_vanishes_under_the_loss_operator() {
    let mask = station_mask(H, W).expect("造不出站体掩码");
    let sc = Scenario::default();
    let (dx, dy) = sc.spacing(H, W);
    let sol = solve_flow(&mask, H, W, &demo_case(), &sc, 4000).expect("解不出流场");

    let div = to_vec(eser::physics::divergence(
        to_t(&sol.vx, H, W),
        to_t(&sol.vy, H, W),
        dx,
        dy,
    ));

    let speed = masked_rms(
        &sol.vx.iter().zip(&sol.vy).map(|(a, b)| (a * a + b * b).sqrt()).collect::<Vec<_>>(),
        &mask,
    );
    assert!(speed > 0.05, "流场几乎是静止的（RMS |v| = {:.3e}），下面的 div 判据就没意义了", speed);

    // 无量纲化：div 的量纲是 1/s，用 |v|/dx 做尺度才谈得上「相对误差」
    let scale = speed / dx;
    let rel = masked_rms(&div, &mask) / scale;
    assert!(
        rel < 1e-6,
        "div(v) 没有抵消干净：相对 RMS = {:.3e}（|v|_rms={:.3e}, dx={:.4}）。\n\
         流函数构造下这一项应当是浮点抵消量级，出现有限值说明速度不是由 ψ 的中心差分来的",
        rel, speed, dx
    );
}

/// f64 的生成器与 f32 的 `ops::grad_xy` 必须逐格一致。
///
/// 生成器全程走 f64（抵消性质对精度敏感），但训练侧是 f32 张量。两套实现要是口径
/// 不同——比如边界补值方式不一样——上面那条测试会在 f64 上绿、真训练时却不满足。
#[test]
fn central_diff_matches_ops_grad_xy() {
    let (h, w) = (17, 23);
    // 用非对称、非周期的场，免得某些错误实现在对称场上碰巧也对
    let f: Vec<f64> = (0..h * w)
        .map(|k| {
            let (i, j) = ((k / w) as f64, (k % w) as f64);
            (0.3 * i).sin() * (0.21 * j + 0.7).cos() + 0.05 * i * j
        })
        .collect();
    let (dx, dy) = (0.37, 0.53);

    let (mine_x, mine_y) = central_diff(&f, h, w, dx, dy);
    let (ops_x, ops_y) = eser::ops::grad_xy(to_t(&f, h, w), dx, dy);
    let (ops_x, ops_y) = (to_vec(ops_x), to_vec(ops_y));

    let scale = f.iter().fold(0.0f64, |a, &b| a.max(b.abs())) / dx.min(dy);
    for (name, a, b) in [("d/dx", &mine_x, &ops_x), ("d/dy", &mine_y, &ops_y)] {
        let worst = a
            .iter()
            .zip(b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f64, f64::max);
        // f32 张量那侧只有约 7 位有效数字，容差按它定
        assert!(
            worst / scale < 1e-5,
            "{} 与 ops::grad_xy 对不上：最大绝对差 {:.3e}（尺度 {:.3e}）",
            name, worst, scale
        );
    }
}

/// 符号写反是最容易犯又最难看出来的错：`(∂ψ/∂y, +∂ψ/∂x)` 得到的是梯度场，
/// 它**无旋而非无散**，div 会变成宽模板 Laplacian，不再是 0。
///
/// 这条是对上面那条判据「有没有鉴别力」的直接检验。
#[test]
fn wrong_sign_would_break_divergence() {
    let mask = station_mask(H, W).expect("造不出站体掩码");
    let sc = Scenario::default();
    let (dx, dy) = sc.spacing(H, W);
    let om: Vec<f64> = (0..H * W)
        .map(|k| ((k / W) as f64 * 0.2).sin() * ((k % W) as f64 * 0.13).cos())
        .collect();
    let (psi, _) = solve_stream(&mask, &om, H, W, dx, dy, 3000, 1e-12).expect("解不出 ψ");

    let (vx, vy) = velocity_from_stream(&psi, H, W, dx, dy);
    let good = masked_rms(&to_vec(eser::physics::divergence(to_t(&vx, H, W), to_t(&vy, H, W), dx, dy)), &mask);

    // 故意把符号写反
    let (gx, gy) = central_diff(&psi, H, W, dx, dy);
    let bad = masked_rms(&to_vec(eser::physics::divergence(to_t(&gy, H, W), to_t(&gx, H, W), dx, dy)), &mask);

    assert!(
        bad > good * 1e3,
        "符号写反后 div 只从 {:.3e} 变到 {:.3e}，说明这条判据根本分辨不出对错",
        good, bad
    );
}

// ================================================================ 壁面不透风

/// `ψ` 在边界上取常数 ⇒ 壁面法向速度为 0。这里检查「域内紧贴壁面的一圈」速度确实很小。
#[test]
fn walls_do_not_leak() {
    let mask = station_mask(H, W).expect("造不出站体掩码");
    let sc = Scenario::default();
    let sol = solve_flow(&mask, H, W, &demo_case(), &sc, 4000).expect("解不出流场");

    let speed: Vec<f64> = sol.vx.iter().zip(&sol.vy).map(|(a, b)| (a * a + b * b).sqrt()).collect();
    let inside_max = masked_max_abs(&speed, &mask);

    // 域外一格：ψ 恒 0，速度应当也恒 0
    let outside: Vec<f64> = speed
        .iter()
        .enumerate()
        .map(|(i, &v)| if mask[i] > 0.5 { 0.0 } else { v })
        .collect();
    let leak = outside.iter().fold(0.0f64, |a, &b| a.max(b));
    assert!(
        leak < inside_max * 1e-9,
        "计算域外出现了速度：最大 {:.3e}，域内最大 {:.3e}——壁面在漏风",
        leak, inside_max
    );
}

// ================================================================ 对流扩散残差

/// 生成的浓度场必须满足 `∂C/∂t + u·∇C = D∇²C + S`，且是**在损失用的那个残差函数下**。
///
/// 做法：取一个时刻前后各推进一个小步长，用 `transport::central_time_derivative` 算
/// `∂C/∂t`，再喂给 `transport::advection_diffusion_residual`。这条走的正是训练时
/// 那条代码路径（§3.2 的「网络两次前向做中心差分」，只是这里前向换成了求解器）。
#[test]
fn concentration_satisfies_advection_diffusion() {
    let mask = station_mask(H, W).expect("造不出站体掩码");
    let sc = Scenario::default();
    let (dx, dy) = sc.spacing(H, W);
    let sol = solve_flow(&mask, H, W, &demo_case(), &sc, 4000).expect("解不出流场");

    let src = source_map(H, W, 0.50, 0.25, 0.04, &mask);
    let dt = stable_dt(&sol.vx, &sol.vy, dx, dy, sc.diffusivity);

    // 先推到 t=60 s，再从那儿前后各走一小步做中心差分
    let (c0, _) = advance(
        &vec![0.0; H * W], &sol.vx, &sol.vy, &mask, &src,
        H, W, dx, dy, sc.diffusivity, dt, 60.0,
    );
    let (c_next, _) = advance(
        &c0, &sol.vx, &sol.vy, &mask, &src, H, W, dx, dy, sc.diffusivity, dt, dt,
    );
    // 往回走一步：源项与对流都反号，等价于逆时间推进一个小步
    let (c_prev, _) = advance(
        &c0, &sol.vx.iter().map(|v| -v).collect::<Vec<_>>(),
        &sol.vy.iter().map(|v| -v).collect::<Vec<_>>(), &mask,
        &src.iter().map(|v| -v).collect::<Vec<_>>(),
        H, W, dx, dy, -sc.diffusivity, dt, dt,
    );

    let dcdt = eser::transport::central_time_derivative(
        to_t(&c_prev, H, W), to_t(&c_next, H, W), dt,
    )
    .expect("dt 应当为正");

    let res = eser::transport::advection_diffusion_residual(
        to_t(&c0, H, W),
        to_t(&sol.vx, H, W),
        to_t(&sol.vy, H, W),
        sc.diffusivity,
        dx,
        dy,
        Some(dcdt),
        Some(to_t(&src, H, W)),
    );
    let res = to_vec(res);

    // 残差要跟方程里最大的那一项比，而不是跟 0 比——绝对值小可能只是因为浓度本身小
    let (dc_dx, dc_dy) = central_diff(&c0, H, W, dx, dy);
    let adv: Vec<f64> = (0..H * W).map(|k| sol.vx[k] * dc_dx[k] + sol.vy[k] * dc_dy[k]).collect();
    let core = eroded(&mask, H, W);
    let n_core = core.iter().filter(|&&m| m > 0.5).count();
    assert!(
        n_core > 200,
        "腐蚀后只剩 {} 个格点，样本太少，这条判据说明不了问题",
        n_core
    );
    let scale = masked_rms(&adv, &core).max(masked_rms(&src, &core));
    assert!(scale > 0.0, "对流项和源项都是 0，这个判据不成立");

    let rel = masked_rms(&res, &core) / scale;
    assert!(
        rel < 5e-2,
        "对流扩散残差偏大：相对 RMS = {:.3e}（对流/源项尺度 {:.3e}）。\n\
         若远大于时间离散误差 O(dt)，说明生成器与残差函数的离散口径对不上",
        rel, scale
    );
}

/// 没有源、没有流出时，掩码内的总量应当守恒（扩散只搬运不产生）。
#[test]
fn diffusion_conserves_mass_without_source() {
    let mask = station_mask(H, W).expect("造不出站体掩码");
    let sc = Scenario::default();
    let (dx, dy) = sc.spacing(H, W);
    // 关掉对流，只留扩散：有对流时域外置零会真的带走物质，守恒不成立
    let zero = vec![0.0f64; H * W];
    let c0 = source_map(H, W, 0.50, 0.50, 0.06, &mask);
    let total = |c: &[f64]| -> f64 {
        c.iter().enumerate().filter(|(i, _)| mask[*i] > 0.5).map(|(_, v)| v).sum()
    };
    let before = total(&c0);
    let dt = stable_dt(&zero, &zero, dx, dy, sc.diffusivity);
    let (c1, steps) = advance(&c0, &zero, &zero, &mask, &zero, H, W, dx, dy, sc.diffusivity, dt, 20.0);
    assert!(steps > 5, "步数太少（{}），这条测试没走够路", steps);

    let after = total(&c1);
    // 贴着壁面的格点其五点模板会伸到域外（那里恒为 0），相当于壁面在吸收——
    // 所以不是严格守恒。放宽到 5%：真正要抓的是「总量爆掉或者掉光」这类错误
    let drift = (after - before).abs() / before;
    assert!(
        drift < 0.05,
        "扩散把总量改变了 {:.1}%（{:.4} -> {:.4}）",
        drift * 100.0, before, after
    );
    assert!(after > 0.0, "浓度全没了");
}

// ================================================================ 工况可分辨

/// 三个输入自由度都必须真的改变流场，否则网络学到的是「输入无关的常数映射」，
/// 训练指标照样好看。
#[test]
fn every_case_knob_changes_the_flow() {
    let mask = station_mask(H, W).expect("造不出站体掩码");
    let sc = Scenario::default();
    let base = FlowCase { vent: 1.0, psd_open: true, train_phase: 1 };
    let solve = |c: FlowCase| solve_flow(&mask, H, W, &c, &sc, 3000).expect("解不出流场");
    let b = solve(base);
    let bref = masked_rms(&b.vx, &mask).max(1e-12);

    for (name, other) in [
        ("通风量", FlowCase { vent: 0.4, ..base }),
        ("屏蔽门", FlowCase { psd_open: false, ..base }),
        ("列车工况", FlowCase { train_phase: 3, ..base }),
    ] {
        let o = solve(other);
        let diff: Vec<f64> = b.vx.iter().zip(&o.vx).map(|(p, q)| p - q).collect();
        let rel = masked_rms(&diff, &mask) / bref;
        assert!(
            rel > 0.01,
            "改动「{}」几乎没改变流场（相对差 {:.3e}）——这一路输入网络学不到东西",
            name, rel
        );
    }
}

// ================================================================ 小件

#[test]
fn log_times_are_increasing_and_log_spaced() {
    let t = log_times(5.0, 320.0, 10);
    assert_eq!(t.len(), 10);
    assert!((t[0] - 5.0).abs() < 1e-9 && (t[9] - 320.0).abs() < 1e-6, "端点不对：{:?}", t);
    assert!(t.windows(2).all(|p| p[1] > p[0]), "时刻没有递增：{:?}", t);
    // 对数等距 <=> 相邻比值恒定
    let ratios: Vec<f64> = t.windows(2).map(|p| p[1] / p[0]).collect();
    let spread = ratios.iter().fold(0.0f64, |a, &b| a.max(b))
        - ratios.iter().fold(f64::MAX, |a: f64, &b| a.min(b));
    assert!(spread < 1e-9, "相邻比值不恒定，不是对数等距：{:?}", ratios);
    // 退化输入不能 panic
    assert_eq!(log_times(5.0, 320.0, 1).len(), 1);
    assert!(log_times(5.0, 320.0, 0).is_empty());
    assert_eq!(log_times(0.0, 10.0, 3), vec![10.0]);
}

#[test]
fn source_map_peaks_at_one_and_stays_inside() {
    let mask = station_mask(H, W).expect("造不出站体掩码");
    for &(cy, cx) in synth::source_positions().iter() {
        let s = source_map(H, W, cy, cx, 0.04, &mask);
        let peak = s.iter().fold(0.0f64, |a, &b| a.max(b));
        assert!(
            (peak - 1.0).abs() < 1e-9,
            "点位 ({:.2},{:.2}) 的源峰值是 {:.4}，不是 1——§3.3 的线性缩放前提要求源强固定",
            cy, cx, peak
        );
        let outside = s
            .iter()
            .enumerate()
            .filter(|(i, _)| mask[*i] <= 0.5)
            .map(|(_, v)| v.abs())
            .fold(0.0f64, f64::max);
        assert_eq!(outside, 0.0, "点位 ({:.2},{:.2}) 的源漏到计算域外了", cy, cx);
    }
    assert!(
        synth::source_positions().len() > synth::N_TRAIN_SOURCES,
        "必须留出不参与训练的点位，否则 §7.1 的外推测试无从做起"
    );
}

#[test]
fn stable_dt_respects_both_limits() {
    let (dx, dy) = (0.5, 0.25);
    // 只有扩散
    let zero = vec![0.0; 4];
    let d = 0.5;
    let dt = stable_dt(&zero, &zero, dx, dy, d);
    let limit = 0.5 / (d * (1.0 / (dx * dx) + 1.0 / (dy * dy)));
    assert!(dt <= limit && dt > 0.0, "扩散限：dt={} 上限={}", dt, limit);
    // 加上大速度后必须更小
    let fast = vec![20.0; 4];
    assert!(stable_dt(&fast, &fast, dx, dy, d) < dt, "速度变大后步长没变小");
    // 无扩散也不能给出 inf
    assert!(stable_dt(&fast, &fast, dx, dy, 0.0).is_finite());
}

#[test]
fn bernoulli_pressure_drops_where_flow_is_fast() {
    let vx = vec![0.0, 3.0];
    let vy = vec![0.0, 4.0];
    let p = bernoulli_pressure(&vx, &vy, 1.2, 101_325.0);
    assert!((p[0] - 101_325.0).abs() < 1e-9, "静止处应当等于参考压力");
    // |v| = 5 -> 0.5 * 1.2 * 25 = 15
    assert!((p[1] - (101_325.0 - 15.0)).abs() < 1e-9, "动压算错了：{}", p[1]);
}

#[test]
fn solve_stream_rejects_degenerate_input() {
    let sc = Scenario::default();
    let (dx, dy) = sc.spacing(8, 8);
    let empty = vec![0.0f32; 64];
    assert!(
        solve_stream(&empty, &vec![0.0; 64], 8, 8, dx, dy, 10, 1e-9).is_err(),
        "全零掩码应当被拒绝"
    );
    assert!(
        solve_stream(&vec![1.0f32; 64], &vec![0.0; 63], 8, 8, dx, dy, 10, 1e-9).is_err(),
        "形状不符应当被拒绝"
    );
}

#[test]
fn sor_actually_converges() {
    let mask = station_mask(H, W).expect("造不出站体掩码");
    let sc = Scenario::default();
    let (dx, dy) = sc.spacing(H, W);
    let om = eser::synth::station_mask(H, W)
        .unwrap()
        .iter()
        .map(|&m| m as f64)
        .collect::<Vec<_>>();
    let (_, r_few) = solve_stream(&mask, &om, H, W, dx, dy, 20, 0.0).expect("解不出 ψ");
    let (_, r_many) = solve_stream(&mask, &om, H, W, dx, dy, 4000, 0.0).expect("解不出 ψ");
    assert!(
        r_many < r_few * 1e-3,
        "迭代 4000 次的残差 {:.3e} 相比 20 次的 {:.3e} 没有明显下降，SOR 可能没在收敛",
        r_many, r_few
    );
}
