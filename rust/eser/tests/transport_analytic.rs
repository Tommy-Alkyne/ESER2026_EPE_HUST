//! 解析解验证：证明对流扩散方程**写对了**，而不只是「跑起来了」。
//!
//! PINN 的方程写错不会报错，只会输出「看起来很光滑很合理但完全错误」的场，事后
//! 极难发现。黄金数据能保证 Rust 版与 Python 版算的是同一个东西，但如果 Python
//! 版本身写错了，两边会一起错。所以还需要一条与语言、与实现都无关的证据链：把
//! 已知解析解代进残差，看残差是不是趋近 0。
//!
//! 而「残差很小」本身是弱证据——把 D 写成 0、把速度乘 0.001，残差也会很小。所以
//! 主检验是**收敛阶**：中心差分的截断误差是 O(h²)，网格加密一倍残差应下降约 4 倍。
//! 方程写错时这个 4 倍关系立刻破掉（残差由错误项主导，不随网格收敛）。
//!
//! 用到的解析解：
//!
//! 1. **移动高斯烟团**（瞬态，精确满足含时间项的完整方程）
//!    `C(x,y,t) = 1/(4πDt)·exp(-((x-ut)² + (y-vt)²)/(4Dt))`
//! 2. **一维指数解**（稳态，精确满足 `u·dC/dx = D·d²C/dx²`）
//!    `C(x) = exp(u·x/D)`
//!
//! 全程用 **f64**：f32 的舍入误差会盖过 O(h²) 截断误差，让收敛阶测试失去意义。

use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use eser::ops::{erode_mask, grad_xy, masked_mean};
use eser::physics::{scalar, Fields};
use eser::transport::{
    advection_diffusion_loss, advection_diffusion_residual, central_time_derivative,
};

type B = NdArray<f64>;

// ------------------------------------------------------------------ 构造工具

/// 返回 `(coords, h)`。`h` 由坐标本身相减得到，与 Python 的
/// `float(coords[1] - coords[0])` 逐位一致。
fn grid(n: usize, lo: f64, hi: f64) -> (Vec<f64>, f64) {
    let coords: Vec<f64> = (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect();
    let h = coords[1] - coords[0];
    (coords, h)
}

fn t4<Bk: Backend>(v: &[f64], n: usize) -> Tensor<Bk, 4> {
    Tensor::<Bk, 1>::from_data(TensorData::from(v), &Default::default()).reshape([1, 1, n, n])
}

fn full4<Bk: Backend>(value: f64, n: usize) -> Tensor<Bk, 4> {
    Tensor::<Bk, 4>::full([1, 1, n, n], value, &Default::default())
}

/// 移动高斯烟团解析解。
fn gaussian_puff(coords: &[f64], t: f64, u: f64, v: f64, d: f64) -> Vec<f64> {
    let denom = 4.0 * d * t;
    let amp = 1.0 / (std::f64::consts::PI * denom);
    let n = coords.len();
    let mut out = vec![0.0; n * n];
    for iy in 0..n {
        for ix in 0..n {
            let r2 = (coords[ix] - u * t).powi(2) + (coords[iy] - v * t).powi(2);
            out[iy * n + ix] = amp * (-r2 / denom).exp();
        }
    }
    out
}

/// 烟团解的 ∂C/∂t 解析式。
///
/// 用解析时间导数而不是时间上的中心差分：后者自带 O(Δt²) 误差，会在细网格上变成
/// 残差的下限，把空间项的 O(h²) 收敛阶压平（Python 侧实测 n=257 时比值从 4 掉到 1）。
fn gaussian_puff_dcdt(coords: &[f64], t: f64, u: f64, v: f64, d: f64) -> Vec<f64> {
    let denom = 4.0 * d * t;
    let c = gaussian_puff(coords, t, u, v, d);
    let n = coords.len();
    let mut out = vec![0.0; n * n];
    for iy in 0..n {
        for ix in 0..n {
            let (ax, ay) = (coords[ix] - u * t, coords[iy] - v * t);
            let r2 = ax * ax + ay * ay;
            let dexp_dt = (2.0 * u * ax + 2.0 * v * ay) / denom + r2 / (denom * t);
            out[iy * n + ix] = c[iy * n + ix] * (-1.0 / t + dexp_dt);
        }
    }
    out
}

/// 全 1 掩码向内腐蚀 `radius` 圈。
///
/// 差分算子在边界上用 replicate padding，那里的导数本来就不对，必须排除掉，否则
/// 测的是边界伪影而不是方程本身。
fn interior_mask<Bk: Backend>(n: usize, radius: usize) -> Tensor<Bk, 4> {
    let mut mask = Tensor::<Bk, 4>::ones([1, 1, n, n], &Default::default());
    for _ in 0..radius {
        mask = erode_mask(mask, 1);
    }
    mask
}

fn rms<Bk: Backend>(x: Tensor<Bk, 4>, mask: &Tensor<Bk, 4>) -> f64 {
    scalar(masked_mean(x.clone() * x, Some(mask.clone())).sqrt())
}

// ------------------------------------------- 测试一：稳态一维指数解（精确解）

#[test]
fn steady_exponential_converges_second_order() {
    // u·dC/dx = D·d²C/dx² 的精确解 C = exp(ux/D)，残差应只剩截断误差
    let (u, v, d) = (1.0, 0.0, 1.0);
    let mut rel = [0.0f64; 2];

    for (slot, n) in [65usize, 129].iter().enumerate().map(|(i, &n)| (i, n)) {
        let (coords, h) = grid(n, 0.0, 1.0);
        let mut cv = vec![0.0; n * n];
        for iy in 0..n {
            for ix in 0..n {
                cv[iy * n + ix] = (u * coords[ix] / d).exp();
            }
        }
        let c = t4::<B>(&cv, n);
        let vx = full4::<B>(u, n);
        let vy = full4::<B>(v, n);
        let mask = interior_mask::<B>(n, 2);

        let res =
            advection_diffusion_residual(c.clone(), vx.clone(), vy.clone(), d, h, h, None, None);
        // 用平流项自身量级做无量纲化，否则「残差小」可能只是因为场本身就小
        let scale = rms(vx * grad_xy(c, h, h).0, &mask).max(f64::MIN_POSITIVE);
        rel[slot] = rms(res, &mask) / scale;
    }

    assert!(
        rel[1] < 1e-3,
        "稳态指数解的相对残差应远小于方程各项量级，n=129 实得 {:.3e}",
        rel[1]
    );
    let ratio = rel[0] / rel[1];
    assert!(
        (3.0..5.0).contains(&ratio),
        "稳态解残差应按 O(h²) 收敛（网格加倍降约 4 倍），实得比值 {:.2}",
        ratio
    );
}

// --------------------------------------- 测试二：瞬态高斯烟团（含时间导数项）

#[test]
fn transient_puff_converges_second_order() {
    // 完整方程（含 ∂C/∂t）在解析解上的收敛阶。这是最强的正确性证据。
    let (u, v, d, t) = (0.10, 0.05, 0.01, 2.0);
    let mut residuals = [0.0f64; 3];
    let mut magnitudes = [0.0f64; 3];

    for (slot, n) in [65usize, 129, 257].iter().enumerate().map(|(i, &n)| (i, n)) {
        let (coords, h) = grid(n, -1.0, 1.0);
        let c = t4::<B>(&gaussian_puff(&coords, t, u, v, d), n);
        // 时间项用解析值代入，只留空间算子的截断误差，收敛阶才测得准
        let dcdt = t4::<B>(&gaussian_puff_dcdt(&coords, t, u, v, d), n);
        let mask = interior_mask::<B>(n, 2);

        let res = advection_diffusion_residual(
            c,
            full4::<B>(u, n),
            full4::<B>(v, n),
            d,
            h,
            h,
            Some(dcdt.clone()),
            None,
        );
        residuals[slot] = rms(res, &mask);
        magnitudes[slot] = rms(dcdt, &mask);
    }

    let rel = residuals[2] / magnitudes[2];
    assert!(
        rel < 1e-2,
        "瞬态烟团残差应远小于方程各项量级，实得相对残差 {:.3e}",
        rel
    );

    let r1 = residuals[0] / residuals[1];
    let r2 = residuals[1] / residuals[2];
    assert!(
        (3.0..5.0).contains(&r1),
        "65->129 的收敛比应约为 4.0，实得 {:.2}",
        r1
    );
    assert!(
        (3.0..5.0).contains(&r2),
        "129->257 的收敛比应约为 4.0，实得 {:.2}",
        r2
    );
}

// ------------------------------------------------------- 测试三：防假绿（变异）

#[test]
fn wrong_equations_are_detected() {
    // 故意把方程写错，确认上面的判据真的能抓到——不然测试全绿也没意义。
    // 这里不改动生产代码，而是就地构造三种典型写错的残差。
    let (u, v, d, t, n) = (0.10, 0.05, 0.01, 2.0, 129usize);
    let (coords, h) = grid(n, -1.0, 1.0);
    let c = t4::<B>(&gaussian_puff(&coords, t, u, v, d), n);
    let dcdt = t4::<B>(&gaussian_puff_dcdt(&coords, t, u, v, d), n);
    let vx = full4::<B>(u, n);
    let vy = full4::<B>(v, n);
    let mask = interior_mask::<B>(n, 2);
    let scale = rms(dcdt.clone(), &mask);

    let relative = |res: Tensor<B, 4>| rms(res, &mask) / scale;

    let correct = relative(advection_diffusion_residual(
        c.clone(),
        vx.clone(),
        vy.clone(),
        d,
        h,
        h,
        Some(dcdt.clone()),
        None,
    ));
    // 变异①：扩散项符号写反（PINN 里最常见的低级错误）
    let mut_sign = relative(advection_diffusion_residual(
        c.clone(),
        vx.clone(),
        vy.clone(),
        -d,
        h,
        h,
        Some(dcdt.clone()),
        None,
    ));
    // 变异②：漏掉时间项（把瞬态当稳态）
    let mut_no_time = relative(advection_diffusion_residual(
        c.clone(),
        vx.clone(),
        vy.clone(),
        d,
        h,
        h,
        None,
        None,
    ));
    // 变异③：vx / vy 接反（等价于把风向转了 90 度）
    let mut_swap = relative(advection_diffusion_residual(
        c,
        vy,
        vx,
        d,
        h,
        h,
        Some(dcdt),
        None,
    ));

    assert!(
        correct < 1e-2,
        "正确实现的相对残差应接近 0，实得 {:.3e}",
        correct
    );
    for (name, got) in [
        ("①扩散项符号写反", mut_sign),
        ("②漏掉时间项", mut_no_time),
        ("③速度分量接反", mut_swap),
    ] {
        assert!(
            got > 100.0 * correct,
            "变异{}应被抓到：相对残差 {:.3e}，仅为正确实现 {:.3e} 的 {:.1} 倍",
            name,
            got,
            correct,
            got / correct
        );
    }
}

// ------------------------------------------------------- 测试四：掩码与梯度

#[test]
fn pollution_outside_mask_does_not_leak_in() {
    // 域外塞进巨大的垃圾值，掩码内的损失不应变化。
    // 只污染最外一圈——掩码腐蚀了 2 圈，从第 2 行起才参与计算，其 3x3 模板最远
    // 只够到第 1 行。污染第 1 行会合法地渗进来（这正是腐蚀存在的原因），不是 bug。
    let n = 33usize;
    let (coords, h) = grid(n, -1.0, 1.0);
    let c = t4::<B>(&gaussian_puff(&coords, 2.0, 0.1, 0.05, 0.01), n);
    let vx = full4::<B>(0.1, n);
    let vy = full4::<B>(0.05, n);
    let mask = interior_mask::<B>(n, 2);

    let make = |c: Tensor<B, 4>| Fields {
        c: Some(c),
        vx: Some(vx.clone()),
        vy: Some(vy.clone()),
        p: None,
    };
    let loss = |f: &Fields<B>| {
        scalar(advection_diffusion_loss(f, Some(mask.clone()), 0.01, h, h, None, None).unwrap())
    };

    let clean = loss(&make(c.clone()));
    assert!(clean.is_finite(), "损失必须是有限标量，实得 {}", clean);

    let big = full4::<B>(1e6, n);
    let neg = full4::<B>(-1e6, n);
    let polluted = c
        .slice_assign(
            [0..1, 0..1, 0..1, 0..n],
            big.clone().slice([0..1, 0..1, 0..1, 0..n]),
        )
        .slice_assign(
            [0..1, 0..1, n - 1..n, 0..n],
            neg.clone().slice([0..1, 0..1, 0..1, 0..n]),
        )
        .slice_assign(
            [0..1, 0..1, 0..n, 0..1],
            big.slice([0..1, 0..1, 0..n, 0..1]),
        )
        .slice_assign(
            [0..1, 0..1, 0..n, n - 1..n],
            neg.slice([0..1, 0..1, 0..n, 0..1]),
        );

    let dirty = loss(&make(polluted));
    assert!(
        (clean - dirty).abs() < 1e-9 * clean.abs().max(1.0),
        "掩码外的污染渗进了掩码内损失：{:.6e} -> {:.6e}",
        clean,
        dirty
    );
}

#[test]
fn central_time_derivative_approaches_analytic() {
    let n = 33usize;
    let (coords, _) = grid(n, -1.0, 1.0);
    let (u, v, d, t, dt) = (0.1, 0.05, 0.01, 2.0, 1e-3);
    let approx = central_time_derivative(
        t4::<B>(&gaussian_puff(&coords, t - dt, u, v, d), n),
        t4::<B>(&gaussian_puff(&coords, t + dt, u, v, d), n),
        dt,
    )
    .unwrap();
    let exact = t4::<B>(&gaussian_puff_dcdt(&coords, t, u, v, d), n);
    let mask = interior_mask::<B>(n, 2);

    let rel = rms(approx - exact.clone(), &mask) / rms(exact, &mask);
    assert!(
        rel < 1e-4,
        "中心时间差分与解析时间导数的相对误差 {:.3e}",
        rel
    );
}

#[test]
fn gradient_flows_back_to_concentration() {
    // 梯度必须能回传，否则物理约束根本没参与训练——损失打印得再好看也没用
    type AB = Autodiff<NdArray<f64>>;
    let n = 33usize;
    let (coords, h) = grid(n, -1.0, 1.0);
    let c = t4::<AB>(&gaussian_puff(&coords, 2.0, 0.1, 0.05, 0.01), n).require_grad();
    let fields = Fields {
        c: Some(c.clone()),
        vx: Some(full4::<AB>(0.1, n)),
        vy: Some(full4::<AB>(0.05, n)),
        p: None,
    };
    let mask = interior_mask::<AB>(n, 2);
    let loss = advection_diffusion_loss(&fields, Some(mask), 0.01, h, h, None, None).unwrap();

    let grads = loss.backward();
    let g = c.grad(&grads).expect("浓度场没有拿到梯度");
    let total: f64 = g.abs().sum().into_scalar();
    assert!(total > 0.0, "梯度全为 0，物理约束没有真的参与优化");
}
