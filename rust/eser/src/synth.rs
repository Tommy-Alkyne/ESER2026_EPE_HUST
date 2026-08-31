//! 按《地铁站污染物扩散代理模型·实施方案》§4 造**满足方程**的合成数据。
//!
//! # 这不是 CFD，也不打算冒充 CFD
//!
//! 没有湍流模型、没有浮力、没有真实站体几何。它换来的是另一样东西：产出的场
//! **严格满足方案里保留的那两个方程**，因此物理残差项在上面训练是有意义的，
//! 而不是像三角函数合成场那样把模型往随机方向拉。当筛选/联调用可以，
//! 当最终校核依据不行。
//!
//! # 为什么用流函数而不是速度势
//!
//! 方案 §3.5 定案：阶段一只保留连续性方程 `div(v) = 0`。所以数据必须**在损失
//! 实际用的那个离散算子下**满足它，而不是「在连续意义下差不多满足」。
//!
//! 取 `vx = ∂ψ/∂y`、`vy = -∂ψ/∂x`，两者都用中心差分，则
//!
//! ```text
//! div = ∂vx/∂x + ∂vy/∂y
//!     = [ψ(i+1,j+1) - ψ(i-1,j+1) - ψ(i+1,j-1) + ψ(i-1,j-1)] / (4·dx·dy)
//!       - [ψ(i+1,j+1) - ψ(i+1,j-1) - ψ(i-1,j+1) + ψ(i-1,j-1)] / (4·dx·dy)
//!     = 0
//! ```
//!
//! **两个方括号逐项相同**，差出来恒等于 0——不是「趋近 0」，是浮点意义上的抵消。
//! 换成速度势 `v = ∇φ` 就没这个性质：那时 `div(v)` 是宽模板 Laplacian，要靠解方程
//! 压到小量，残留量级取决于迭代次数。
//!
//! 这条性质由 `tests/synth_physics.rs` 用 `physics::divergence`（损失里同一个函数）
//! 实测钉住。
//!
//! # 已知的物理局限（不要当成 bug）
//!
//! * **无净通流**：`ψ` 在整个计算域边界上取 0，等价于「墙面不透风且没有对外的进出口」。
//!   流动由内部涡量驱动（可以理解成风机/活塞风在域内做功），是回流而非贯穿流。
//!   要贯穿流得让 `ψ` 在边界上分段取不同常数，那需要沿边界走一圈定序，对任意掩码
//!   不平凡；当前不值这个复杂度。
//! * **无粘无涡以外的一切**：没有边界层、没有分离、没有雷诺应力。
//! * **扩散系数用的是湍流量级**，不是分子扩散（见 [`Scenario::diffusivity`]）。

use crate::geometry::{rect_mask, Segment};

// ================================================================ 站体几何

/// 简化的地铁站平面：一条贯通的站台层走廊 + 两个竖向出入口通道 + 两段楼梯口。
///
/// 与 `dataset/station_mask.npy` 同构，但**留了 2 格以上的图像边距**：
/// 中心差分在图像边界靠 replicate padding 补值，域紧贴图像边会让那里的
/// `div` 抵消不干净。留边距之后边界附近的场恒为 0，padding 补的也是 0，抵消成立。
pub fn station_segments() -> Vec<Segment> {
    vec![
        // 站台层走廊（横向贯通，两端各留 6% 边距）
        Segment::new(0.42, 0.58, 0.06, 0.94),
        // 左出入口竖井
        Segment::new(0.10, 0.42, 0.20, 0.29),
        // 右出入口竖井
        Segment::new(0.10, 0.42, 0.70, 0.79),
        // 站厅层联络通道（走廊下方两段楼梯口）
        Segment::new(0.58, 0.70, 0.44, 0.50),
        Segment::new(0.58, 0.70, 0.58, 0.64),
    ]
}

/// 在 `h x w` 网格上生成站体掩码（1 = 计算域内）。
pub fn station_mask(h: usize, w: usize) -> Result<Vec<f32>, String> {
    rect_mask(h, w, &station_segments())
}

// ================================================================ 工况

/// 一个通风工况。三个自由度对应方案 §1 的「通风量 / 屏蔽门 / 列车工况」。
#[derive(Debug, Clone, Copy)]
pub struct FlowCase {
    /// 通风强度，缩放全部涡量源。0 附近是「几乎静止」，1.0 是设计工况
    pub vent: f64,
    /// 屏蔽门开闭：关门时站台层与竖井之间的那个涡量源被掐掉
    pub psd_open: bool,
    /// 列车工况：0=无车 1=进站 2=停站 3=出站。改的是活塞风涡量源的位置与符号
    pub train_phase: u8,
}

impl FlowCase {
    /// 铺开成 `(N, 3, H, W)` 输入张量里的三个标量（各自会被广播成常数图）。
    ///
    /// **列车工况归一化到 [0,1] 而不是原样给 0..3**：三个输入通道量级差一个数量级时，
    /// 网络第一层要靠权重去补，收敛明显更慢。
    pub fn as_channels(&self) -> [f32; 3] {
        [
            self.vent as f32,
            if self.psd_open { 1.0 } else { 0.0 },
            self.train_phase as f32 / 3.0,
        ]
    }
}

/// 物性与标度。字段名与 `eser-train` 的同名旗标一一对应，方便照抄。
#[derive(Debug, Clone, Copy)]
pub struct Scenario {
    /// 计算域物理尺寸（米），`(Ly, Lx)`。对应 `--domain-size`
    pub domain_size: (f64, f64),
    /// 空气密度 kg/m³
    pub rho: f64,
    /// **湍流**扩散系数 m²/s，对应 `--diffusivity`。
    ///
    /// 刻意不用分子扩散的 1e-5：地铁站是高 Re 湍流，实际输运由湍流扩散主导，
    /// 量级在 0.1~1 m²/s。而且这也是中心差分稳定的前提——格点 Péclet 数
    /// `|u|·dx/D` 必须小于 2，取 1e-5 会让格点 Péclet 数上万，中心差分直接振荡。
    /// 用上风格式能稳，但那样算出的场在**中心差分**残差下就不再满足方程了，
    /// 而损失里量的正是中心差分残差。
    pub diffusivity: f64,
    /// 温度场的热扩散系数 m²/s（同样是湍流量级）
    pub thermal_diffusivity: f64,
    /// 参考压力 Pa（伯努利里的 P₀）
    pub p_ref: f64,
    /// 参考温度 ℃
    pub t_ref: f64,
}

impl Default for Scenario {
    fn default() -> Self {
        Self {
            // 一个中等车站：60 m 长、12 m 高
            domain_size: (12.0, 60.0),
            rho: 1.2,
            diffusivity: 0.5,
            thermal_diffusivity: 0.5,
            p_ref: 101_325.0,
            t_ref: 26.0,
        }
    }
}

impl Scenario {
    /// 网格间距。与 `cli::Args::spacing` 同口径：除以 N−1，不是 N。
    pub fn spacing(&self, h: usize, w: usize) -> (f64, f64) {
        let dy = self.domain_size.0 / (h.max(2) - 1) as f64;
        let dx = self.domain_size.1 / (w.max(2) - 1) as f64;
        (dx, dy)
    }
}

// ================================================================ 涡量源

/// 一个高斯涡量斑：`(中心 y 比例, 中心 x 比例, 半径比例, 强度)`。
#[derive(Debug, Clone, Copy)]
struct Blob {
    cy: f64,
    cx: f64,
    r: f64,
    amp: f64,
}

/// 把工况翻译成涡量场 ω。
///
/// 这是「工况 → 流场」这条映射的全部随机性来源，网络要学的就是它。
fn vorticity(h: usize, w: usize, case: &FlowCase) -> Vec<f64> {
    let mut blobs = vec![
        // 两个竖井底部的主环流，方向相反 -> 左进右出的回流
        Blob { cy: 0.38, cx: 0.245, r: 0.055, amp: 1.0 },
        Blob { cy: 0.38, cx: 0.745, r: 0.055, amp: -1.0 },
        // 走廊中段的次级涡，让场不至于左右完全对称
        Blob { cy: 0.50, cx: 0.50, r: 0.07, amp: 0.45 },
    ];
    if case.psd_open {
        // 屏蔽门开着时，站厅联络通道口多两个涡——关门时这两个消失，
        // 这是网络能从「屏蔽门」这一路输入上学到的差别
        blobs.push(Blob { cy: 0.64, cx: 0.47, r: 0.04, amp: -0.6 });
        blobs.push(Blob { cy: 0.64, cx: 0.61, r: 0.04, amp: 0.6 });
    }
    // 活塞风：车在哪一侧，哪一侧多一个强涡；停站时没有
    let piston = match case.train_phase {
        1 => Some((0.30, 1.2)),  // 进站：从左侧推
        2 => None,               // 停站
        3 => Some((0.70, -1.2)), // 出站：向右侧抽
        _ => None,               // 无车
    };
    if let Some((cx, amp)) = piston {
        blobs.push(Blob { cy: 0.50, cx, r: 0.06, amp });
    }

    let mut om = vec![0.0f64; h * w];
    for b in &blobs {
        let (cy, cx) = (b.cy * h as f64, b.cx * w as f64);
        let rr = (b.r * (h.min(w)) as f64).max(1.0);
        let amp = b.amp * case.vent;
        for i in 0..h {
            for j in 0..w {
                let dy = i as f64 - cy;
                let dx = j as f64 - cx;
                let e = -(dy * dy + dx * dx) / (2.0 * rr * rr);
                if e > -30.0 {
                    om[i * w + j] += amp * e.exp();
                }
            }
        }
    }
    om
}

// ================================================================ 流函数

/// 解 `∇²ψ = -ω`，计算域外与边界上 `ψ = 0`。SOR 迭代。
///
/// 边界条件取常数（这里是 0）保证**壁面不透风**：`ψ` 沿壁面不变 ⇒ 壁面法向速度
/// `v·n = -∂ψ/∂s = 0`。取 0 而不是分段常数的代价见本模块头部的「已知局限」。
///
/// 返回收敛时的最大残差，调用方据此判断迭代够不够。
#[allow(clippy::too_many_arguments)]
pub fn solve_stream(
    mask: &[f32],
    omega: &[f64],
    h: usize,
    w: usize,
    dx: f64,
    dy: f64,
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, f64), String> {
    if mask.len() != h * w || omega.len() != h * w {
        return Err(format!("形状不符：mask {} omega {} 期望 {}", mask.len(), omega.len(), h * w));
    }
    let (ix2, iy2) = (1.0 / (dx * dx), 1.0 / (dy * dy));
    let denom = 2.0 * (ix2 + iy2);
    // SOR 松弛因子。1.9 对几百格的网格是常用的经验值；收敛判据仍以残差为准
    let relax = 1.9;

    // 只在「四邻都在域内」的格点上更新：贴着壁面的格点保持 ψ=0，这就是边界条件
    let interior: Vec<bool> = (0..h * w)
        .map(|k| {
            let (i, j) = (k / w, k % w);
            if i == 0 || j == 0 || i + 1 >= h || j + 1 >= w || mask[k] <= 0.5 {
                return false;
            }
            mask[k - w] > 0.5 && mask[k + w] > 0.5 && mask[k - 1] > 0.5 && mask[k + 1] > 0.5
        })
        .collect();
    if !interior.iter().any(|&b| b) {
        return Err("掩码内没有任何四邻俱全的格点，网格太粗或几何太窄".into());
    }

    let mut psi = vec![0.0f64; h * w];
    let mut resid = f64::INFINITY;
    for _ in 0..max_iter {
        resid = 0.0f64;
        for k in 0..h * w {
            if !interior[k] {
                continue;
            }
            let sum = (psi[k - 1] + psi[k + 1]) * ix2 + (psi[k - w] + psi[k + w]) * iy2;
            let target = (sum + omega[k]) / denom;
            let delta = target - psi[k];
            psi[k] += relax * delta;
            resid = resid.max(delta.abs());
        }
        if resid < tol {
            break;
        }
    }
    Ok((psi, resid))
}

// ================================================================ 中心差分

/// 一阶中心差分，**与 `ops::grad_xy` 逐格等价**：图像边界用 replicate 补值。
///
/// 自己写一份而不是走张量算子，是因为生成器全程在 `f64` 上算——`ops` 那套是
/// `f32` 张量，中间降精度会让「div 抵消到机器精度」这条性质掉两个数量级。
/// 两份实现的一致性由 `tests/synth_physics.rs` 直接对拍。
pub fn central_diff(field: &[f64], h: usize, w: usize, dx: f64, dy: f64) -> (Vec<f64>, Vec<f64>) {
    let at = |i: isize, j: isize| -> f64 {
        let ii = i.clamp(0, h as isize - 1) as usize;
        let jj = j.clamp(0, w as isize - 1) as usize;
        field[ii * w + jj]
    };
    let mut gx = vec![0.0; h * w];
    let mut gy = vec![0.0; h * w];
    for i in 0..h as isize {
        for j in 0..w as isize {
            let k = i as usize * w + j as usize;
            gx[k] = (at(i, j + 1) - at(i, j - 1)) / (2.0 * dx);
            gy[k] = (at(i + 1, j) - at(i - 1, j)) / (2.0 * dy);
        }
    }
    (gx, gy)
}

/// 五点 Laplacian，与 `ops::laplacian` 同口径。
pub fn laplacian(field: &[f64], h: usize, w: usize, dx: f64, dy: f64) -> Vec<f64> {
    let at = |i: isize, j: isize| -> f64 {
        let ii = i.clamp(0, h as isize - 1) as usize;
        let jj = j.clamp(0, w as isize - 1) as usize;
        field[ii * w + jj]
    };
    let mut out = vec![0.0; h * w];
    for i in 0..h as isize {
        for j in 0..w as isize {
            let k = i as usize * w + j as usize;
            let c = at(i, j);
            out[k] = (at(i, j + 1) - 2.0 * c + at(i, j - 1)) / (dx * dx)
                + (at(i + 1, j) - 2.0 * c + at(i - 1, j)) / (dy * dy);
        }
    }
    out
}

/// `v = (∂ψ/∂y, -∂ψ/∂x)`。**符号不能反**：反了就成了 `∇ψ` 旋转 90°的另一支，
/// `div` 不再抵消（`tests/synth_physics.rs` 里有一条变异测试专门守这个）。
pub fn velocity_from_stream(
    psi: &[f64],
    h: usize,
    w: usize,
    dx: f64,
    dy: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (dpsi_dx, dpsi_dy) = central_diff(psi, h, w, dx, dy);
    let vx = dpsi_dy;
    let vy: Vec<f64> = dpsi_dx.iter().map(|v| -v).collect();
    (vx, vy)
}

/// 伯努利压力 `P = P₀ - ½ρ|v|²`。
///
/// 无旋流下这是精确的。本模块的流场由涡量驱动、并非处处无旋，所以这只是一个
/// **量级和空间结构都合理的压力代理**，不是该流场真正的压力解。阶段一的物理
/// 约束只有 `div(v)=0`（§3.5 已去掉动量残差），压力不进残差，够用。
pub fn bernoulli_pressure(vx: &[f64], vy: &[f64], rho: f64, p_ref: f64) -> Vec<f64> {
    vx.iter()
        .zip(vy)
        .map(|(&u, &v)| p_ref - 0.5 * rho * (u * u + v * v))
        .collect()
}

// ================================================================ 对流扩散

/// 显式推进一步 `∂c/∂t + u·∇c = D∇²c + S`，全程中心差分，**壁面无通量**。
///
/// # 壁面为什么必须是零梯度而不是零值
///
/// 最初这里把域外当成常数 0，差分模板伸出去就取到 0。那等价于把墙做成**完美吸收体**：
/// 实测 20 秒内掩码内总量从 182.96 掉到 0.0006——污染物没有扩散开，是被墙吃光了。
/// 真实站体壁面是不可穿透的（无通量），浓度应当在墙根堆积而不是消失。
///
/// 改法是：模板碰到域外邻居时，用**中心格点自己的值**顶替。于是那个面上的差分
/// `c_邻 - c_中 = 0`，通量为零。离散上每个内部面的贡献成对相消，总量因此严格守恒
/// （由 `tests/synth_physics.rs` 的守恒测试钉住）。
///
/// 写成就地更新（`out` 由调用方复用）不是过早优化：温度场要推上万步，每步分配四个
/// `Vec` 会让一次生成从秒级掉到分钟级。
#[allow(clippy::too_many_arguments)]
fn step_scalar_into(
    out: &mut [f64],
    c: &[f64],
    vx: &[f64],
    vy: &[f64],
    mask: &[f32],
    source: &[f64],
    h: usize,
    w: usize,
    dx: f64,
    dy: f64,
    d: f64,
    dt: f64,
) {
    for i in 0..h {
        for j in 0..w {
            let k = i * w + j;
            if mask[k] <= 0.5 {
                out[k] = 0.0;
                continue;
            }
            let c0 = c[k];
            // 域外邻居用中心值顶替 => 该方向差分为 0 => 无通量
            let nb = |cond: bool, idx: usize| -> f64 {
                if cond && mask[idx] > 0.5 {
                    c[idx]
                } else {
                    c0
                }
            };
            let e = nb(j + 1 < w, k + 1);
            let we = nb(j > 0, k.wrapping_sub(1));
            let n = nb(i + 1 < h, k + w);
            let so = nb(i > 0, k.wrapping_sub(w));

            let dc_dx = (e - we) / (2.0 * dx);
            let dc_dy = (n - so) / (2.0 * dy);
            let lap = (e - 2.0 * c0 + we) / (dx * dx) + (n - 2.0 * c0 + so) / (dy * dy);
            let adv = vx[k] * dc_dx + vy[k] * dc_dy;
            out[k] = c0 + dt * (-adv + d * lap + source[k]);
        }
    }
}

/// 稳定的时间步长：对流 CFL 与扩散显式稳定条件取较严者，再乘 0.4 的安全系数。
pub fn stable_dt(vx: &[f64], vy: &[f64], dx: f64, dy: f64, d: f64) -> f64 {
    let umax = vx
        .iter()
        .zip(vy)
        .map(|(a, b)| a.abs() / dx + b.abs() / dy)
        .fold(0.0f64, f64::max);
    let adv = if umax > 0.0 { 1.0 / umax } else { f64::INFINITY };
    let diff = if d > 0.0 {
        0.5 / (d * (1.0 / (dx * dx) + 1.0 / (dy * dy)))
    } else {
        f64::INFINITY
    };
    0.4 * adv.min(diff)
}

/// 从 `c` 推进到物理时刻 `t_end`，内部自动细分成若干稳定步。
///
/// 返回 `(场, 实际用了多少步)`。步数返回出来是给上层报进度的——一个 300 秒的
/// 算例可能要走上万步，闷头跑很难判断是慢还是卡死。
#[allow(clippy::too_many_arguments)]
pub fn advance(
    c: &[f64],
    vx: &[f64],
    vy: &[f64],
    mask: &[f32],
    source: &[f64],
    h: usize,
    w: usize,
    dx: f64,
    dy: f64,
    d: f64,
    dt: f64,
    t_span: f64,
) -> (Vec<f64>, usize) {
    if t_span <= 0.0 {
        return (c.to_vec(), 0);
    }
    let n = (t_span / dt).ceil().max(1.0) as usize;
    let h_step = t_span / n as f64;
    let mut cur = c.to_vec();
    let mut nxt = vec![0.0f64; h * w];
    for _ in 0..n {
        step_scalar_into(&mut nxt, &cur, vx, vy, mask, source, h, w, dx, dy, d, h_step);
        std::mem::swap(&mut cur, &mut nxt);
    }
    (cur, n)
}

/// 对数间隔的采样时刻，对应方案 §4.3。
///
/// `5, 8, 13, 20, 32, 50, 80, 125, 200, 320 ...` 那一串就是这么来的：泄漏早期
/// 梯度大、后期趋于准稳态，均匀采样在早期欠采、后期浪费。
pub fn log_times(t0: f64, t_end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 || t0 <= 0.0 || t_end <= t0 {
        return vec![t_end];
    }
    let (a, b) = (t0.ln(), t_end.ln());
    (0..n)
        .map(|i| (a + (b - a) * i as f64 / (n - 1) as f64).exp())
        .collect()
}

/// 点源的高斯栅格化，对应方案 §4.2 的「源位置图」。
///
/// **源强恒为 1**（§3.3）：对流扩散对 C 是线性的，源强翻倍浓度处处翻倍，
/// 所以不采样这个维度，用的时候乘系数即可。
pub fn source_map(h: usize, w: usize, cy: f64, cx: f64, radius: f64, mask: &[f32]) -> Vec<f64> {
    let (yc, xc) = (cy * h as f64, cx * w as f64);
    let r = (radius * h.min(w) as f64).max(1.0);
    let mut s: Vec<f64> = (0..h * w)
        .map(|k| {
            if mask[k] <= 0.5 {
                return 0.0;
            }
            let dy = (k / w) as f64 - yc;
            let dx = (k % w) as f64 - xc;
            let e = -(dy * dy + dx * dx) / (2.0 * r * r);
            if e > -30.0 {
                e.exp()
            } else {
                0.0
            }
        })
        .collect();
    // 归一到峰值 1：不同点位落在宽窄不同的通道里，不归一的话源强其实是随位置变的，
    // 而 §3.3 的线性缩放前提是源强固定
    let peak = s.iter().fold(0.0f64, |a, &b| a.max(b));
    if peak > 0.0 {
        for v in &mut s {
            *v /= peak;
        }
    }
    s
}

/// 方案 §1 的固定关心点位（比例坐标）。前 7 个用于训练，后 3 个留作外推测试（§7.1）。
pub fn source_positions() -> Vec<(f64, f64)> {
    vec![
        (0.50, 0.12),
        (0.50, 0.25),
        (0.26, 0.245),
        (0.50, 0.38),
        (0.64, 0.47),
        (0.50, 0.62),
        (0.26, 0.745),
        // ---- 以下三个不参与训练，专做外推 ----
        (0.50, 0.50),
        (0.64, 0.61),
        (0.50, 0.88),
    ]
}

/// 训练用点位的个数；其余是外推测试点位。
pub const N_TRAIN_SOURCES: usize = 7;

// ================================================================ 温度场

/// 稳态温度场：把环境热源推到准稳态。
///
/// 阶段一输出 `T` 但**不进物理残差**（§3.5），所以这里只要是一个由同一个流场
/// 输运出来、空间结构合理的标量场即可。热源摆在客流集中处（楼梯口、站台中部）。
#[allow(clippy::too_many_arguments)]
pub fn steady_temperature(
    vx: &[f64],
    vy: &[f64],
    mask: &[f32],
    h: usize,
    w: usize,
    dx: f64,
    dy: f64,
    sc: &Scenario,
) -> Vec<f64> {
    let mut src = vec![0.0f64; h * w];
    for (cy, cx, amp) in [(0.64, 0.47, 1.0), (0.64, 0.61, 1.0), (0.50, 0.50, 0.6)] {
        let g = source_map(h, w, cy, cx, 0.05, mask);
        for (s, v) in src.iter_mut().zip(&g) {
            *s += amp * v;
        }
    }
    let dt = stable_dt(vx, vy, dx, dy, sc.thermal_diffusivity);
    // 推到扩散平衡时间的若干倍即可。竖直方向最窄（走廊高约 2 m），
    // 特征时间 L²/D ≈ 8 s，取 60 s 已经很富余；原来写 400 s 纯属拍脑袋，
    // 代价是六万多步显式推进，一次生成要跑几分钟
    let t_span = 60.0;
    let (rise, _) = advance(
        &vec![0.0; h * w],
        vx,
        vy,
        mask,
        &src,
        h,
        w,
        dx,
        dy,
        sc.thermal_diffusivity,
        dt,
        t_span,
    );
    // 温升幅度按峰值 8 ℃ 标定——只影响量纲好看，不影响任何残差
    let peak = rise.iter().fold(0.0f64, |a, &b| a.max(b));
    let k = if peak > 0.0 { 8.0 / peak } else { 0.0 };
    rise.iter()
        .enumerate()
        .map(|(i, &v)| if mask[i] > 0.5 { sc.t_ref + k * v } else { 0.0 })
        .collect()
}

// ================================================================ 一个完整工况

/// 阶段一的一个稳态解。
pub struct FlowSolution {
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub p: Vec<f64>,
    pub t: Vec<f64>,
    /// SOR 退出时的最大增量，用来判断迭代够不够
    pub residual: f64,
}

/// 解一个通风工况的稳态流场 + 温度场。
pub fn solve_flow(
    mask: &[f32],
    h: usize,
    w: usize,
    case: &FlowCase,
    sc: &Scenario,
    max_iter: usize,
) -> Result<FlowSolution, String> {
    let (dx, dy) = sc.spacing(h, w);
    let om = vorticity(h, w, case);
    let (psi, residual) = solve_stream(mask, &om, h, w, dx, dy, max_iter, 1e-10)?;
    let (vx, vy) = velocity_from_stream(&psi, h, w, dx, dy);
    let p = bernoulli_pressure(&vx, &vy, sc.rho, sc.p_ref);
    let t = steady_temperature(&vx, &vy, mask, h, w, dx, dy, sc);
    // 域外置零：训练侧的掩码也会把域外排除，两边一致才不会在边界上产生假梯度
    let zero_outside = |f: Vec<f64>| -> Vec<f64> {
        f.into_iter()
            .enumerate()
            .map(|(i, v)| if mask[i] > 0.5 { v } else { 0.0 })
            .collect()
    };
    Ok(FlowSolution {
        vx: zero_outside(vx),
        vy: zero_outside(vy),
        p: zero_outside(p),
        t,
        residual,
    })
}
