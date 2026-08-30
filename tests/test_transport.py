"""对流扩散残差 + 栅格掩码的验证测试（方案文档第 1、2 步）。

独立可跑：``python -m tests.test_transport``

为什么必须做解析解验证
----------------------
PINN 的方程写错**不会报错**，只会输出「看起来很光滑很合理但完全错误」的场，事后
极难发现。唯一能证明方程写对了的办法，是把已知解析解代入残差、确认残差趋近 0。

而「残差很小」本身是弱证据——把 D 写成 0、把速度乘 0.001，残差也会很小。所以本
文件的主检验是**收敛阶**：中心差分的截断误差是 O(h²)，网格加密一倍残差应下降约
4 倍。方程写错时这个 4 倍关系立刻破掉（残差由错误项主导，不随网格收敛）。

用到的解析解
------------
1. **移动高斯烟团**（瞬态，精确满足完整方程，含时间项）

       C(x,y,t) = 1/(4πDt) · exp(-((x-ut)² + (y-vt)²) / (4Dt))

2. **一维指数解**（稳态，精确满足 u·dC/dx = D·d²C/dx²）

       C(x) = exp(u·x/D)
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.physics_loss import erode_mask, masked_mean          # noqa: E402
from utils.transport import (advection_diffusion_loss,           # noqa: E402
                             advection_diffusion_residual,
                             central_time_derivative)

PASSED = []
FAILED = []


def check(name, condition, detail=''):
    if condition:
        PASSED.append(name)
        print('[PASS] {}{}'.format(name, '  -- ' + detail if detail else ''))
    else:
        FAILED.append(name)
        print('[FAIL] {}{}'.format(name, '  -- ' + detail if detail else ''))


# ------------------------------------------------------------------ 构造工具
def grid(n, lo=-1.0, hi=1.0, dtype=torch.float64):
    """返回 (yy, xx, h)。用 float64——float32 的舍入误差会盖过 O(h²) 截断误差，
    让收敛阶测试失去意义。"""
    coords = torch.linspace(lo, hi, n, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    h = float(coords[1] - coords[0])
    return yy, xx, h


def gaussian_puff(xx, yy, t, u, v, diffusivity):
    """移动高斯烟团解析解。"""
    denom = 4.0 * diffusivity * t
    amp = 1.0 / (np.pi * denom)
    r2 = (xx - u * t) ** 2 + (yy - v * t) ** 2
    return amp * torch.exp(-r2 / denom)


def gaussian_puff_dcdt(xx, yy, t, u, v, diffusivity):
    """烟团解的 ∂C/∂t 解析式。

    用解析时间导数而不是时间上的中心差分：后者自带 O(Δt²) 误差，会在细网格上
    变成残差的下限，把空间项的 O(h²) 收敛阶压平（实测 n=257 时比值从 4 掉到 1）。
    """
    denom = 4.0 * diffusivity * t
    c = gaussian_puff(xx, yy, t, u, v, diffusivity)
    r2 = (xx - u * t) ** 2 + (yy - v * t) ** 2
    dexp_dt = (2 * u * (xx - u * t) + 2 * v * (yy - v * t)) / denom \
        + r2 / (denom * t)
    return c * (-1.0 / t + dexp_dt)


def interior_mask(shape, radius=2, dtype=torch.float64):
    """全 1 掩码向内腐蚀 radius 圈。

    差分算子在边界上用 replicate padding，那里的导数本来就不对，必须排除掉，
    否则测的是边界伪影而不是方程本身。
    """
    mask = torch.ones(1, 1, *shape, dtype=dtype)
    for _ in range(radius):
        mask = erode_mask(mask, radius=1)
    return mask


def rms(x, mask):
    return float(masked_mean(x ** 2, mask).sqrt())


# ------------------------------------------- 测试一：稳态一维指数解（精确解）
def test_steady_exponential():
    """u·dC/dx = D·d²C/dx² 的精确解 C = exp(ux/D)，残差应只剩截断误差。"""
    u, v, D = 1.0, 0.0, 1.0
    results = {}
    for n in (65, 129):
        yy, xx, h = grid(n, 0.0, 1.0)
        c = torch.exp(u * xx / D).view(1, 1, n, n)
        vx = torch.full_like(c, u)
        vy = torch.full_like(c, v)
        mask = interior_mask((n, n))
        res = advection_diffusion_residual(c, vx, vy, diffusivity=D, dx=h, dy=h)
        # 用平流项自身量级做无量纲化，否则"残差小"可能只是因为场本身就小
        scale = rms(vx * torch.gradient(c, dim=3)[0] / h, mask) or 1.0
        results[n] = rms(res, mask) / scale

    check('S1 稳态指数解残差远小于方程各项量级',
          results[129] < 1e-3,
          '相对残差 n=129: {:.3e}'.format(results[129]))
    ratio = results[65] / max(results[129], 1e-30)
    check('S1 稳态解残差按 O(h²) 收敛（网格加倍应降约 4 倍）',
          3.0 < ratio < 5.0,
          '比值 {:.2f}（理论 4.0）'.format(ratio))


# --------------------------------------- 测试二：瞬态高斯烟团（含时间导数项）
def test_transient_puff_convergence():
    """完整方程（含 ∂C/∂t）在解析解上的收敛阶。这是最强的正确性证据。"""
    u, v, D, t = 0.10, 0.05, 0.01, 2.0
    residuals = {}
    magnitudes = {}

    for n in (65, 129, 257):
        yy, xx, h = grid(n)
        c = gaussian_puff(xx, yy, t, u, v, D).view(1, 1, n, n)
        # 时间项用解析值代入，只留空间算子的截断误差，收敛阶才测得准
        dcdt = gaussian_puff_dcdt(xx, yy, t, u, v, D).view(1, 1, n, n)

        vx = torch.full_like(c, u)
        vy = torch.full_like(c, v)
        mask = interior_mask((n, n))

        res = advection_diffusion_residual(c, vx, vy, diffusivity=D,
                                           dx=h, dy=h, dcdt=dcdt)
        residuals[n] = rms(res, mask)
        magnitudes[n] = rms(dcdt, mask)

    rel = residuals[257] / max(magnitudes[257], 1e-30)
    check('S1 瞬态烟团残差远小于方程各项量级',
          rel < 1e-2, '相对残差 {:.3e}'.format(rel))

    r1 = residuals[65] / max(residuals[129], 1e-30)
    r2 = residuals[129] / max(residuals[257], 1e-30)
    check('S1 瞬态残差按 O(h²) 收敛（65->129）',
          3.0 < r1 < 5.0, '比值 {:.2f}（理论 4.0）'.format(r1))
    check('S1 瞬态残差按 O(h²) 收敛（129->257）',
          3.0 < r2 < 5.0, '比值 {:.2f}（理论 4.0）'.format(r2))


# ------------------------------------------------------- 测试三：防假绿（变异）
def test_mutation_detected():
    """故意把方程写错，确认上面的判据真的能抓到——不然测试全绿也没意义。

    这里不改动生产代码，而是就地构造三种典型写错的残差，逐一验证它们都会显著
    偏离 0（因此会被收敛阶 / 相对量级判据拒绝）。
    """
    u, v, D, t, n = 0.10, 0.05, 0.01, 2.0, 129
    yy, xx, h = grid(n)
    c = gaussian_puff(xx, yy, t, u, v, D).view(1, 1, n, n)
    dcdt = gaussian_puff_dcdt(xx, yy, t, u, v, D).view(1, 1, n, n)
    vx, vy = torch.full_like(c, u), torch.full_like(c, v)
    mask = interior_mask((n, n))
    scale = rms(dcdt, mask)

    correct = rms(advection_diffusion_residual(
        c, vx, vy, diffusivity=D, dx=h, dy=h, dcdt=dcdt), mask) / scale

    # 变异 1：扩散项符号写反（PINN 里最常见的低级错误）
    mut_sign = rms(advection_diffusion_residual(
        c, vx, vy, diffusivity=-D, dx=h, dy=h, dcdt=dcdt), mask) / scale
    # 变异 2：漏掉时间项（把瞬态当稳态）
    mut_no_time = rms(advection_diffusion_residual(
        c, vx, vy, diffusivity=D, dx=h, dy=h, dcdt=None), mask) / scale
    # 变异 3：vx / vy 接反（等价于把风向转了 90 度）
    mut_swap = rms(advection_diffusion_residual(
        c, vy, vx, diffusivity=D, dx=h, dy=h, dcdt=dcdt), mask) / scale

    check('S1 正确实现的相对残差接近 0',
          correct < 1e-2, '{:.3e}'.format(correct))
    check('S1 变异①扩散项符号写反 -> 被抓到',
          mut_sign > 100 * correct, '{:.3e} vs 正确 {:.3e}'.format(mut_sign, correct))
    check('S1 变异②漏掉时间项 -> 被抓到',
          mut_no_time > 100 * correct,
          '{:.3e} vs 正确 {:.3e}'.format(mut_no_time, correct))
    check('S1 变异③速度分量接反 -> 被抓到',
          mut_swap > 100 * correct, '{:.3e} vs 正确 {:.3e}'.format(mut_swap, correct))


# ------------------------------------------------------- 测试四：接口与掩码行为
def test_loss_interface():
    n = 33
    yy, xx, h = grid(n)
    c = gaussian_puff(xx, yy, 2.0, 0.1, 0.05, 0.01).view(1, 1, n, n)
    vx, vy = torch.full_like(c, 0.1), torch.full_like(c, 0.05)
    mask = interior_mask((n, n))

    loss = advection_diffusion_loss({'c': c, 'vx': vx, 'vy': vy}, mask,
                                    diffusivity=0.01, dx=h, dy=h)
    check('S1 损失返回标量且有限',
          loss is not None and loss.dim() == 0 and torch.isfinite(loss))

    check('S1 缺少速度场时返回 None（而不是静默算错）',
          advection_diffusion_loss({'c': c}, mask, dx=h, dy=h) is None)

    # 掩码必须真的起作用：域外塞进巨大的垃圾值，掩码内的损失不应变化。
    # 只污染最外一圈——掩码腐蚀了 2 圈，从第 2 行起才参与计算，其 3x3 模板最远
    # 只够到第 1 行。污染第 1 行会合法地渗进来（这正是腐蚀存在的原因），不是 bug。
    polluted = c.clone()
    polluted[..., 0, :] = 1e6
    polluted[..., -1, :] = -1e6
    polluted[..., :, 0] = 1e6
    polluted[..., :, -1] = -1e6
    loss_polluted = advection_diffusion_loss(
        {'c': polluted, 'vx': vx, 'vy': vy}, mask, diffusivity=0.01, dx=h, dy=h)
    check('S1 掩码外的污染不影响掩码内损失',
          abs(float(loss) - float(loss_polluted)) < 1e-9 * max(abs(float(loss)), 1.0),
          '{:.6e} vs {:.6e}'.format(float(loss), float(loss_polluted)))

    # central_time_derivative 单独验精度：对解析解应收敛到解析时间导数
    yy2, xx2, _ = grid(n)
    dt = 1e-3
    approx = central_time_derivative(
        gaussian_puff(xx2, yy2, 2.0 - dt, 0.1, 0.05, 0.01).view(1, 1, n, n),
        gaussian_puff(xx2, yy2, 2.0 + dt, 0.1, 0.05, 0.01).view(1, 1, n, n), dt)
    exact = gaussian_puff_dcdt(xx2, yy2, 2.0, 0.1, 0.05, 0.01).view(1, 1, n, n)
    rel_t = rms(approx - exact, mask) / max(rms(exact, mask), 1e-30)
    check('S1 中心时间差分逼近解析时间导数',
          rel_t < 1e-4, '相对误差 {:.3e}'.format(rel_t))

    # 梯度必须能回传，否则物理约束根本没参与训练
    c_grad = c.clone().requires_grad_(True)
    advection_diffusion_loss({'c': c_grad, 'vx': vx, 'vy': vy}, mask,
                             diffusivity=0.01, dx=h, dy=h).backward()
    check('S1 梯度可回传到浓度场',
          c_grad.grad is not None and float(c_grad.grad.abs().sum()) > 0)

    try:
        central_time_derivative(c, c, 0.0)
        check('S1 dt<=0 被拒绝', False, '没有抛异常')
    except ValueError:
        check('S1 dt<=0 被拒绝', True)


# ------------------------------------------------- 测试五：.npy 栅格掩码（第 2 步）
def test_raster_mask():
    import tempfile

    from utils.geometry import (mask_coverage, parse_geometry, t_mask_like,
                                t_mask_numpy, validate_geometry)

    # 造一个非矩形的"地铁站平面"：站台横条 + 两条竖向楼梯 + 一个斜角
    src = np.zeros((80, 200), dtype=np.float32)
    src[30:50, :] = 1                      # 站台层
    src[10:30, 40:55] = 1                  # 楼梯 1
    src[10:30, 150:165] = 1                # 楼梯 2
    for i in range(20):                    # 斜向通道，矩形段拼不出来
        src[50 + i // 2, 100 + i] = 1

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'station.npy')
        np.save(path, src)

        geom = parse_geometry(path)
        check('S2 .npy 路径被识别为栅格掩码',
              isinstance(geom, np.ndarray) and geom.shape == (80, 200))

        # 同一份栅格在不同分辨率下覆盖率应基本一致（这正是硬编码像素坐标的病根）
        cov = {n: mask_coverage(n, int(n * 2.5), path) for n in (80, 160, 320)}
        spread = max(cov.values()) - min(cov.values())
        check('S2 栅格掩码在不同分辨率下覆盖同一物理区域',
              spread < 0.01,
              ' '.join('{}:{:.3%}'.format(k, v) for k, v in cov.items()))

        m = t_mask_numpy(160, 400, path)
        check('S2 重采样后仍是纯 0/1',
              set(np.unique(m).tolist()) <= {0.0, 1.0})

        t4 = t_mask_like(torch.zeros(2, 3, 160, 400), path)
        check('S2 可生成与 4D 张量广播的掩码', tuple(t4.shape) == (1, 1, 160, 400))
        t5 = t_mask_like(torch.zeros(2, 3, 4, 160, 400), path)
        check('S2 可生成与 5D 张量广播的掩码', tuple(t5.shape) == (1, 1, 1, 160, 400))

        # 矩形段几何必须仍然工作（不能为了新功能破坏老路径）
        default_cov = mask_coverage(137, 137, None)
        check('S2 默认 T 型几何未被破坏',
              abs(default_cov - 0.449) < 0.01, '{:.3%}'.format(default_cov))

    # 非法输入必须报错，而不是静默给出错误的计算域
    for bad, why in [(np.zeros((10, 10), dtype=np.float32), '全 0'),
                     (np.full((10, 10), 0.5, dtype=np.float32), '非 0/1'),
                     (np.ones((2, 3, 4), dtype=np.float32), '非 2D')]:
        try:
            validate_geometry(bad)
            check('S2 非法栅格被拒绝（{}）'.format(why), False, '没有抛异常')
        except ValueError:
            check('S2 非法栅格被拒绝（{}）'.format(why), True)


# ------------------------------------------------- 测试六：任务分派与参数接线
def test_task_dispatch():
    from args import parse_args
    from utils.physics_loss import combined_physics_loss

    a = parse_args(['--task', 'dispersion', '--channel-order', 'c',
                    '--diffusivity', '3e-4', '--advdiff-weight', '0.2',
                    '--domain-mask', 'foo.npy'], default_ndim=2)
    check('S0 --task / --diffusivity / --advdiff-weight 可解析',
          a.task == 'dispersion' and abs(a.diffusivity - 3e-4) < 1e-12
          and abs(a.advdiff_weight - 0.2) < 1e-12)
    check('S0 --domain-mask 是 --t-geometry 的别名',
          a.t_geometry == 'foo.npy')

    torch.manual_seed(0)
    out = torch.rand(2, 3, 24, 24)
    tgt = torch.rand(2, 3, 24, 24)
    kw = dict(y_min=[0.0] * 3, y_max=[1.0] * 3, dx=0.05, dy=0.05)

    # flow：ns_weight=0 必须真正跳过动量项（而不是算完乘 0）
    _, p_on = combined_physics_loss(out, tgt, ns_weight=0.005, **kw)
    _, p_off = combined_physics_loss(out, tgt, ns_weight=0.0, **kw)
    check('S0 --ns-weight=0 时动量残差为 0',
          p_off['ns'] == 0.0 and p_on['ns'] > 0.0,
          'on={:.3e} off={:.3e}'.format(p_on['ns'], p_off['ns']))
    check('S0 --ns-weight=0 不影响散度项',
          abs(p_on['div'] - p_off['div']) < 1e-9,
          '{:.6e} vs {:.6e}'.format(p_on['div'], p_off['div']))

    # dispersion：advdiff 必须真的被算出来，div/ns 归零
    c_out = torch.rand(2, 1, 24, 24)
    c_tgt = torch.rand(2, 1, 24, 24)
    vel = (torch.full(c_out.shape, 0.3), torch.full(c_out.shape, -0.2))
    _, p_d = combined_physics_loss(
        c_out, c_tgt, channel_order='c', task='dispersion', velocity=vel,
        diffusivity=1e-3, advdiff_weight=0.1, y_min=[0.0], y_max=[1.0],
        dx=0.05, dy=0.05)
    check('S0 dispersion 任务真的算了对流扩散残差',
          p_d['advdiff'] > 0.0, '{:.3e}'.format(p_d['advdiff']))
    check('S0 dispersion 任务不再计算散度/动量项',
          p_d['div'] == 0.0 and p_d['ns'] == 0.0)

    # 不给速度场时必须归零而不是拿输出当速度乱算
    _, p_nov = combined_physics_loss(
        c_out, c_tgt, channel_order='c', task='dispersion', velocity=None,
        advdiff_weight=0.1, y_min=[0.0], y_max=[1.0], dx=0.05, dy=0.05)
    check('S0 dispersion 缺速度场时残差为 0（不静默算错）',
          p_nov['advdiff'] == 0.0)


def test_checkpoint_hp_whitelist():
    """推理侧曾把 t_geometry 丢在白名单外，导致车站模型被 T 型几何裁剪。"""
    from surrogate_predict import DEFAULT_HP
    for key in ('t_geometry', 'no_mask', 'task', 'diffusivity', 'channel_order'):
        check('S0 checkpoint 超参白名单含 {}'.format(key), key in DEFAULT_HP)


def test_erode_reaches_image_border():
    """腐蚀必须把贴着图像边缘的格点也去掉，否则差分模板会伸到图像外。"""
    from utils.physics_loss import erode_mask
    m = torch.ones(1, 1, 9, 9)
    e = erode_mask(m, radius=1)
    check('S0 全 1 掩码腐蚀后不含图像边缘',
          float(e.sum()) == 49.0 and float(e[0, 0, 0].sum()) == 0.0,
          '掩码内像素 {}（期望 49）'.format(int(e.sum())))

    # 计算域贴边的情形（地铁站站台横贯整幅网格）
    m2 = torch.zeros(1, 1, 9, 9)
    m2[..., 3:6, :] = 1.0
    e2 = erode_mask(m2, radius=1)
    check('S0 贴边的计算域两端也被腐蚀掉',
          float(e2[0, 0, 4, 0]) == 0.0 and float(e2[0, 0, 4, -1]) == 0.0
          and float(e2[0, 0, 4, 4]) == 1.0)


def main():
    tests = [test_steady_exponential, test_transient_puff_convergence,
             test_mutation_detected, test_loss_interface, test_raster_mask,
             test_task_dispatch, test_checkpoint_hp_whitelist,
             test_erode_reaches_image_border]
    for fn in tests:
        print('\n--- {} ---'.format(fn.__name__))
        try:
            fn()
        except Exception:
            import traceback
            FAILED.append(fn.__name__)
            print('[FAIL] {} (未捕获异常)'.format(fn.__name__))
            traceback.print_exc()

    print('\n' + '=' * 68)
    print('通过 {}/{}'.format(len(PASSED), len(PASSED) + len(FAILED)))
    if FAILED:
        print('失败项:')
        for name in FAILED:
            print('  - {}'.format(name))
    print('=' * 68)
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
