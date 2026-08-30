"""导出黄金参考数据（Golden Reference），供 Rust 重写版逐元素对照。

    python -m tests.export_golden            # 导出，写到 tests/golden/
    python -m tests.export_golden --verify   # 用同一套计算回验，防止参考数据过期

为什么必须有这个
----------------
重写最大的风险不是写不出来，是**写出来了但算的不是一回事，而且没人发现**。
物理残差、归一化、掩码腐蚀这些地方写错了都不报错，只会让 R² 差一点，你会以为
是超参没调好。「跑起来了、loss 在降」不是验收标准。

所以这里用固定随机种子，把 Python 版每一层的中间结果冻结成参考张量。Rust 版读
同一份数据，逐元素比对。对不上就是错了，没有商量余地。

覆盖顺序刻意从底层往上：**差分算子 → 掩码 → 物理残差 → 网络前向 → 训练一步**。
下层对不上时，上层的偏差没有诊断意义，所以 Rust 侧也应当按这个顺序逐层验收。

产物
----
* ``tests/golden/golden.npz``      —— 所有参考张量
* ``tests/golden/manifest.json``   —— 每个条目的形状/容差/它在验证什么

容差
----
统一 1e-5（f32 的相对量级）。逐元素比较用 ``|a-b| <= atol + rtol*|b|``，
标量损失值同理。浮点累加顺序不同会带来微小漂移，但 1e-5 足够抓出真正的算错。
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dense_ed import DenseED                                  # noqa: E402
from models.dense_ed_3d import DenseED as DenseED3D                   # noqa: E402
from utils.data_common import (denormalize_array, normalize_array,    # noqa: E402
                               split_indices)
from utils.geometry import t_mask_numpy                               # noqa: E402
from utils.misc import RunningMetrics                                 # noqa: E402
from utils.physics_loss import (_conv_replicate, _diff_kernels,       # noqa: E402
                                combined_physics_loss, denormalize,
                                divergence, divergence_loss, erode_mask,
                                grad_xy, laplacian, masked_mean,
                                navier_stokes_loss, physics_report,
                                reconstruction_loss, smoothness_loss)
from utils.transport import (advection_diffusion_loss,                # noqa: E402
                             advection_diffusion_residual,
                             central_time_derivative)

OUT = {}
MANIFEST = []

# 'export' 写参考数据；'verify' 用同一套计算回比，抓「改了生产代码但没重新导出」
MODE = 'export'
REF = None
MISMATCH = []
CHECKED = 0


def put(name, value, what, tol=1e-5):
    """导出模式登记参考条目；验证模式与已有参考逐元素比对。"""
    global CHECKED
    arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    arr = arr.astype(np.float64 if arr.dtype == np.float64 else np.float32)

    if MODE == 'verify':
        CHECKED += 1
        if name not in REF:
            MISMATCH.append((name, '参考数据里没有这一项（导出脚本改了？）'))
            return
        ref = REF[name]
        if ref.shape != arr.shape:
            MISMATCH.append((name, '形状 {} != 参考 {}'.format(arr.shape, ref.shape)))
            return
        # 与 manifest 里声明的判据一致：|a-b| <= tol + tol*|b|
        diff = np.abs(arr.astype(np.float64) - ref.astype(np.float64))
        limit = tol + tol * np.abs(ref.astype(np.float64))
        bad = diff > limit
        if bad.any():
            i = int(np.argmax(diff - limit))
            MISMATCH.append((name, '{} 处超差，最大偏差 {:.3e}（限 {:.3e}）'.format(
                int(bad.sum()), float(diff.flat[i]), float(limit.flat[i]))))
        return

    OUT[name] = arr
    MANIFEST.append({
        'name': name,
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
        'tol': tol,
        'validates': what,
    })


# ============================================================ 第 1 层：差分算子
def layer1_operators():
    """最底层。这几个错了，后面全部没有意义。"""
    g = torch.Generator().manual_seed(1001)
    x = torch.rand(2, 3, 17, 19, generator=g) * 4 - 2
    put('op/input', x, '所有算子测试共用的输入场')

    kx, ky = _diff_kernels(x.device, x.dtype)
    put('op/kernel_x', kx, 'x 方向中心差分核')
    put('op/kernel_y', ky, 'y 方向中心差分核')

    # replicate padding —— Burn 的 Tensor::pad 只支持常量填充，这块要手写，是最高风险项
    put('op/conv_replicate_x', _conv_replicate(x, kx),
        'replicate padding + 3x3 卷积（x 向）。零填充会在边界造出虚假巨梯度，必须逐元素对上')
    put('op/conv_replicate_y', _conv_replicate(x, ky), 'replicate padding（y 向）')

    dx, dy = 0.037, 0.11          # 刻意用各向异性且非整数的间距
    gx, gy = grad_xy(x, dx, dy)
    put('op/grad_x', gx, '一阶中心差分 d/dx，各向异性网格')
    put('op/grad_y', gy, '一阶中心差分 d/dy')
    put('op/laplacian', laplacian(x, dx, dy), '五点 Laplacian，dx≠dy 时分别缩放')

    # 单通道路径与多通道路径在 _conv_replicate 里走的是不同分支，都要覆盖
    x1 = x[:, :1]
    put('op/grad_x_1ch', grad_xy(x1, dx, dy)[0], '单通道分支（groups 卷积不生效的那条路）')


# ============================================================ 第 2 层：掩码
def layer2_masks():
    """掩码腐蚀刚修过一个真实缺陷（图像边缘腐蚀不掉），必须重点覆盖。"""
    cases = {
        'all_ones': torch.ones(1, 1, 9, 9),
        'band_touching_edges': torch.zeros(1, 1, 9, 9),      # 站台层：横贯整幅网格
        'interior_block': torch.zeros(1, 1, 9, 9),
        'single_pixel': torch.zeros(1, 1, 9, 9),
        'checker': torch.zeros(1, 1, 9, 9),
    }
    cases['band_touching_edges'][..., 3:6, :] = 1.0
    cases['interior_block'][..., 2:7, 2:7] = 1.0
    cases['single_pixel'][..., 4, 4] = 1.0
    cases['checker'][..., ::2, ::2] = 1.0

    for name, m in cases.items():
        put('mask/{}/src'.format(name), m, '腐蚀前的掩码')
        put('mask/{}/erode1'.format(name), erode_mask(m, radius=1),
            '腐蚀 1 圈。图像边界必须当作域外——旧实现用 max_pool 的 -inf 补边，'
            '贴边格点腐蚀不掉，差分模板伸到图像外')
        put('mask/{}/erode2'.format(name), erode_mask(m, radius=2), '腐蚀 2 圈')

    # 几何：矩形段（默认 T 型）与 .npy 栅格两条路径
    for h, w in [(65, 65), (137, 137), (96, 240)]:
        put('geom/rect_{}x{}'.format(h, w), t_mask_numpy(h, w, None),
            '默认 T 型几何按归一化比例生成，任意分辨率覆盖同一物理区域')

    raster = np.zeros((40, 100), dtype=np.float32)
    raster[16:24, :] = 1                      # 贴左右边
    raster[6:16, 20:26] = 1
    for i in range(10):
        raster[24 + i // 2, 45 + i] = 1       # 斜向通道，矩形段拼不出来
    put('geom/raster_src', raster, '车站平面栅格掩码原图')
    for h, w in [(40, 100), (80, 200), (20, 50)]:
        put('geom/raster_{}x{}'.format(h, w), t_mask_numpy(h, w, raster),
            '栅格掩码最近邻重采样（放大/等尺/缩小三种情况）')

    # masked_mean：分母必须是掩码内像素数，不是全图
    g = torch.Generator().manual_seed(1002)
    v = torch.rand(2, 1, 9, 9, generator=g)
    put('mask/masked_mean_input', v, 'masked_mean 输入')
    put('mask/masked_mean_banded', masked_mean(v, cases['band_touching_edges']),
        '掩码内均值，分母是掩码内像素数')
    put('mask/masked_mean_none', masked_mean(v, None), 'mask=None 时退化成普通 mean')


# ============================================================ 第 3 层：物理残差
def layer3_physics():
    g = torch.Generator().manual_seed(1003)
    fields4 = torch.rand(2, 4, 24, 28, generator=g) * 2 - 1
    put('phys/fields_pvvt', fields4, '(P,Vx,Vy,T) 四通道输入场')

    dx, dy = 0.021, 0.043
    mask = torch.zeros(1, 1, 24, 28)
    mask[..., 4:20, :] = 1.0                  # 刻意贴左右边界
    put('phys/mask', mask, '计算域掩码（左右贴边）')
    interior = erode_mask(mask, radius=1)
    put('phys/interior', interior, '腐蚀后的物理残差评估域')

    f = {'p': fields4[:, 0:1], 'vx': fields4[:, 1:2], 'vy': fields4[:, 2:3]}
    put('phys/divergence_field', divergence(f['vx'], f['vy'], dx, dy), 'div(v) 逐点场')
    put('phys/div_loss', divergence_loss(f, interior, dx, dy), '连续性方程残差均方')
    put('phys/ns_loss', navier_stokes_loss(f, interior, 1.3, 7e-3, dx, dy),
        '稳态 N-S 动量残差（rho=1.3, nu=7e-3）')
    put('phys/smooth_loss', smoothness_loss(fields4, interior, dx, dy), '平滑项')

    tgt = torch.rand(2, 4, 24, 28, generator=g)
    put('phys/target', tgt, '重建损失的目标场')
    put('phys/recon_weighted', reconstruction_loss(fields4, tgt, mask, True),
        '加权重建损失 0.6*MSE + 0.4*L1，分母是掩码内像素数')
    put('phys/recon_plain', reconstruction_loss(fields4, tgt, mask, False), 'MSE + L1')

    # 反归一化：物理项必须在有量纲场上算
    y_min = np.array([-50.0, -2.0, -0.5, 290.0], dtype=np.float32)
    y_max = np.array([150.0, 3.0, 0.8, 305.0], dtype=np.float32)
    put('phys/y_min', y_min, '逐通道最小值')
    put('phys/y_max', y_max, '逐通道最大值')
    put('phys/denormalized', denormalize(fields4, y_min, y_max),
        '[0,1] -> 有量纲。逐通道缩放，Vx/Vy 尺度不同，在归一化空间求 div 是错的')

    # 组合损失（flow）：含 warmup 斜坡与量级归一化
    total, parts = combined_physics_loss(
        fields4, tgt, mask=mask, epoch=37, total_epochs=200,
        div_weight=0.05, ns_weight=0.005, smooth_weight=0.005,
        channel_order='pvvt', y_min=y_min, y_max=y_max, rho=1.3, nu=7e-3,
        dx=dx, dy=dy, physics_warmup=0.1,
        scales={'div': 12.0, 'ns': 3400.0, 'smooth': 7.0})
    put('phys/combined_flow_total', total, 'flow 任务组合损失总值')
    for k in ('base', 'div', 'ns', 'smooth', 'w_div', 'w_ns', 'w_smooth'):
        put('phys/combined_flow_{}'.format(k), np.float32(parts[k]),
            'flow 组合损失分项 {}（含 epoch 斜坡）'.format(k))

    # ns_weight=0 必须真正跳过，而不是算完乘 0
    _, parts0 = combined_physics_loss(
        fields4, tgt, mask=mask, epoch=37, total_epochs=200, ns_weight=0.0,
        channel_order='pvvt', y_min=y_min, y_max=y_max, dx=dx, dy=dy)
    put('phys/nsweight0_ns', np.float32(parts0['ns']), '--ns-weight=0 时动量残差应为 0')
    put('phys/nsweight0_div', np.float32(parts0['div']), '关掉动量项不影响散度项')

    rep = physics_report(fields4, mask, 'pvvt', y_min, y_max, 1.3, 7e-3, dx, dy)
    for k, v in rep.items():
        if v == v:                              # 跳过 nan
            put('phys/report_{}'.format(k), np.float32(v), '诊断报告 {}'.format(k))


# ============================================================ 第 4 层：对流扩散
def layer4_transport():
    g = torch.Generator().manual_seed(1004)
    n = 32
    c = torch.rand(2, 1, n, n, generator=g)
    vx = torch.rand(2, 1, n, n, generator=g) * 0.4 - 0.2
    vy = torch.rand(2, 1, n, n, generator=g) * 0.4 - 0.2
    put('trans/c', c, '浓度场')
    put('trans/vx', vx, '速度 x（来自输入通道，不是网络输出）')
    put('trans/vy', vy, '速度 y')

    dx, dy, D = 0.019, 0.031, 3.2e-3
    mask = torch.ones(1, 1, n, n)
    interior = erode_mask(mask, radius=1)

    put('trans/residual_steady',
        advection_diffusion_residual(c, vx, vy, D, dx, dy),
        '稳态残差 u·∇C - D∇²C')

    dcdt = torch.rand(2, 1, n, n, generator=g) * 0.1
    put('trans/dcdt', dcdt, '时间导数输入')
    put('trans/residual_transient',
        advection_diffusion_residual(c, vx, vy, D, dx, dy, dcdt=dcdt),
        '瞬态残差，含 ∂C/∂t')

    src = torch.rand(2, 1, n, n, generator=g) * 0.05
    put('trans/source', src, '源项 S')
    put('trans/residual_with_source',
        advection_diffusion_residual(c, vx, vy, D, dx, dy, dcdt=dcdt, source=src),
        '含源项的完整残差')

    put('trans/loss', advection_diffusion_loss(
        {'c': c, 'vx': vx, 'vy': vy}, interior, D, dx, dy, dcdt=dcdt),
        '对流扩散残差均方')

    put('trans/central_dt', central_time_derivative(c, c * 1.1, 0.25),
        '中心时间差分，分母是 2Δt')

    # dispersion 任务的组合损失
    tgt = torch.rand(2, 1, n, n, generator=g)
    total, parts = combined_physics_loss(
        c, tgt, mask=mask, epoch=90, total_epochs=200, channel_order='c',
        task='dispersion', velocity=(vx, vy), diffusivity=D,
        advdiff_weight=0.08, y_min=np.array([0.0], np.float32),
        y_max=np.array([5.0], np.float32), dx=dx, dy=dy)
    put('trans/combined_total', total, 'dispersion 组合损失总值')
    for k in ('base', 'advdiff', 'smooth', 'div', 'ns'):
        put('trans/combined_{}'.format(k), np.float32(parts[k]),
            'dispersion 分项 {}（div/ns 应为 0）'.format(k))


# ============================================================ 第 5 层：数据处理
def layer5_data():
    g = np.random.RandomState(1005)
    y = (g.rand(6, 3, 8, 10).astype(np.float32) * 20 - 5)
    y_min = y.min(axis=(0, 2, 3))
    y_max = y.max(axis=(0, 2, 3))
    put('data/raw', y, '原始有量纲场')
    put('data/y_min', y_min, '逐通道最小值（全量扫描，不是只取前 N 个）')
    put('data/y_max', y_max, '逐通道最大值')
    norm = normalize_array(y, y_min, y_max)
    put('data/normalized', norm, '逐通道 min-max 归一化到 [0,1]，不 clamp')
    put('data/roundtrip', denormalize_array(norm, y_min, y_max), '反归一化应还原原值')

    for n, vs, seed in [(100, 0.2, 1), (37, 0.3, 7), (5, 0.5, 3)]:
        tr, va = split_indices(n, vs, seed)
        put('data/split_{}_{}_{}_train'.format(n, vs, seed), tr.astype(np.float32),
            '训练下标（固定 seed 可复现，与验证集互不重叠）')
        put('data/split_{}_{}_{}_val'.format(n, vs, seed), va.astype(np.float32),
            '验证下标')

    # R² / RMSE 累计器
    gt = torch.Generator().manual_seed(1006)
    out = torch.rand(4, 3, 12, 12, generator=gt)
    tgt = torch.rand(4, 3, 12, 12, generator=gt)
    msk = torch.zeros(1, 1, 12, 12)
    msk[..., 2:10, 2:10] = 1.0
    put('metrics/out', out, '指标累计器输入')
    put('metrics/tgt', tgt, '指标累计器目标')
    put('metrics/mask', msk, '指标只统计掩码内像素')
    m = RunningMetrics(3)
    m.update(out, tgt, msk)
    put('metrics/r2', np.float32(m.r2()), '整体 R²')
    put('metrics/rmse', np.float32(m.rmse()), '整体 RMSE')
    put('metrics/r2_per_channel', np.asarray(m.r2_per_channel(), np.float32), '分通道 R²')


# ============================================================ 第 6 层：网络前向
def _fixed_init(model, seed):
    """用固定序列填充所有参数——随机初始化在两个框架间不可能一致，
    所以权重必须由外部给定，Rust 侧读同一份。"""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_((torch.rand(p.shape, generator=g) - 0.5) * 0.2)
        for name, buf in model.named_buffers():
            if 'running_mean' in name:
                buf.zero_()
            elif 'running_var' in name:
                buf.fill_(1.0)
    return model


def _dump_params(prefix, model, what):
    for name, p in model.named_parameters():
        put('{}/param/{}'.format(prefix, name), p, what)


def layer6_model():
    configs = [
        ('m3blk', dict(in_channels=2, out_channels=3, blocks=[2, 2, 2],
                       growth_rate=4, init_features=8), (2, 2, 32, 32)),
        ('m5blk', dict(in_channels=3, out_channels=4, blocks=[2, 2, 2, 2, 2],
                       growth_rate=4, init_features=8), (2, 3, 40, 40)),
        ('m_odd', dict(in_channels=1, out_channels=1, blocks=[2, 2, 2],
                       growth_rate=4, init_features=8), (1, 1, 33, 29)),
    ]
    for tag, kw, shape in configs:
        model = DenseED(verbose=False, output_size=shape[-2:], **kw)
        _fixed_init(model, 2000 + len(tag))
        model.eval()
        _dump_params(tag, model, '{} 的固定权重，Rust 侧必须读这份而不是自己初始化'.format(tag))
        g = torch.Generator().manual_seed(3000)
        x = torch.rand(*shape, generator=g)
        put('{}/input'.format(tag), x, '{} 前向输入'.format(tag))
        with torch.no_grad():
            put('{}/output'.format(tag), model(x),
                '{} 前向输出。逐元素对上才说明 DenseBlock/Transition/skip/'
                'Transformer 瓶颈/位置编码全部一致'.format(tag))

    # 3D 路径
    m3d = DenseED3D(in_channels=2, out_channels=3, blocks=[2, 2, 2],
                    growth_rate=4, init_features=8, verbose=False,
                    output_size=(2, 24, 24))
    _fixed_init(m3d, 2100)
    m3d.eval()
    _dump_params('m3d', m3d, '3D 模型固定权重')
    g = torch.Generator().manual_seed(3100)
    x3 = torch.rand(1, 2, 2, 24, 24, generator=g)
    put('m3d/input', x3, '3D 前向输入 (B,C,D,H,W)')
    with torch.no_grad():
        put('m3d/output', m3d(x3), '3D 前向输出')


# ============================================================ 第 7 层：训练一步
def layer7_train_step():
    """反向传播 + Adam 更新。前六层都对上了，这一层才有意义。"""
    model = DenseED(in_channels=2, out_channels=3, blocks=[2, 2, 2],
                    growth_rate=4, init_features=8, verbose=False,
                    output_size=(32, 32))
    _fixed_init(model, 2200)
    model.train()
    _dump_params('step/before', model, '训练前参数')

    g = torch.Generator().manual_seed(3200)
    x = torch.rand(2, 2, 32, 32, generator=g)
    y = torch.rand(2, 3, 32, 32, generator=g)
    put('step/input', x, '训练步输入')
    put('step/target', y, '训练步目标')

    mask = torch.zeros(1, 1, 32, 32)
    mask[..., 6:26, :] = 1.0
    put('step/mask', mask, '训练步掩码')

    y_min = np.array([-10.0, -1.0, -0.4], np.float32)
    y_max = np.array([30.0, 2.0, 0.6], np.float32)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=5e-4)
    opt.zero_grad(set_to_none=True)
    out = model(x)
    put('step/forward', out, '训练模式下的前向输出（BatchNorm 走 batch 统计）')
    loss, parts = combined_physics_loss(
        out, y, mask=mask, epoch=50, total_epochs=200,
        channel_order='pvv', y_min=y_min, y_max=y_max,
        dx=0.03, dy=0.05, physics_warmup=0.1)
    put('step/loss', loss, '训练步总损失')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for name, p in model.named_parameters():
        if p.grad is not None:
            put('step/grad/{}'.format(name), p.grad, '梯度裁剪后的梯度')
    opt.step()
    _dump_params('step/after', model, 'Adam 一步之后的参数')


LAYERS = (layer1_operators, layer2_masks, layer3_physics, layer4_transport,
          layer5_data, layer6_model, layer7_train_step)


def main():
    global MODE, REF
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')
    npz_path = os.path.join(out_dir, 'golden.npz')
    torch.use_deterministic_algorithms(False)

    if '--verify' in sys.argv:
        MODE = 'verify'
        if not os.path.exists(npz_path):
            print('没有参考数据，先跑 python -m tests.export_golden')
            return 1
        REF = dict(np.load(npz_path))
        for fn in LAYERS:
            n0 = len(MISMATCH)
            fn()
            bad = len(MISMATCH) - n0
            print('  {:24s} {}'.format(fn.__name__, 'OK' if bad == 0 else
                                       '{} 项不符'.format(bad)))
        print()
        print('=' * 68)
        if MISMATCH:
            print('比对 {} 项，{} 项不符：'.format(CHECKED, len(MISMATCH)))
            for name, why in MISMATCH[:20]:
                print('  - {}: {}'.format(name, why))
            if len(MISMATCH) > 20:
                print('  ...另有 {} 项'.format(len(MISMATCH) - 20))
            print()
            print('生产代码改了就要重新导出：python -m tests.export_golden')
            print('=' * 68)
            return 1
        print('比对 {} 项全部一致，参考数据与当前代码同步'.format(CHECKED))
        print('=' * 68)
        return 0

    os.makedirs(out_dir, exist_ok=True)
    for fn in LAYERS:
        n0 = len(MANIFEST)
        fn()
        print('  {:24s} 导出 {} 项'.format(fn.__name__, len(MANIFEST) - n0))

    np.savez_compressed(npz_path, **OUT)

    manifest = {
        'generated_by': 'tests/export_golden.py',
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'default_tol': 1e-5,
        'comparison': '|a-b| <= tol + tol*|b|',
        'layer_order': ['op', 'mask/geom', 'phys', 'trans', 'data/metrics',
                        'model', 'step'],
        'note': '按层从底往上验收：下层对不上时，上层的偏差没有诊断意义',
        'entries': MANIFEST,
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    size = os.path.getsize(npz_path) / 1024
    print('\n共 {} 项参考数据 -> {} ({:.1f} KB)'.format(len(MANIFEST), npz_path, size))
    return 0


if __name__ == '__main__':
    sys.exit(main())
