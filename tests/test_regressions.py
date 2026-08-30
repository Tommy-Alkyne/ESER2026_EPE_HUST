"""针对代码审查报告中每一条问题的回归测试。

    python -m tests.test_regressions        # 在项目根目录运行
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = []


def check(name, condition, detail=''):
    RESULTS.append((name, bool(condition), detail))
    flag = 'PASS' if condition else 'FAIL'
    print('[{}] {}{}'.format(flag, name, ('  -- ' + detail) if detail else ''))
    return bool(condition)


# ---------------------------------------------------------------- P0-1 通道顺序
def test_channel_order():
    from utils.physics_loss import resolve_channels
    pvv = resolve_channels('pvv', 3)
    vvp = resolve_channels('vvp', 3)
    check('P0-1 通道顺序 pvv 映射正确',
          pvv == {'p': 0, 'vx': 1, 'vy': 2}, str(pvv))
    check('P0-1 通道顺序 vvp 映射正确',
          vvp == {'vx': 0, 'vy': 1, 'p': 2}, str(vvp))
    check('P0-1 通道数不足时自动降级',
          resolve_channels('pvv', 2) == {'p': 0, 'vx': 1})
    try:
        resolve_channels('nonsense', 3)
        check('P0-1 非法顺序会报错', False)
    except ValueError:
        check('P0-1 非法顺序会报错', True)


# ------------------------------------------------------------ P0-2 有量纲物理
def test_physical_units():
    from utils.physics_loss import divergence, denormalize

    h = w = 33
    ys, xs = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w),
                            indexing='ij')
    # 构造解析无散场 v = (sin(2*pi*x)*cos(2*pi*y), -cos(2*pi*x)*sin(2*pi*y))
    vx = torch.sin(2 * np.pi * xs) * torch.cos(2 * np.pi * ys)
    vy = -torch.cos(2 * np.pi * xs) * torch.sin(2 * np.pi * ys)
    dx = dy = 1.0 / (h - 1)

    div = divergence(vx[None, None], vy[None, None], dx, dy)
    inner = div[..., 2:-2, 2:-2]
    check('P0-2 解析无散场的离散散度接近 0',
          float(inner.abs().max()) < 0.6,
          'max|div|={:.4f}'.format(float(inner.abs().max())))

    # 逐通道归一化后，两个分量被不同因子缩放 -> 归一化空间的散度不再为零
    scale_x, scale_y = 1.0, 10.0
    vxn, vyn = vx / scale_x, vy / scale_y
    div_norm = divergence(vxn[None, None], vyn[None, None], dx, dy)
    check('P0-2 归一化空间的散度确实是错的（这正是旧版在做的事）',
          float(div_norm[..., 2:-2, 2:-2].abs().max()) > 1.0,
          'max|div_norm|={:.3f}'.format(float(div_norm[..., 2:-2, 2:-2].abs().max())))

    # 反归一化能把物理量还原回去
    y_min = torch.tensor([-1.0, -2.0, -3.0])
    y_max = torch.tensor([1.0, 2.0, 3.0])
    normed = torch.rand(2, 3, 8, 8)
    back = denormalize(normed, y_min, y_max)
    expect = normed * (y_max - y_min).view(1, 3, 1, 1) + y_min.view(1, 3, 1, 1)
    check('P0-2 反归一化数值正确',
          torch.allclose(back, expect, atol=1e-6))


# ------------------------------------------------------- P0-3/4 划分与统计
def test_split_and_stats():
    from utils.data_common import compute_output_stats, split_indices
    import h5py

    train_idx, val_idx = split_indices(100, 0.2, seed=1)
    overlap = set(train_idx.tolist()) & set(val_idx.tolist())
    check('P0-3 训练/验证集互不重叠', not overlap)
    check('P0-3 划分比例正确',
          len(val_idx) == 20 and len(train_idx) == 80,
          '{} / {}'.format(len(train_idx), len(val_idx)))
    t2, v2 = split_indices(100, 0.2, seed=1)
    check('P0-3 划分可复现（同 seed 同结果）',
          np.array_equal(train_idx, t2) and np.array_equal(val_idx, v2))

    # 全量统计：把极值故意放在最后一个样本，验证不会被"只取前 N 个"漏掉
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'stats.hdf5')
        data = np.random.RandomState(0).rand(1500, 2, 4, 4).astype('f4')
        data[-1, 0] = 999.0        # 最后一个样本才出现的极值
        data[-1, 1] = -888.0
        with h5py.File(path, 'w') as f:
            f.create_dataset('input', data=np.zeros((1500, 1, 4, 4), 'f4'))
            f.create_dataset('output', data=data)
        stats = compute_output_stats(path, use_cache=False, verbose=False)
        check('P0-4 全量扫描能抓到第 1500 个样本里的极值',
              abs(float(stats['y_max'][0]) - 999.0) < 1e-3
              and abs(float(stats['y_min'][1]) + 888.0) < 1e-3,
              'max0={} min1={}'.format(stats['y_max'][0], stats['y_min'][1]))

        # 归一化不再 clamp：极值样本应恰好落在 0/1，且没有被削平的中间值
        from utils.load_data import CFDHDF5Dataset
        ds = CFDHDF5Dataset(path, stats=stats)
        _, y_last = ds[1499]
        check('P0-4 极值样本归一化后落在边界而非被截断',
              abs(float(y_last[0].max()) - 1.0) < 1e-4,
              'max={:.6f}'.format(float(y_last[0].max())))
        ds.close()      # 不关句柄的话 Windows 上删临时目录会抛 WinError 32


# -------------------------------------------------------- P0-5 5D 张量重排
def test_5d_reshape():
    from utils.physics_loss import flatten_depth, unflatten_depth
    b, c, d, h, w = 2, 3, 4, 5, 6
    x = torch.zeros(b, c, d, h, w)
    for ch in range(c):
        x[:, ch] = ch + 1
    flat, (bb, dd) = flatten_depth(x)
    per_slice = [[float(flat[i, ch].mean()) for ch in range(c)]
                 for i in range(flat.shape[0])]
    expected = [float(ch + 1) for ch in range(c)]
    check('P0-5 5D->4D 不再搅乱通道与深度',
          all(row == expected for row in per_slice),
          'slice0={}'.format(per_slice[0]))
    check('P0-5 往返变换可逆',
          torch.equal(unflatten_depth(flat, bb, dd), x))


# -------------------------------------------------- P1-6/7 loader 与推理
def test_loader_signature():
    import inspect
    from utils.load_data import load_data as ld2
    from utils.load_data_3d import load_data as ld3
    sig2 = set(inspect.signature(ld2).parameters)
    sig3 = set(inspect.signature(ld3).parameters)
    needed = {'stats', 'input_key', 'prefer_sensor_input',
              'align_input_to_output', 'shuff'}
    check('P1-6 2D loader 接受推理脚本用到的全部参数',
          needed <= sig2, str(sorted(needed - sig2)))
    check('P1-6 3D loader 接受同一组参数（2D 签名是 3D 的子集）',
          needed <= sig3 and sig2 <= sig3,
          '2D 独有: {} · 3D 独有: {}'.format(sorted(sig2 - sig3),
                                            sorted(sig3 - sig2)))


def test_denormalize_roundtrip():
    from utils.data_common import denormalize_array, normalize_array
    y = np.random.RandomState(0).randn(5, 3, 7, 7).astype('f4') * [100, 2, 0.3][0]
    y_min = y.min(axis=(0, 2, 3))
    y_max = y.max(axis=(0, 2, 3))
    back = denormalize_array(normalize_array(y, y_min, y_max), y_min, y_max)
    check('P1-7 归一化/反归一化往返一致',
          np.allclose(back, y, atol=1e-3),
          'max err={:.2e}'.format(float(np.abs(back - y).max())))


# ------------------------------------------------------------ P1-8 绘图
def test_plotting():
    from utils.plot import plot_prediction
    with tempfile.TemporaryDirectory() as tmp:
        for c in (1, 2, 3, 5):
            for fn in ('imshow', 'contourf'):
                try:
                    plot_prediction(tmp, np.random.rand(c, 20, 20),
                                    np.random.rand(c, 20, 20), 1, 0, plot_fn=fn)
                    passed, detail = True, ''
                except Exception as exc:
                    passed, detail = False, '{}: {}'.format(type(exc).__name__, exc)
                check('P1-8 绘图 C={} / {}'.format(c, fn), passed, detail)
        # 4D (C, D, H, W) 输入
        try:
            plot_prediction(tmp, np.random.rand(3, 4, 16, 16),
                            np.random.rand(3, 4, 16, 16), 1, 1)
            check('P1-8 绘图支持 (C,D,H,W)', True)
        except Exception as exc:
            check('P1-8 绘图支持 (C,D,H,W)', False, str(exc))


# --------------------------------------------------- P1-9 模型输出尺寸
def test_model_output_size():
    from models.dense_ed import DenseED
    from models.dense_ed_3d import DenseED3D
    net = DenseED(1, 3, blocks=[3, 6, 3], growth_rate=8, init_features=16,
                  verbose=False)
    for size in (64, 65, 128, 137, 200):
        out = net(torch.zeros(1, 1, size, size))
        check('P1-9 2D {0}x{0} 输入输出等尺寸'.format(size),
              tuple(out.shape[-2:]) == (size, size), str(tuple(out.shape)))
    forced = DenseED(1, 2, blocks=[3, 6, 3], growth_rate=8, init_features=16,
                     output_size=(137, 137), verbose=False)
    check('P1-9 output_size 强制生效',
          tuple(forced(torch.zeros(1, 1, 64, 64)).shape[-2:]) == (137, 137))
    net3 = DenseED3D(1, 3, blocks=[3, 6, 3], growth_rate=4, init_features=8,
                     verbose=False)
    out3 = net3(torch.zeros(1, 1, 3, 40, 40))
    check('P1-9 3D 输入输出等尺寸',
          tuple(out3.shape[-3:]) == (3, 40, 40), str(tuple(out3.shape)))


# ------------------------------------------------ P1-11/12 参数解析
def test_args_parsing():
    from args import parse_args
    a = parse_args(['--ndim', '2', '--blocks', '[3,6,3]'], default_ndim=2)
    check('P1-11 2D 分支认识 --ndim', a.ndim == 2)
    check('P1-12 --blocks 解析成整数列表', a.blocks == [3, 6, 3], str(a.blocks))
    for text, want in [('3,6,3', [3, 6, 3]), ('[5, 10, 5]', [5, 10, 5])]:
        got = parse_args(['--blocks', text]).blocks
        check('P1-12 --blocks "{}"'.format(text), got == want, str(got))
    b = parse_args(['--use-3d'])
    check('P1-11 --use-3d 生效', b.ndim == 3)
    c = parse_args(['--epochs', '10', '--ckpt-freq', '3'])
    check('P1-11 epochs 不整除 ckpt-freq 不再 assert 崩溃', c.epochs == 10)


def test_args_no_side_effects():
    """import args 不应该创建目录 / 写文件。"""
    code = ('import sys, os; sys.argv=["x"]; '
            'before=set(os.listdir(".")); import args; '
            'after=set(os.listdir(".")); '
            'print("SIDE_EFFECT" if after-before else "CLEAN")')
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, PYTHONPATH=os.getcwd())
        out = subprocess.run([sys.executable, '-c', code], cwd=tmp, env=env,
                             capture_output=True, text=True)
        check('P1-11 import args 无文件系统副作用',
              'CLEAN' in out.stdout, out.stdout.strip() + out.stderr[-200:])


# ------------------------------------------------------- P2-13 掩码几何
def test_mask_geometry():
    from utils.geometry import mask_coverage, t_mask_numpy
    ref = mask_coverage(137, 137)
    check('P2-13 137x137 覆盖率与原设计一致',
          abs(ref - 0.449) < 0.005, '{:.3f}'.format(ref))
    for hw in (65, 137, 257, 512):
        cov = mask_coverage(hw, hw)
        check('P2-13 {0}x{0} 覆盖率与 137 网格一致（旧版分别是 63.8%/44.9%/12.8%）'.format(hw),
              abs(cov - ref) < 0.01, '{:.3f} vs {:.3f}'.format(cov, ref))
    custom = t_mask_numpy(100, 100, '{"a": [0.0, 0.5, 0.0, 0.5]}')
    check('P2-13 自定义几何生效', abs(custom.mean() - 0.25) < 0.01,
          '{:.3f}'.format(float(custom.mean())))


# -------------------------------------------------- P2-14 位置编码
def test_positional_encoding():
    from models.attention import TransformerBottleneck
    torch.manual_seed(0)
    block = TransformerBottleneck(channels=8, num_heads=2).eval()
    x = torch.randn(1, 8, 6, 6)
    with torch.no_grad():
        out_a = block(x)
        # 打乱空间位置后再送进去；有位置编码时结果不应只是同样的置换
        perm = torch.randperm(36)
        x_shuf = x.flatten(2)[:, :, perm].view_as(x)
        out_b = block(x_shuf)
    out_a_shuf = out_a.flatten(2)[:, :, perm]
    diff = float((out_a_shuf - out_b.flatten(2)).abs().max())
    check('P2-14 注意力块不再对空间排列等变（位置编码生效）',
          diff > 1e-4, 'max diff={:.3e}'.format(diff))

    plain = TransformerBottleneck(channels=8, num_heads=2,
                                  use_positional_encoding=False).eval()
    with torch.no_grad():
        pa = plain(x).flatten(2)[:, :, perm]
        pb = plain(x.flatten(2)[:, :, perm].view_as(x)).flatten(2)
    check('P2-14 关闭位置编码后确实退化为等变（对照组）',
          float((pa - pb).abs().max()) < 1e-4)


# -------------------------------------------------- P2-15/16 架构
def test_architecture():
    from models.dense_ed import DenseED
    from models.dense_ed_3d import DenseED3D
    import torch.nn as nn

    net = DenseED(1, 3, blocks=[3, 6, 3], growth_rate=8, init_features=16,
                  verbose=False)
    check('P2-15 2D 模型有 skip connection（存在 trans_ups 与多分支 forward）',
          len(net.trans_ups) > 0 and hasattr(net, 'enc_blocks'),
          'trans_ups={}'.format(len(net.trans_ups)))
    # 通道数能对上说明确实拼接了 skip
    dec_in = net.dec_blocks[1].denselayer1.norm1.num_features
    check('P2-15 解码器输入通道 = 上采样通道 + skip 通道',
          dec_in > net.skip_channels[-1], 'dec_in={} skip={}'
          .format(dec_in, net.skip_channels[-1]))

    net3 = DenseED3D(1, 3, blocks=[3, 6, 3], growth_rate=4, init_features=8,
                     verbose=False)
    leaky = [n for n, m in net3.named_modules() if isinstance(m, nn.LeakyReLU)]
    silu = [n for n, m in net3.named_modules() if isinstance(m, nn.SiLU)]
    check('P2-16 3D 模型不再使用 LeakyReLU', not leaky, str(leaky[:3]))
    check('P2-16 3D 模型使用 SiLU', len(silu) > 0, '{} 层'.format(len(silu)))


# -------------------------------------------------- P2-17 掩码化损失
def test_masked_loss():
    from utils.physics_loss import combined_physics_loss, masked_mean

    x = torch.ones(1, 1, 10, 10)
    mask = torch.zeros(1, 1, 10, 10)
    mask[..., :5, :] = 1
    check('P2-17 masked_mean 分母是掩码内像素数',
          abs(float(masked_mean(x, mask)) - 1.0) < 1e-6,
          '{:.4f}（旧版会得到 0.5）'.format(float(masked_mean(x, mask))))

    # 掩码外的差异不应影响损失
    out = torch.rand(2, 3, 12, 12)
    tgt = torch.rand(2, 3, 12, 12)
    m = torch.zeros(2, 3, 12, 12)
    m[..., :6, :] = 1
    l1, _ = combined_physics_loss(out, tgt, mask=m, epoch=100, total_epochs=100)
    out2 = out.clone()
    out2[..., 7:, :] += 5.0   # 只改掩码外区域（留 1 格给差分模板的腐蚀带）
    l2, _ = combined_physics_loss(out2, tgt, mask=m, epoch=100, total_epochs=100)
    rel = abs(float(l1) - float(l2)) / max(abs(float(l1)), 1e-8)
    check('P2-17 掩码外的改动不影响数据项与物理项',
          rel < 0.05, '相对变化 {:.2%}'.format(rel))


# -------------------------------------------------- P2-18 权重可配置
def test_weights_configurable():
    from args import parse_args
    a = parse_args(['--div-weight', '0.5', '--ns-weight', '0.02',
                    '--smooth-weight', '0.001', '--physics-warmup', '0.3'])
    check('P2-18 物理权重来自命令行而非硬编码',
          (a.div_weight, a.ns_weight, a.smooth_weight, a.physics_warmup)
          == (0.5, 0.02, 0.001, 0.3))
    check('P2-18 lambda-l1/lambda-grad 死参数已移除',
          not hasattr(a, 'lambda_l1') and not hasattr(a, 'lambda_grad'))


# -------------------------------------------------- P2-20 指标与清理
def test_metrics():
    from utils.misc import RunningMetrics
    m = RunningMetrics(2)
    target = torch.zeros(4, 2, 8, 8)
    target[:, 0] = torch.randn(4, 8, 8)
    target[:, 1] = torch.randn(4, 8, 8)
    output = target.clone()
    output[:, 1] += 5.0                        # 第二个通道故意预测得很差
    m.update(output, target)
    per_channel = m.r2_per_channel()
    check('P2-20 分通道 R² 能分辨好坏通道',
          per_channel[0] > 0.99 and per_channel[1] < 0,
          str(np.round(per_channel, 3).tolist()))

    m2 = RunningMetrics(1)
    t = torch.randn(2, 1, 6, 6)
    mask = torch.zeros(2, 1, 6, 6)
    mask[..., :3, :] = 1
    bad = t.clone()
    bad[..., 3:, :] += 100.0                   # 只在掩码外制造巨大误差
    m2.update(bad, t, mask)
    check('P2-20 R²/RMSE 只统计掩码内像素',
          m2.rmse() < 1e-5, 'rmse={:.3e}'.format(m2.rmse()))


def test_dead_code_removed():
    import utils.misc as misc
    gone = [n for n in ('log_sum_exp', 'parameters_to_vector',
                        'vector_to_parameters', 'logger') if hasattr(misc, n)]
    check('P2-20 misc.py 的上游死代码已删除', not gone, str(gone))


def test_physics_scaler():
    from utils.physics_loss import PhysicsScaler
    s = PhysicsScaler(momentum=0.5)
    for _ in range(50):
        s.update({'div': 1000.0, 'ns': 2e6, 'smooth': 300.0})
    cur = s.current()
    check('P2-新 物理项量级归一化生效',
          abs(cur['div'] - 1000.0) < 1.0 and abs(cur['ns'] - 2e6) < 1e4,
          str({k: round(v, 2) for k, v in cur.items()}))
    off = PhysicsScaler(enabled=False)
    off.update({'div': 5.0})
    check('P2-新 可关闭量级归一化', off.current() == {})


def test_gui_importable():
    """GUI 用 sys.executable 而不是写死的 'python'。"""
    src = open('gui_trainer.py', encoding='utf-8').read()
    check('P2-20 GUI 使用 sys.executable',
          'sys.executable' in src and '[sys.executable' in src
          and '["python"' not in src and "['python'" not in src)
    check('P2-20 GUI 默认实验名是 ASCII',
          "'run01'" in src or '"run01"' in src)


def main():
    tests = [test_channel_order, test_physical_units, test_split_and_stats,
             test_5d_reshape, test_loader_signature, test_denormalize_roundtrip,
             test_plotting, test_model_output_size, test_args_parsing,
             test_args_no_side_effects, test_mask_geometry,
             test_positional_encoding, test_architecture, test_masked_loss,
             test_weights_configurable, test_metrics, test_dead_code_removed,
             test_physics_scaler, test_gui_importable]
    for fn in tests:
        print('\n--- {} ---'.format(fn.__name__))
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            check(fn.__name__ + ' (未捕获异常)', False, str(exc))

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print('\n' + '=' * 68)
    print('通过 {}/{}'.format(passed, total))
    if passed != total:
        print('失败项:')
        for name, ok, detail in RESULTS:
            if not ok:
                print('  - {}  {}'.format(name, detail))
    print('=' * 68)
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
