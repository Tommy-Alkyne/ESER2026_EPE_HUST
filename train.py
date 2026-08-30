"""T 型管流场 PINN 代理模型 —— 主训练脚本。

用法
----
    # 单文件自动切 20% 做验证（推荐）
    python train.py --train-file dataset/p_v_256.hdf5 --epochs 200 --batch-size 8

    # 显式的独立测试集
    python train.py --train-file dataset/train.hdf5 --test-file dataset/test.hdf5

    # 3D
    python train.py --ndim 3 --train-file dataset/10243d.hdf5

关键行为变更（相对旧版）
------------------------
* 训练集与测试集是同一个文件时，**自动按 --val-split 切出互不重叠的验证集**，
  不再拿训练集当测试集报 R²。
* 测试集**复用训练集的归一化统计量**，两边标度一致，R² 才有可比性。
* 物理损失在**有量纲**的场上计算，通道语义由 `--channel-order` 显式指定。
* R² / RMSE **分通道**统计；学习率调度依据**验证** RMSE。
* checkpoint 保存成自描述的 dict（含归一化统计量与全部超参），
  `surrogate_predict.py` 据此还原物理量纲。
"""

import json
import os
import sys
from time import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ------------------------------------------------------------------ 参数解析
# 旧版靠手工扫 sys.argv 决定 import 哪个 args 模块，而两个模块又各缺对方的参数
# （`--ndim 2` 在 2D 分支下直接 unrecognized）。现在只有一份参数定义。
from args import parse_args, prepare_run_dir

_use_3d = ('--use-3d' in sys.argv) or \
    ('--ndim' in sys.argv and sys.argv[sys.argv.index('--ndim') + 1:][:1] == ['3'])
args = parse_args(default_ndim=3 if _use_3d else 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.ndim == 3:
    from models.dense_ed_3d import DenseED
    from utils.load_data_3d import load_data, load_train_val
else:
    from models.dense_ed import DenseED
    from utils.load_data import load_data, load_train_val

from utils.data_common import compute_output_stats                    # noqa: E402
from utils.geometry import mask_coverage, parse_geometry, t_mask_like  # noqa: E402
from utils.misc import RunningMetrics, mkdirs, new_logger             # noqa: E402
from utils.physics_loss import (PhysicsScaler, combined_physics_loss,  # noqa: E402
                                physics_report, resolve_channels)
from utils.plot import plot_prediction, save_stats                    # noqa: E402


# ------------------------------------------------------------------ 路径解析
def resolve_path(value, data_dir, fallback=None):
    if value is None:
        return fallback
    if os.path.isabs(value) or os.path.dirname(value):
        return value
    return os.path.join(data_dir, value)


def default_train_path(args):
    if args.ndim == 3:
        return os.path.join(args.data_dir, '10243d.hdf5')
    return os.path.join(args.data_dir,
                        'kle{}_lhs{}.hdf5'.format(args.kle, args.ntrain))


train_path = resolve_path(args.train_file, args.data_dir, default_train_path(args))
test_path = resolve_path(args.test_file, args.data_dir, None)

if not os.path.exists(train_path):
    raise FileNotFoundError(
        '找不到训练数据 {}\n请把 HDF5 放进 dataset/ 或用 --train-file 指定路径。'
        .format(train_path))

same_file = (test_path is None) or (os.path.abspath(test_path) ==
                                    os.path.abspath(train_path))

prepare_run_dir(args)
args.train_dir = args.run_dir + '/training'
args.pred_dir = args.train_dir + '/predictions'
mkdirs([args.train_dir, args.pred_dir])


# ------------------------------------------------------------------ 形状推断
def infer_shapes(path, args):
    import h5py
    with h5py.File(path, 'r') as f:
        y_shape = f['output'].shape
        noc = int(y_shape[1])
        spatial = list(y_shape[2:])
        if args.ndim == 3 and len(spatial) == 2:
            spatial = [1] + spatial

        input_key = args.input_key
        if input_key is None:
            candidates = ['sensor_input', 'input'] if args.prefer_sensor_input \
                else ['input', 'sensor_input']
            input_key = next((k for k in candidates if k in f), None)
        if input_key is None:
            raise KeyError('HDF5 中找不到 input / sensor_input')
        x_shape = f[input_key].shape
        nic = int(x_shape[1]) if len(x_shape) >= 4 else 1
    return nic, noc, spatial


inferred_nic, inferred_noc, inferred_spatial = infer_shapes(train_path, args)
args.nic = args.nic or inferred_nic
args.noc = args.noc or inferred_noc
args.image_size = args.image_size or inferred_spatial

# 3D 的输入是把 spatial_input 与标量向量拼在一起，通道数要以实际组装结果为准
print('[train] 数据维度 ndim={} · 输入通道={} · 输出通道={} · 分辨率={}'
      .format(args.ndim, args.nic, args.noc, args.image_size))

# ------------------------------------------------------------------ 物理常量
grid_hw = args.image_size[-2:]
dy = float(args.domain_size[0]) / max(int(grid_hw[0]) - 1, 1)
dx = float(args.domain_size[1]) / max(int(grid_hw[1]) - 1, 1)
geometry = None if args.no_mask else parse_geometry(args.t_geometry)

print('[train] 网格间距 dx={:.6g} dy={:.6g}（domain_size={}）'
      .format(dx, dy, args.domain_size))
_ch_map = resolve_channels(args.channel_order, args.noc)
print('[train] 物理任务 --task={} · 通道顺序 --channel-order={} -> {}'.format(
    args.task, args.channel_order,
    ', '.join('{}={}'.format(k.upper(), v) for k, v in sorted(
        _ch_map.items(), key=lambda kv: kv[1]))))
print('!!! 请确认该顺序与数据集 output 的实际通道语义一致，'
      '否则所有物理约束都作用在错误的通道上 !!!')
if args.task == 'dispersion':
    print('[train] 对流扩散：D={:.3g} · 速度取自输入通道 {} · 权重={}'.format(
        args.diffusivity, args.velocity_channels, args.advdiff_weight))
else:
    if args.ns_weight <= 0:
        print('[train] --ns-weight=0，跳过 N-S 动量残差（只保留连续性方程）。'
              '高雷诺数 RANS 时均场用层流动量残差会把模型往错误方向拉，'
              '要恢复需 CFD 导出湍流粘度场并改用有效粘度')
    if args.div_weight <= 0:
        print('[train] --div-weight=0，跳过连续性方程残差')
if geometry is not None:
    import numpy as _np
    _kind = '栅格' if isinstance(geometry, _np.ndarray) else '矩形段'
    print('[train] 计算域掩码（{}）覆盖率 {:.1%} @ {}x{}'.format(
        _kind, mask_coverage(int(grid_hw[0]), int(grid_hw[1]), geometry),
        int(grid_hw[0]), int(grid_hw[1])))
else:
    print('[train] 已关闭计算域掩码（--no-mask），在整幅网格上训练')


# ------------------------------------------------------------------ 数据
loader_kwargs = dict(input_key=args.input_key,
                     prefer_sensor_input=args.prefer_sensor_input,
                     num_workers=args.num_workers)

stats = compute_output_stats(train_path)

if same_file:
    train_loader, test_loader, stats = load_train_val(
        train_path, args.batch_size, val_split=args.val_split, seed=args.seed,
        test_batch_size=args.test_batch_size, stats=stats, **loader_kwargs)
    if test_loader is None:
        print('[train] 警告：--val-split=0，验证集为空，将回退到用训练集评估'
              '（此时 R² 是样本内指标，不代表泛化能力）')
        test_loader, _ = load_data(train_path, args.test_batch_size, stats=stats,
                                   shuff=False, **loader_kwargs)
else:
    train_loader, stats = load_data(train_path, args.batch_size, stats=stats,
                                    shuff=True, **loader_kwargs)
    # 测试集必须用训练集的统计量归一化，否则两边标度不同，R² 不可比
    test_loader, _ = load_data(test_path, args.test_batch_size, stats=stats,
                               shuff=False, **loader_kwargs)

y_min = torch.as_tensor(stats['y_min'], dtype=torch.float32, device=device)
y_max = torch.as_tensor(stats['y_max'], dtype=torch.float32, device=device)
phys_min = y_min if args.physics_in_physical_units else None
phys_max = y_max if args.physics_in_physical_units else None
print('[train] 训练样本 {} · 测试样本 {}'
      .format(len(train_loader.dataset), len(test_loader.dataset)))


# ------------------------------------------------------------------ 模型
model = DenseED(
    in_channels=args.nic, out_channels=args.noc, blocks=args.blocks,
    growth_rate=args.growth_rate, init_features=args.init_features,
    drop_rate=args.drop_rate, bn_size=args.bn_size, bottleneck=args.bottleneck,
    out_activation=None,
    output_size=tuple(args.image_size) if args.ndim == 3 else tuple(args.image_size[-2:]),
    num_heads=args.num_heads,
    use_positional_encoding=not args.no_positional_encoding,
    attn_max_tokens=args.attn_max_tokens,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                              patience=args.lr_patience, threshold=1e-4,
                              threshold_mode='rel', min_lr=1e-6)

logger = new_logger()
physics_scaler = PhysicsScaler(enabled=(args.physics_loss_scaling == 'ema'))


def extract_velocity(inputs):
    """dispersion 任务下，从**输入**张量里取出速度场 (Vx, Vy)。

    两阶段架构里流场是阶段一的产物，作为输入通道喂给阶段二，不在 output 里。
    输入不参与 target 的归一化流程（loader 只归一化 output），因此这里拿到的
    就是有量纲的物理速度，可直接送进对流扩散残差。
    """
    if args.task != 'dispersion':
        return None
    ix, iy = args.velocity_channels
    if inputs.shape[1] <= max(ix, iy):
        raise ValueError(
            '--velocity-channels {} 超出输入通道数 {}，请检查数据集或该参数'
            .format(args.velocity_channels, inputs.shape[1]))
    return inputs[:, ix:ix + 1], inputs[:, iy:iy + 1]


def build_mask(tensor):
    if geometry is None:
        return None
    return t_mask_like(tensor, geometry)


def save_checkpoint(epoch, path=None):
    """自描述 checkpoint：权重 + 超参 + 归一化统计量。"""
    path = path or '{}/model_epoch{}.pth'.format(args.ckpt_dir, epoch)
    torch.save({
        'format': 2,
        'epoch': epoch,
        'model': model.state_dict(),
        'args': {k: v for k, v in vars(args).items()
                 if isinstance(v, (int, float, str, bool, list, tuple, type(None)))},
        'stats': {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in stats.items()},
    }, path)
    return path


# ------------------------------------------------------------------ 评估
@torch.no_grad()
def log_physics(logger, report):
    """只记录当前任务真正计算了的物理残差，跳过 nan。

    把 dispersion 的残差硬塞进 div_rms 会让曲线图和历史数据串味，所以按 key 各归各位。
    """
    for key in ('div_rms', 'ns_rms', 'advdiff_rms'):
        value = report.get(key, float('nan'))
        if value == value:                      # 非 nan
            logger[key].append(value)


def evaluate(epoch, do_plot=False):
    model.eval()
    metrics = RunningMetrics(args.noc)
    phys = {}

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        mask = build_mask(outputs)
        metrics.update(outputs, targets, mask)

        if batch_idx == 0:
            report = physics_report(outputs, mask, args.channel_order,
                                    phys_min, phys_max, args.rho, args.nu, dx, dy,
                                    task=args.task,
                                    velocity=extract_velocity(inputs),
                                    diffusivity=args.diffusivity)
            phys = report

        if do_plot and batch_idx == 0:
            n_samples = min(6 if epoch >= args.epochs else 2, inputs.size(0))
            mask_np = None
            if mask is not None:
                mask_np = mask.reshape(-1, *mask.shape[-2:])[0].cpu().numpy()
            for i in range(n_samples):
                plot_prediction(args.pred_dir, targets[i].cpu(), outputs[i].cpu(),
                                epoch, i, plot_fn=args.plot_fn,
                                channel_order=args.channel_order, mask=mask_np)

    return metrics, phys


# ------------------------------------------------------------------ 训练
def train_one_epoch(epoch):
    model.train()
    metrics = RunningMetrics(args.noc)
    accum = {'base': 0.0, 'div': 0.0, 'ns': 0.0, 'advdiff': 0.0,
             'smooth': 0.0, 'total': 0.0}
    n_batches = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        mask = build_mask(outputs)

        loss, parts = combined_physics_loss(
            outputs, targets, mask=mask, epoch=epoch, total_epochs=args.epochs,
            div_weight=args.div_weight, ns_weight=args.ns_weight,
            smooth_weight=args.smooth_weight, channel_order=args.channel_order,
            y_min=phys_min, y_max=phys_max, rho=args.rho, nu=args.nu,
            dx=dx, dy=dy, physics_warmup=args.physics_warmup,
            scales=physics_scaler.current(),
            task=args.task, velocity=extract_velocity(inputs),
            diffusivity=args.diffusivity, advdiff_weight=args.advdiff_weight)
        physics_scaler.update(parts)

        if not torch.isfinite(loss):
            print('[train] epoch {}: 损失出现 NaN/Inf，跳过该 batch'.format(epoch))
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        metrics.update(outputs.detach(), targets, mask)
        for key in accum:
            accum[key] += parts[key]
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError('训练集为空，请检查 --val-split 与数据文件')
    for key in accum:
        accum[key] /= n_batches
    return metrics, accum


def main():
    print('=' * 72)
    print('开始训练 · device={} · epochs={}'.format(device, args.epochs))
    print('=' * 72)
    tic = time()
    best_rmse, best_path = float('inf'), None

    if args.post:
        ckpt_epoch = args.ckpt_epoch or args.epochs
        ckpt_path = '{}/model_epoch{}.pth'.format(args.ckpt_dir, ckpt_epoch)
        if os.path.exists(ckpt_path):
            blob = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(blob['model'] if isinstance(blob, dict)
                                  and 'model' in blob else blob)
            print('[post] 已加载 {}'.format(ckpt_path))
        else:
            print('[post] 警告：找不到 {}，使用当前（未训练）权重'.format(ckpt_path))

        metrics, phys = evaluate(ckpt_epoch, do_plot=True)
        logger['r2_test'].append(metrics.r2())
        logger['rmse_test'].append(metrics.rmse())
        logger['r2_test_per_channel'].append(metrics.r2_per_channel().tolist())
        log_physics(logger, phys)
        print('[post] R²={:.6f}  RMSE={:.6f}  分通道 R²={}'
              .format(metrics.r2(), metrics.rmse(),
                      np.round(metrics.r2_per_channel(), 4).tolist()))
    else:
        for epoch in range(1, args.epochs + 1):
            train_metrics, parts = train_one_epoch(epoch)
            do_plot = (epoch % args.plot_freq == 0) or (epoch == args.epochs)
            test_metrics, phys = evaluate(epoch, do_plot=do_plot)

            # 学习率按**验证** RMSE 调度（旧版用的是训练 RMSE）
            scheduler.step(test_metrics.rmse())

            # 只打当前任务真正参与的物理项，否则一排恒为 0 的数字反而掩盖信息
            if args.task == 'dispersion':
                phys_txt = 'advdiff={:.3e}'.format(parts['advdiff'])
            else:
                phys_txt = 'div={:.3e} ns={:.3e}'.format(parts['div'], parts['ns'])
            print('epoch {:>4}｜train R²={:.5f} RMSE={:.5f}｜'
                  'test R²={:.5f} RMSE={:.5f}｜'
                  'base={:.4e} {} smooth={:.3e}｜lr={:.2e}'
                  .format(epoch, train_metrics.r2(), train_metrics.rmse(),
                          test_metrics.r2(), test_metrics.rmse(),
                          parts['base'], phys_txt,
                          parts['smooth'], optimizer.param_groups[0]['lr']))

            if epoch % args.log_freq == 0 or epoch == args.epochs:
                logger['r2_train'].append(train_metrics.r2())
                logger['rmse_train'].append(train_metrics.rmse())
                logger['r2_test'].append(test_metrics.r2())
                logger['rmse_test'].append(test_metrics.rmse())
                logger['r2_test_per_channel'].append(
                    test_metrics.r2_per_channel().tolist())
                log_physics(logger, phys)

            if epoch % args.ckpt_freq == 0 or epoch == args.epochs:
                save_checkpoint(epoch)

            if test_metrics.rmse() < best_rmse:
                best_rmse = test_metrics.rmse()
                best_path = save_checkpoint(epoch, args.ckpt_dir + '/model_best.pth')

        print('最优验证 RMSE={:.6f} -> {}'.format(best_rmse, best_path))
        final = logger['r2_test_per_channel'][-1] if logger['r2_test_per_channel'] else []
        print('最终分通道 R²: {}'.format(np.round(final, 4).tolist()))

    elapsed = time() - tic
    logger['time'].append(elapsed)
    print('耗时 {:.1f} 秒（含绘图）'.format(elapsed))

    save_stats(args.train_dir, logger)

    args.training_time = elapsed
    args.n_params, args.n_layers = model._num_parameters_convlayers()
    args.best_rmse = best_rmse if best_rmse != float('inf') else None
    args.final_r2 = logger['r2_test'][-1] if logger['r2_test'] else None
    args.norm_stats = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in stats.items()}
    with open(args.run_dir + '/args.txt', 'w', encoding='utf-8') as fh:
        json.dump(vars(args), fh, indent=4, ensure_ascii=False, default=str)


if __name__ == '__main__':
    main()
