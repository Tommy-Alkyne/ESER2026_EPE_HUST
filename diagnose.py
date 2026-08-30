#!/usr/bin/env python3
"""训练结果智能诊断。

读取 experiments/ 下最新一次实验的日志与指标，输出中文体检报告。

相对旧版的改进
--------------
* 识别新格式产物：分通道 R²、物理残差 (div_rms / ns_rms)、model_best.pth。
* 新增**配置合理性检查**：训练集与测试集是否是同一个文件、val_split 是否为 0、
  通道顺序是否显式确认过 —— 这些配置问题会让再漂亮的 R² 也失去意义，
  比调参建议重要得多。
* 预测图的扩展名不再写死 `.pdf`。
"""

import json
import sys
from pathlib import Path

import numpy as np


class C:
    HEAD = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'


def header(title):
    print('\n{}{}{}'.format(C.HEAD, '=' * 72, C.END))
    print('{}{}{}'.format(C.HEAD, title.center(72), C.END))
    print('{}{}{}\n'.format(C.HEAD, '=' * 72, C.END))


def info(msg):
    print('{}[INFO]{} {}'.format(C.BLUE, C.END, msg))


def ok(msg):
    print('{}[ OK ]{} {}'.format(C.GREEN, C.END, msg))


def warn(msg):
    print('{}[WARN]{} {}'.format(C.YELLOW, C.END, msg))


def err(msg):
    print('{}[FAIL]{} {}'.format(C.RED, C.END, msg))


def metric(name, value, good=None, okay=None, unit=''):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        print('  {:26s}: --'.format(name))
        return
    status, color = '', ''
    if good is not None and okay is not None:
        higher_better = good > okay
        if (value >= good) if higher_better else (value <= good):
            color, status = C.GREEN, '✓ 优秀'
        elif (value >= okay) if higher_better else (value <= okay):
            color, status = C.YELLOW, '~ 达标'
        else:
            color, status = C.RED, '✗ 欠佳'
        print('  {:26s}: {}{:12.6f}{} {:8s} [{}]'
              .format(name, color, value, C.END, unit, status))
    else:
        print('  {:26s}: {:12.6f} {}'.format(name, value, unit))


def find_latest_experiment(root='experiments'):
    exp_root = Path(root)
    if not exp_root.exists():
        return None
    candidates = []
    for d in exp_root.rglob('args.txt'):
        candidates.append(d.parent)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception as exc:
        warn('读取 {} 失败: {}'.format(path, exc))
        return {}


def load_series(train_dir, name):
    path = Path(train_dir) / '{}.txt'.format(name)
    if not path.exists():
        return None
    try:
        arr = np.atleast_1d(np.loadtxt(path))
        return arr if arr.size else None
    except Exception:
        return None


def check_configuration(cfg):
    """配置合理性检查 —— 比任何调参建议都更值得先看。"""
    header('配置合理性检查')
    problems = []

    train_file = cfg.get('train_file')
    test_file = cfg.get('test_file')
    val_split = cfg.get('val_split')

    if train_file and test_file and str(train_file) == str(test_file):
        if not val_split:
            problems.append(
                '训练集与测试集是同一个文件且 val_split=0 —— 报告的 R² 是'
                '训练集内指标，不代表泛化能力。请设 --val-split 0.2 或提供独立测试集。')
        else:
            ok('训练/测试同文件，但已按 val_split={} 切出独立验证集'.format(val_split))
    elif train_file and test_file:
        ok('使用独立的测试集文件')
    elif not val_split:
        problems.append('未提供独立测试集且 val_split=0，评估指标不可信')
    else:
        ok('已按 val_split={} 切分验证集'.format(val_split))

    order = cfg.get('channel_order')
    if order:
        mapping = '(P, Vx, Vy)' if order == 'pvv' else '(Vx, Vy, P)'
        warn('通道顺序设为 {} -> {}。请确认与数据集 output 的实际语义一致，'
             '否则物理约束全部作用在错误通道上。'.format(order, mapping))
    else:
        problems.append('args.txt 里没有 channel_order，可能是旧版实验产物')

    if cfg.get('no_mask'):
        warn('本次实验关闭了 T 型掩码（--no-mask），指标覆盖整幅网格')

    if cfg.get('physics_in_physical_units') is False:
        warn('物理损失运行在归一化空间（--physics-in-physical-units false），'
             '散度/动量残差不代表真实物理量')

    domain = cfg.get('domain_size')
    if domain and list(domain) == [1.0, 1.0]:
        warn('domain_size 仍是默认的 1x1，若真实计算域不是单位尺寸，'
             'dx/dy 与雷诺数就对不上，物理残差的绝对数值没有意义')

    if problems:
        for p in problems:
            err(p)
    else:
        ok('未发现配置层面的硬伤')
    return problems


def diagnose_quality(metrics):
    header('模型质量诊断')
    r2 = metrics.get('final_r2')
    rmse = metrics.get('final_rmse')
    if r2 is None or rmse is None:
        warn('指标数据不足，无法诊断')
        return

    print('核心指标:')
    metric('R² Score', r2, good=0.95, okay=0.90)
    metric('RMSE (归一化尺度)', rmse, good=0.03, okay=0.05)

    per_channel = metrics.get('r2_per_channel')
    if per_channel is not None:
        print('\n分通道 R²:')
        for c, value in enumerate(np.atleast_1d(per_channel)):
            metric('  ch{}'.format(c), float(value), good=0.95, okay=0.90)
        spread = float(np.nanmax(per_channel) - np.nanmin(per_channel))
        if spread > 0.15:
            warn('各通道 R² 差距 {:.3f}，整体指标被好通道拉高了，'
                 '建议单独看最差的那个通道'.format(spread))

    if metrics.get('div_rms') is not None:
        print('\n物理残差（有量纲）:')
        metric('  连续性 |div v| RMS', float(metrics['div_rms']))
        metric('  动量方程残差 RMS', float(metrics['ns_rms'])
               if metrics.get('ns_rms') is not None else None)

    print('\n建议:')
    suggestions = []
    if r2 < 0.80:
        suggestions += ['提高模型容量：--growth-rate 24~32 / --init-features 64',
                        '训练轮数加到 400+',
                        '确认输入信息是否足以决定输出（传感器点位是否太稀疏）']
    elif r2 < 0.90:
        suggestions += ['再跑 300~400 轮', '微调学习率与 --lr-patience']
    elif r2 < 0.95:
        suggestions += ['已经不错；可尝试更大的 growth-rate 或更长训练']
    else:
        ok('R² ≥ 0.95')

    if rmse > 0.08:
        suggestions += ['检查归一化统计量是否过期（删掉 dataset/*.stats.npz 重算）',
                        '检查 --t-geometry 是否与实际管道几何匹配']
    elif rmse > 0.05:
        suggestions += ['适当提高 --smooth-weight 或延长训练']

    r2_series = metrics.get('r2_series')
    if r2_series is not None and len(r2_series) > 3:
        recent = r2_series[-max(3, len(r2_series) // 5):]
        if float(np.max(recent) - np.min(recent)) < 1e-4:
            suggestions.append('验证 R² 已经平台化，继续训练收益有限')
        if len(r2_series) > 5 and r2_series[-1] < np.max(r2_series) - 0.02:
            warn('验证 R² 从峰值回落了 {:.3f}，可能开始过拟合，'
                 '请用 model_best.pth 而不是最后一轮'
                 .format(float(np.max(r2_series) - r2_series[-1])))

    for s in suggestions:
        info('• ' + s)


def main():
    header('T 型流场代理模型 · 训练诊断')

    exp_dir = find_latest_experiment()
    if exp_dir is None:
        err('未找到实验目录（experiments/ 下需要有含 args.txt 的文件夹）')
        return 1

    info('实验目录: {}'.format(exp_dir))
    train_dir = exp_dir / 'training'
    pred_dir = train_dir / 'predictions'

    cfg = load_json(exp_dir / 'args.txt')

    header('实验配置')
    for key in ('ndim', 'train_file', 'test_file', 'val_split', 'channel_order',
                'epochs', 'batch_size', 'lr', 'weight_decay', 'blocks',
                'growth_rate', 'init_features', 'div_weight', 'ns_weight',
                'smooth_weight', 'domain_size', 'rho', 'nu', 'n_params',
                'training_time'):
        if key in cfg:
            print('  {:24s}: {}'.format(key, cfg[key]))

    problems = check_configuration(cfg)

    metrics = {}
    r2_series = load_series(train_dir, 'r2_test')
    rmse_series = load_series(train_dir, 'rmse_test')
    if r2_series is not None:
        metrics['final_r2'] = float(r2_series[-1])
        metrics['r2_series'] = r2_series
    if rmse_series is not None:
        metrics['final_rmse'] = float(rmse_series[-1])
    per_channel = load_series(train_dir, 'r2_test_per_channel')
    if per_channel is not None:
        arr = np.atleast_2d(per_channel)
        metrics['r2_per_channel'] = arr[-1]
    for key in ('div_rms', 'ns_rms'):
        series = load_series(train_dir, key)
        if series is not None:
            metrics[key] = float(series[-1])

    header('训练结果')
    if metrics:
        metric('最终 R²', metrics.get('final_r2'), good=0.95, okay=0.90)
        metric('最终 RMSE', metrics.get('final_rmse'), good=0.03, okay=0.05)
    else:
        warn('没有找到 r2_test.txt / rmse_test.txt，训练可能未正常结束')

    header('产物文件')
    ckpt_dir = exp_dir / 'checkpoints'
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob('*.pth'))
        info('checkpoint 数量: {}'.format(len(ckpts)))
        best = ckpt_dir / 'model_best.pth'
        if best.exists():
            ok('最优权重: {}'.format(best))
        elif ckpts:
            info('最新权重: {}'.format(ckpts[-1]))
    if pred_dir.exists():
        images = sorted(list(pred_dir.glob('pred_*.*')))
        info('预测对比图: {} 张'.format(len(images)))
        if images:
            ok('最新一张: {}'.format(images[-1]))

    if metrics:
        diagnose_quality(metrics)

    header('下一步')
    if problems:
        err('先解决上面的配置问题，否则任何指标都不可信')
    info('1. 打开预测对比图：第 1 行真值 / 第 2 行预测 / 第 3 行残差')
    info('   残差行用以 0 为中心的对称色阶，接近纯白即拟合良好')
    info('2. 曲线图：{}/r2.pdf, {}/rmse.pdf, {}/r2_per_channel.pdf'
         .format(train_dir, train_dir, train_dir))
    info('3. 物理残差曲线：{}/div_rms.pdf, {}/ns_rms.pdf'
         .format(train_dir, train_dir))
    info('4. 离线推理：python surrogate_predict.py --ckpt-file {}/model_best.pth '
         '--data-file <你的数据>'.format(ckpt_dir))
    return 0


if __name__ == '__main__':
    sys.exit(main())
