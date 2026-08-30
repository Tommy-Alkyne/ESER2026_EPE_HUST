"""脱机推理：加载 checkpoint，对给定 HDF5 做前向预测并保存结果。

用法
----
    python surrogate_predict.py \
        --ckpt-file experiments/my_run/checkpoints/model_best.pth \
        --data-file dataset/p_v_256.hdf5 \
        --out-file experiments/my_run/pred.hdf5

修复内容
--------
1. **2D 路径不再必崩**。旧版用 4 个关键字参数调 `load_data`，而 2D 版签名只有
   `(data_dir, batch_size)` —— 一跑就 TypeError。现在两边签名统一。
2. **输出反归一化回物理量纲**。旧版把 [0,1] 的归一化结果直接写盘，且没保存
   min/max，下游根本无法还原成压力/速度，代理模型的实用价值归零。现在默认
   写出有量纲结果，并把归一化统计量一并存进文件。
3. **超参从 checkpoint 里读**，不再要求用户在命令行手工重复一遍（写错就静默
   加载失败或形状对不上）。
4. **T 掩码复用 utils/geometry**，不再是第二份硬编码像素坐标。
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch

from utils.data_common import compute_output_stats, denormalize_array
from utils.geometry import parse_geometry, t_mask_numpy


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='T 型管流场代理模型离线推理')
    p.add_argument('--ckpt-file', type=str, required=True)
    p.add_argument('--data-file', type=str, required=True)
    p.add_argument('--out-file', type=str, default=None)

    p.add_argument('--ndim', type=int, default=None, choices=[2, 3],
                   help='缺省从 checkpoint 或数据文件自动判断')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--limit', type=int, default=None, help='只推理前 N 个样本')
    p.add_argument('--device', type=str, default=None)

    p.add_argument('--denormalize', dest='denormalize', action='store_true',
                   default=True, help='输出有量纲物理场（默认）')
    p.add_argument('--keep-normalized', dest='denormalize', action='store_false',
                   help='保留 [0,1] 归一化输出')
    p.add_argument('--apply-mask', dest='apply_mask', action='store_true',
                   default=True, help='把计算域外置零（默认）')
    p.add_argument('--no-mask', dest='apply_mask', action='store_false')
    p.add_argument('--t-geometry', '--domain-mask', dest='t_geometry',
                   type=str, default=None,
                   help='覆盖计算域几何。**缺省沿用 checkpoint 里训练时用的那份**，'
                        '只有明确要换几何时才需要指定')
    p.add_argument('--input-key', type=str, default=None)
    p.add_argument('--prefer-sensor-input', action='store_true', default=False)

    # 只在 checkpoint 里缺少元信息时才需要手工指定
    p.add_argument('--blocks', type=str, default=None)
    p.add_argument('--growth-rate', type=int, default=None)
    p.add_argument('--init-features', type=int, default=None)
    p.add_argument('--drop-rate', type=float, default=None)
    p.add_argument('--bn-size', type=int, default=None)
    p.add_argument('--bottleneck', action='store_true', default=None)
    return p.parse_args(argv)


# checkpoint 里能被读回来的超参白名单。**不在这里的键会被静默丢弃**——
# t_geometry / no_mask 曾经就漏在外面，导致推理拿默认 T 型几何去裁剪一个
# 车站模型的输出，覆盖率 45% vs 25%，而且一句警告都没有。
DEFAULT_HP = {
    'blocks': [3, 6, 3], 'growth_rate': 16, 'init_features': 48,
    'drop_rate': 0.0, 'bn_size': 8, 'bottleneck': False, 'num_heads': 4,
    'attn_max_tokens': 8192, 'no_positional_encoding': False,
    'channel_order': 'pvv',
    'task': 'flow', 'diffusivity': 1e-5,
    't_geometry': None, 'no_mask': False,
}


def _parse_blocks(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    try:
        return [int(v) for v in json.loads(str(value).replace("'", '"'))]
    except Exception:
        return [int(t) for t in str(value).strip('[]()').split(',') if t.strip()]


def load_checkpoint(path, device):
    """兼容新版自描述 checkpoint 与旧版裸 state_dict。"""
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and 'model' in blob and isinstance(blob['model'], dict):
        return blob['model'], blob.get('args', {}) or {}, blob.get('stats', None)
    return blob, {}, None


def infer_channels(state, ckpt_args):
    """从权重形状反推输入/输出通道数，兼容新旧命名。"""
    nic = ckpt_args.get('nic')
    noc = ckpt_args.get('noc')
    for key, tensor in state.items():
        if nic is None and key.endswith('in_conv.weight'):
            nic = int(tensor.shape[1])
        if noc is None and key.endswith('last_trans_up.conv2.weight'):
            noc = int(tensor.shape[0])
        if noc is None and key.endswith('LastTransUp.convT2.weight'):
            noc = int(tensor.shape[1])   # ConvTranspose: (in, out, k, k)
    if nic is None or noc is None:
        raise RuntimeError('无法从 checkpoint 推断通道数，请检查文件是否完整')
    return nic, noc


def detect_ndim(data_file, ckpt_args, override=None):
    if override is not None:
        return override
    if ckpt_args.get('ndim'):
        return int(ckpt_args['ndim'])
    with h5py.File(data_file, 'r') as f:
        return 3 if f['output'].ndim == 5 else 2


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state, ckpt_args, ckpt_stats = load_checkpoint(args.ckpt_file, device)
    ndim = detect_ndim(args.data_file, ckpt_args, args.ndim)
    nic, noc = infer_channels(state, ckpt_args)

    hp = dict(DEFAULT_HP)
    hp.update({k: v for k, v in ckpt_args.items() if k in hp and v is not None})
    for key in ('growth_rate', 'init_features', 'drop_rate', 'bn_size'):
        if getattr(args, key) is not None:
            hp[key] = getattr(args, key)
    if args.blocks is not None:
        hp['blocks'] = _parse_blocks(args.blocks)
    if args.bottleneck is not None:
        hp['bottleneck'] = args.bottleneck
    hp['blocks'] = _parse_blocks(hp['blocks'])

    print('[predict] ndim={} nic={} noc={} blocks={} channel_order={}'
          .format(ndim, nic, noc, hp['blocks'], hp['channel_order']))

    if ndim == 3:
        from models.dense_ed_3d import DenseED
        from utils.load_data_3d import load_data
    else:
        from models.dense_ed import DenseED
        from utils.load_data import load_data

    # 归一化统计量：优先用 checkpoint 里训练时的那一份，保证标度一致
    if ckpt_stats and 'y_min' in ckpt_stats:
        stats = {k: np.asarray(v) if isinstance(v, (list, tuple)) else v
                 for k, v in ckpt_stats.items()}
        print('[predict] 使用 checkpoint 内保存的归一化统计量')
    else:
        stats = compute_output_stats(args.data_file)
        print('[predict] checkpoint 内无统计量，回退到按数据文件重新统计'
              '（若与训练集不同，结果标度会有偏差）')

    loader, stats = load_data(args.data_file, args.batch_size, stats=stats,
                              input_key=args.input_key,
                              prefer_sensor_input=args.prefer_sensor_input,
                              align_input_to_output=True, shuff=False)

    sample_x, sample_y = loader.dataset[0]
    output_size = tuple(sample_y.shape[-3:]) if ndim == 3 else tuple(sample_y.shape[-2:])

    model = DenseED(in_channels=nic, out_channels=noc, blocks=hp['blocks'],
                    growth_rate=hp['growth_rate'],
                    init_features=hp['init_features'], drop_rate=hp['drop_rate'],
                    bn_size=hp['bn_size'], bottleneck=hp['bottleneck'],
                    out_activation=None, output_size=output_size,
                    num_heads=hp['num_heads'],
                    use_positional_encoding=not hp['no_positional_encoding'],
                    attn_max_tokens=hp['attn_max_tokens'], verbose=False).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print('[predict] 警告：权重不完全匹配 · missing={} unexpected={}'
              .format(len(missing), len(unexpected)))
        if missing:
            print('          缺失示例: {}'.format(missing[:5]))
        if unexpected:
            print('          多余示例: {}'.format(unexpected[:5]))
    model.eval()

    n_total = len(loader.dataset)
    if args.limit is not None:
        n_total = min(n_total, args.limit)
    preds = np.zeros((n_total,) + tuple(sample_y.shape), dtype=np.float32)

    written = 0
    with torch.no_grad():
        for x, _ in loader:
            if written >= n_total:
                break
            y = model(x.to(device)).cpu().numpy()
            end = min(written + y.shape[0], n_total)
            preds[written:end] = y[:end - written]
            written = end

    if args.denormalize:
        preds = denormalize_array(preds, stats['y_min'], stats['y_max'])
        print('[predict] 已反归一化回物理量纲')

    if args.apply_mask:
        # 几何必须跟训练时一致，否则推理会按另一个计算域裁剪结果，而且不会报错。
        # 与 --channel-order 同理：缺省从 checkpoint 读，命令行只作为显式覆盖。
        geom_spec = args.t_geometry
        source = '命令行'
        if geom_spec is None:
            if hp.get('no_mask'):
                geom_spec, source = None, '训练时未用掩码'
            else:
                geom_spec, source = hp.get('t_geometry'), 'checkpoint'
        if source == '训练时未用掩码':
            print('[predict] 训练时用的是 --no-mask，跳过掩码以保持一致')
        else:
            geometry = parse_geometry(geom_spec)
            h, w = preds.shape[-2:]
            mask = t_mask_numpy(h, w, geometry).astype(bool)
            preds[..., ~mask] = 0.0
            print('[predict] 计算域掩码（来源：{}）覆盖率 {:.1%} @ {}x{}'
                  .format(source, mask.mean(), h, w))

    out_file = args.out_file
    if out_file is None:
        base = os.path.splitext(os.path.basename(args.data_file))[0]
        out_file = os.path.join(os.path.dirname(args.data_file) or '.',
                                base + '.pred.hdf5')
    mkdir = os.path.dirname(out_file)
    if mkdir:
        os.makedirs(mkdir, exist_ok=True)

    with h5py.File(out_file, 'w') as f:
        f.create_dataset('predictions', data=preds, compression='gzip')
        # 把还原量纲所需的一切都存进去，下游不必再猜
        f.create_dataset('y_min', data=np.asarray(stats['y_min'], dtype=np.float32))
        f.create_dataset('y_max', data=np.asarray(stats['y_max'], dtype=np.float32))
        f.attrs['source_data'] = args.data_file
        f.attrs['ckpt_file'] = args.ckpt_file
        f.attrs['ndim'] = ndim
        f.attrs['nic'] = nic
        f.attrs['noc'] = noc
        f.attrs['denormalized'] = int(bool(args.denormalize))
        f.attrs['masked'] = int(bool(args.apply_mask))
        f.attrs['channel_order'] = str(hp['channel_order'])
        f.attrs['hyperparams'] = json.dumps(hp, default=str)

    print('[predict] 已保存 {} · shape={}'.format(out_file, preds.shape))
    return out_file


if __name__ == '__main__':
    main()
