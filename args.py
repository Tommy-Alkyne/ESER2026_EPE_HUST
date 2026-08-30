"""统一的超参数定义（2D / 3D 共用）。

相对旧版的修正
--------------
1. **`--ndim` / `--use-3d` 两个 parser 都认识**。旧版 `args.py` 没定义
   `--ndim`、`args_3d.py` 没定义 `--use-3d`，于是 `python train.py --ndim 2`
   会直接 `unrecognized arguments` 报错。
2. **`--blocks` 可用**。旧版 `type=list` 会把 "[3, 6, 3]" 拆成单个字符
   （`['[', '3', ',', ...]`），只有默认值能跑。
3. **import 时不再产生文件系统副作用**。旧版在 `parse()` 里就 `mkdirs` 并写
   `args.txt`，模块无法被安全 import；现在这些动作交给 `train.py` 显式调用
   `prepare_run_dir(args)`。
4. 新增物理 / 划分 / 几何相关参数，取代散落在 `train.py` 里的硬编码常量。
"""

import argparse
import json
import random

import torch


def _blocks(value):
    """解析 --blocks，接受 "[3,6,3]" / "3,6,3" / "3 6 3"。"""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    try:
        parsed = json.loads(text.replace("'", '"'))
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
        return [int(parsed)]
    except (ValueError, TypeError):
        pass
    sep = ',' if ',' in text else None
    return [int(tok) for tok in text.strip('[]()').split(sep) if tok.strip()]


def _bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('1', 'true', 'yes', 'y', 'on')


def build_parser(default_ndim=2, description=None):
    p = argparse.ArgumentParser(
        description=description or 'Dense Convolutional Encoder-Decoder Networks')

    # ---------------------------------------------------------------- 实验
    p.add_argument('--exp-name', type=str,
                   default='deterministic_3d' if default_ndim == 3 else 'deterministic',
                   help='实验名（会作为 experiments/ 下的目录名）')
    p.add_argument('--exp-dir', type=str, default='./experiments')
    p.add_argument('--post', action='store_true', default=False,
                   help='只做后处理评估，不训练')

    # ---------------------------------------------------------------- 网络
    p.add_argument('--blocks', type=_blocks, default=[3, 6, 3],
                   help='各 dense block 的层数，如 "[3,6,3]"（长度须为奇数）')
    p.add_argument('--growth-rate', type=int, default=16)
    p.add_argument('--init-features', type=int, default=48)
    p.add_argument('--drop-rate', type=float, default=0.)
    p.add_argument('--bn-size', type=int, default=8)
    p.add_argument('--bottleneck', action='store_true', default=False)
    p.add_argument('--num-heads', type=int, default=4,
                   help='Transformer 瓶颈的注意力头数')
    p.add_argument('--no-positional-encoding', action='store_true', default=False,
                   help='关闭 Transformer 位置编码（仅用于消融实验）')
    p.add_argument('--attn-max-tokens', type=int, default=8192,
                   help='注意力 token 数上限，超过则跳过瓶颈层以免显存爆炸')

    # ---------------------------------------------------------------- 数据
    p.add_argument('--data-dir', type=str, default='./dataset')
    p.add_argument('--train-file', type=str, default=None)
    p.add_argument('--test-file', type=str, default=None)
    p.add_argument('--input-key', type=str, default=None,
                   help='显式指定 HDF5 输入数据集名')
    p.add_argument('--prefer-sensor-input', action='store_true', default=False)
    p.add_argument('--val-split', type=float, default=0.2,
                   help='训练集/测试集是同一个文件时，切出多少比例做验证（0 表示不切）')
    p.add_argument('--kle', type=int, default=4225, help='[上游遗留] KLE 项数')
    p.add_argument('--ntrain', type=int, default=512)
    p.add_argument('--ntest', type=int, default=500)
    p.add_argument('--num-workers', type=int, default=None,
                   help='DataLoader worker 数，缺省按是否有 GPU 自动选择')

    # ---------------------------------------------------------------- 维度
    p.add_argument('--ndim', type=int, default=default_ndim, choices=[2, 3],
                   help='数据维度：2 或 3')
    p.add_argument('--use-3d', action='store_true', default=False,
                   help='等价于 --ndim 3')
    p.add_argument('--nic', type=int, default=None, help='输入通道数（缺省自动推断）')
    p.add_argument('--noc', type=int, default=None, help='输出通道数（缺省自动推断）')
    p.add_argument('--image-size', nargs='+', type=int, default=None,
                   help='输出分辨率，2D 给 H W，3D 给 D H W（缺省自动推断）')

    # ---------------------------------------------------------------- 物理
    p.add_argument('--channel-order', type=str, default='pvv',
                   choices=['pvv', 'vvp'],
                   help="输出通道语义：pvv=(P,Vx,Vy)，vvp=(Vx,Vy,P)。"
                        "务必与数据集实际顺序一致，否则物理约束全部作用在错误通道上")
    p.add_argument('--domain-size', nargs='+', type=float, default=[1.0, 1.0],
                   help='计算域物理尺寸 (Ly Lx)，用来把像素间距换算成真实 dx/dy')
    p.add_argument('--rho', type=float, default=1.0, help='流体密度')
    p.add_argument('--nu', type=float, default=1e-2, help='运动粘度')
    p.add_argument('--physics-in-physical-units', type=_bool, default=True,
                   help='物理损失前先反归一化（关掉即退回旧版行为，仅供对比）')
    p.add_argument('--physics-warmup', type=float, default=0.1,
                   help='前多少比例的 epoch 完全不加物理约束')
    p.add_argument('--physics-loss-scaling', type=str, default='ema',
                   choices=['ema', 'none'],
                   help="物理项量级归一化。物理残差的绝对量级随 dx/dy 变化好几个"
                        "数量级，'ema' 用滑动估计把各项拉回 O(1)，权重才与量纲无关；"
                        "'none' 退回旧版的原始量级")
    p.add_argument('--div-weight', type=float, default=0.05, help='连续性方程权重')
    p.add_argument('--ns-weight', type=float, default=0.005, help='动量方程权重')
    p.add_argument('--smooth-weight', type=float, default=0.005, help='平滑项权重')

    # ---------------------------------------------------------------- 几何
    p.add_argument('--t-geometry', type=str, default=None,
                   help='T 型域几何（JSON 字符串或 JSON 文件路径），'
                        '缺省用 utils/geometry.py 的归一化默认值')
    p.add_argument('--no-mask', action='store_true', default=False,
                   help='关闭 T 型掩码，在整幅网格上训练与评估')

    # ---------------------------------------------------------------- 训练
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--test-batch-size', type=int, default=100)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--grad-clip', type=float, default=1.0,
                   help='梯度范数裁剪阈值，<=0 表示关闭')
    p.add_argument('--lr-patience', type=int, default=10,
                   help='ReduceLROnPlateau 的 patience（依据验证 RMSE）')

    # ---------------------------------------------------------------- 日志
    p.add_argument('--ckpt-epoch', type=int, default=None)
    p.add_argument('--ckpt-freq', type=int, default=25)
    p.add_argument('--log-freq', type=int, default=2)
    p.add_argument('--plot-freq', type=int, default=100)
    p.add_argument('--plot-fn', type=str, default='imshow',
                   choices=['contourf', 'imshow'],
                   help='contourf 模式下 matplotlib 会忽略 vmin/vmax，'
                        '需要真值/预测严格同色阶时请用 imshow')
    return p


def postprocess(args):
    """参数之间的联动与校验。不触碰文件系统。"""
    if args.use_3d:
        args.ndim = 3

    args.run_dir = '{}/{}'.format(args.exp_dir, args.exp_name)
    args.ckpt_dir = args.run_dir + '/checkpoints'

    if args.ckpt_freq <= 0:
        raise ValueError('--ckpt-freq 必须为正')
    if args.epochs % args.ckpt_freq != 0:
        # 旧版这里是 assert 直接崩；改成自动对齐并提示，避免 GUI 上误配就跑不动
        adjusted = max(1, args.epochs // args.ckpt_freq) * args.ckpt_freq
        print('[args] epochs({}) 不是 ckpt_freq({}) 的整数倍，最后一轮将强制存档'
              .format(args.epochs, args.ckpt_freq))
        args.last_ckpt_epoch = args.epochs
        del adjusted

    if args.image_size is not None:
        if args.ndim == 3 and len(args.image_size) == 2:
            args.image_size = [1] + list(args.image_size)
        if args.ndim == 3 and len(args.image_size) != 3:
            raise ValueError('3D 的 --image-size 需要 3 个整数 (D H W)')
        if args.ndim == 2 and len(args.image_size) not in (2, 3):
            raise ValueError('2D 的 --image-size 需要 2 个整数 (H W)')

    if len(args.domain_size) == 1:
        args.domain_size = [args.domain_size[0], args.domain_size[0]]
    if len(args.domain_size) != 2:
        raise ValueError('--domain-size 需要 1 或 2 个浮点数 (Ly Lx)')

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args


def parse_args(argv=None, default_ndim=2):
    return postprocess(build_parser(default_ndim).parse_args(argv))


def prepare_run_dir(args, write_args=True):
    """创建实验目录并落盘参数快照。由 train.py 显式调用。"""
    from utils.misc import mkdirs
    mkdirs([args.run_dir, args.ckpt_dir])
    if write_args and not args.post:
        with open(args.run_dir + '/args.txt', 'w', encoding='utf-8') as fh:
            json.dump(vars(args), fh, indent=4, ensure_ascii=False, default=str)
    return args


# 兼容旧的 `from args import args, device` 用法
class Parser(argparse.ArgumentParser):
    """保留旧类名，行为等价于 build_parser(2)。"""

    def __init__(self, default_ndim=2):
        self._default_ndim = default_ndim
        super(Parser, self).__init__(description='Dense Convolutional Encoder-Decoder')

    def parse(self, argv=None):
        return parse_args(argv, default_ndim=self._default_ndim)


def __getattr__(name):
    """惰性提供旧接口 `from args import args, device`。

    旧版在模块顶层就执行 `Parser().parse()` —— 光是 import 这个模块就会解析
    sys.argv、创建目录并写 args.txt，导致它无法被测试或被其它脚本安全 import。
    改成按需求值后，只有真正访问 `args` 时才解析命令行。
    """
    if name == 'args':
        value = parse_args(default_ndim=2)
    elif name == 'device':
        value = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
