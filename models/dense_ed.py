"""2D Dense Convolutional Encoder-Decoder（TransUNet 风格）。

参考: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

相对旧版的修正
--------------
1. **补上 U-Net skip connection**。旧版 2D 模型整个是一条 `nn.Sequential`，
   编码器特征从不送到解码器 —— README 却称其为 "TransUNet 架构"。缺 skip 对
   壁面边界层、分离点这类高频细节的重建非常吃亏。3D 版本来就有 skip，现在
   两边结构一致。
2. **输出尺寸兜底**。旧版只在奇数分辨率下输入输出等尺寸
   （实测 128 -> 129、256 -> 257），偶数网格会直接在 `F.mse_loss` 处崩掉；
   3D 版有 interpolate 兜底而 2D 版没有。现在统一到 `output_size`（缺省即输入
   尺寸）。
3. **Transformer 瓶颈补位置编码**（见 `models/attention.py`）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SpatialAttention, TransformerBottleneck


def activation(name, *args):
    """按名字返回激活函数。注意：旧版把 'relu' 映射到 SiLU，命名有误导性，
    这里让名字与实现一一对应，需要平滑二阶导时显式写 'silu'。"""
    if name in ('tanh', 'Tanh'):
        return nn.Tanh()
    if name in ('relu', 'ReLU'):
        return nn.ReLU(inplace=True)
    if name in ('silu', 'SiLU', 'swish'):
        return nn.SiLU(inplace=True)
    if name in ('lrelu', 'LReLU'):
        return nn.LeakyReLU(0.1, inplace=True)
    if name in ('sigmoid', 'Sigmoid'):
        return nn.Sigmoid()
    if name in ('softplus', 'Softplus'):
        return nn.Softplus(beta=4)
    raise ValueError('未知的激活函数 {!r}'.format(name))


class _DenseLayer(nn.Sequential):
    """dense block 内的单层。SiLU 保证二阶导连续，配合 PINN 使用。"""

    def __init__(self, in_features, growth_rate, drop_rate=0., bn_size=8,
                 bottleneck=False):
        super(_DenseLayer, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('act1', nn.SiLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, bn_size * growth_rate,
                                               kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('act2', nn.SiLU(inplace=True))
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=3, stride=1, padding=1,
                                               bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('act1', nn.SiLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                                               kernel_size=3, stride=1, padding=1,
                                               bias=False))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout2d(p=drop_rate))

    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        return torch.cat([x, y], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_features, growth_rate, drop_rate,
                 bn_size=4, bottleneck=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i + 1),
                            _DenseLayer(in_features + i * growth_rate, growth_rate,
                                        drop_rate=drop_rate, bn_size=bn_size,
                                        bottleneck=bottleneck))


class _Transition(nn.Sequential):
    """过渡层：下采样或上采样，同时压缩特征图数量。"""

    def __init__(self, in_features, out_features, down, bottleneck=True,
                 drop_rate=0):
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('act1', nn.SiLU(inplace=True))
        if down:
            if bottleneck:
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                                                   kernel_size=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('act2', nn.SiLU(inplace=True))
                self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                                   kernel_size=3, stride=2,
                                                   padding=1, bias=False))
            else:
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                                                   kernel_size=3, stride=2,
                                                   padding=1, bias=False))
            if drop_rate > 0:
                self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
        else:
            # 用 Upsample + Conv 代替 ConvTranspose：不产生棋盘伪影，
            # 且尺寸变换是纯 2 倍，不依赖 output_padding 的奇偶性技巧
            self.add_module('conv1', nn.Conv2d(in_features, out_features,
                                               kernel_size=1, bias=False))
            if drop_rate > 0:
                self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
            self.add_module('norm2', nn.BatchNorm2d(out_features))
            self.add_module('act2', nn.SiLU(inplace=True))
            self.add_module('upsample2', nn.Upsample(scale_factor=2,
                                                     mode='bilinear',
                                                     align_corners=False))
            self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                               kernel_size=3, stride=1,
                                               padding=1, bias=False))
            if drop_rate > 0:
                self.add_module('dropout2', nn.Dropout2d(p=drop_rate))


def last_decoding(in_features, out_channels, drop_rate=0., bias=False):
    """最后一级上采样，直接输出物理场。"""
    last_up = nn.Sequential()
    last_up.add_module('norm1', nn.BatchNorm2d(in_features))
    last_up.add_module('act1', nn.SiLU(inplace=True))
    last_up.add_module('conv1', nn.Conv2d(in_features, in_features // 2,
                                          kernel_size=1, bias=False))
    if drop_rate > 0.:
        last_up.add_module('dropout1', nn.Dropout2d(p=drop_rate))
    last_up.add_module('norm2', nn.BatchNorm2d(in_features // 2))
    last_up.add_module('act2', nn.SiLU(inplace=True))
    last_up.add_module('upsample2', nn.Upsample(scale_factor=2, mode='bilinear',
                                                align_corners=False))
    last_up.add_module('conv2', nn.Conv2d(in_features // 2, out_channels,
                                          kernel_size=3, stride=1, padding=1,
                                          bias=bias))
    return last_up


class DenseED(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, growth_rate=16,
                 init_features=48, bn_size=8, drop_rate=0, bottleneck=False,
                 out_activation=None, output_size=None, num_heads=4,
                 use_positional_encoding=True, attn_max_tokens=8192,
                 verbose=True):
        super(DenseED, self).__init__()
        blocks = list(blocks)
        if len(blocks) > 1 and len(blocks) % 2 == 0:
            raise ValueError('blocks 长度必须是奇数，当前为 {}'.format(len(blocks)))
        enc_block_layers = blocks[:len(blocks) // 2]
        dec_block_layers = blocks[len(blocks) // 2:]

        self.in_conv = nn.Conv2d(in_channels, init_features, kernel_size=7,
                                 stride=2, padding=3, bias=False)

        self.enc_blocks = nn.ModuleList()
        self.trans_downs = nn.ModuleList()
        self.skip_channels = []

        num_features = init_features
        for num_layers in enc_block_layers:
            self.enc_blocks.append(_DenseBlock(
                num_layers=num_layers, in_features=num_features, bn_size=bn_size,
                growth_rate=growth_rate, drop_rate=drop_rate, bottleneck=bottleneck))
            num_features += num_layers * growth_rate
            self.skip_channels.append(num_features)

            self.trans_downs.append(_Transition(
                in_features=num_features, out_features=num_features // 2,
                down=True, drop_rate=drop_rate))
            num_features //= 2

        self.transformer_bottleneck = TransformerBottleneck(
            channels=num_features, num_heads=num_heads,
            use_positional_encoding=use_positional_encoding,
            max_tokens=attn_max_tokens)

        self.dec_blocks = nn.ModuleList()
        self.trans_ups = nn.ModuleList()
        for i, num_layers in enumerate(dec_block_layers):
            if i == 0:
                in_features = num_features
            else:
                self.trans_ups.append(_Transition(
                    in_features=num_features, out_features=num_features // 2,
                    down=False, drop_rate=drop_rate))
                num_features //= 2
                in_features = num_features + self.skip_channels[-i]
            self.dec_blocks.append(_DenseBlock(
                num_layers=num_layers, in_features=in_features, bn_size=bn_size,
                growth_rate=growth_rate, drop_rate=drop_rate, bottleneck=bottleneck))
            num_features = in_features + num_layers * growth_rate

        self.spatial_attention = SpatialAttention(kernel_size=7, ndim=2)
        self.last_trans_up = last_decoding(num_features, out_channels,
                                           drop_rate=drop_rate)
        self.out_activation = activation(out_activation) \
            if out_activation is not None else None
        self.output_size = tuple(output_size) if output_size is not None else None

        if verbose:
            print('# params {}, # conv layers {}'.format(
                *self._num_parameters_convlayers()))

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        input_size = tuple(x.shape[-2:])
        x = self.in_conv(x)

        skips = []
        for enc_block, trans_down in zip(self.enc_blocks, self.trans_downs):
            x = enc_block(x)
            skips.append(x)
            x = trans_down(x)

        x = self.transformer_bottleneck(x)

        for i, dec_block in enumerate(self.dec_blocks):
            if i > 0:
                x = self.trans_ups[i - 1](x)
                skip = skips[-i]
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                                      align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        x = self.spatial_attention(x)
        x = self.last_trans_up(x)
        if self.out_activation is not None:
            x = self.out_activation(x)

        target = self.output_size if self.output_size is not None else input_size
        if tuple(x.shape[-2:]) != tuple(target):
            x = F.interpolate(x, size=tuple(target), mode='bilinear',
                              align_corners=False)
        return x

    # ---------------------------------------------------------------- 辅助
    def forward_test(self, x):
        print('input: {}'.format(tuple(x.shape)))
        out = self.forward(x)
        print('output: {}'.format(tuple(out.shape)))
        return out

    def _num_parameters_convlayers(self):
        n_params, n_conv_layers = 0, 0
        for name, param in self.named_parameters():
            if 'conv' in name:
                n_conv_layers += 1
            n_params += param.numel()
        return n_params, n_conv_layers

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            if isinstance(module, self.__class__):
                continue
            if callable(getattr(module, 'reset_parameters', None)):
                module.reset_parameters()
                if verbose:
                    print('Reset parameters in {}'.format(module))


if __name__ == '__main__':
    net = DenseED(1, 3, blocks=(3, 6, 3), growth_rate=16, init_features=48,
                  drop_rate=0, bn_size=8, bottleneck=False)
    for size in (65, 128, 137, 256):
        net.forward_test(torch.zeros(2, 1, size, size))
