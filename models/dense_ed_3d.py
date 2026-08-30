"""3D Dense Convolutional Encoder-Decoder（体素版）。

相对旧版的修正
--------------
* **激活函数全部换成 SiLU**。旧版 3D 分支仍是 `LeakyReLU(0.1)`，一阶导在 0 处
  就不连续、二阶导处处为 0 —— 恰好破坏了 PINN 需要的高阶光滑性（README 声称
  "全部替换为 SiLU"，其实只有 2D 版改了）。
* **Transformer 瓶颈补 3D sin-cos 位置编码**，并对 token 数设上限，避免
  D*H*W 过大时 O(N^2) 注意力直接爆显存。
* 与 2D 版共用 `models/attention.py`，两边行为保持一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SpatialAttention, TransformerBottleneck


class _DenseLayer3D(nn.Sequential):
    def __init__(self, in_features, growth_rate, drop_rate=0., bn_size=8,
                 bottleneck=False):
        super(_DenseLayer3D, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm3d(in_features))
            self.add_module('act1', nn.SiLU(inplace=True))
            self.add_module('conv1', nn.Conv3d(in_features, bn_size * growth_rate,
                                               kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
            self.add_module('act2', nn.SiLU(inplace=True))
            self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                               kernel_size=(1, 3, 3), stride=1,
                                               padding=(0, 1, 1), bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm3d(in_features))
            self.add_module('act1', nn.SiLU(inplace=True))
            self.add_module('conv1', nn.Conv3d(in_features, growth_rate,
                                               kernel_size=(1, 3, 3), stride=1,
                                               padding=(0, 1, 1), bias=False))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout3d(p=drop_rate))

    def forward(self, x):
        y = super(_DenseLayer3D, self).forward(x)
        return torch.cat([x, y], 1)


class _DenseBlock3D(nn.Sequential):
    def __init__(self, num_layers, in_features, growth_rate, drop_rate,
                 bn_size=4, bottleneck=False):
        super(_DenseBlock3D, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i + 1),
                            _DenseLayer3D(in_features + i * growth_rate, growth_rate,
                                          drop_rate=drop_rate, bn_size=bn_size,
                                          bottleneck=bottleneck))


class _Transition3D(nn.Sequential):
    def __init__(self, in_features, out_features, down, bottleneck=True,
                 drop_rate=0):
        super(_Transition3D, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(in_features))
        self.add_module('act1', nn.SiLU(inplace=True))
        if down:
            if bottleneck:
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                                                   kernel_size=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm3d(out_features))
                self.add_module('act2', nn.SiLU(inplace=True))
                self.add_module('conv2', nn.Conv3d(out_features, out_features,
                                                   kernel_size=(1, 3, 3),
                                                   stride=(1, 2, 2),
                                                   padding=(0, 1, 1), bias=False))
            else:
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                                                   kernel_size=(1, 3, 3),
                                                   stride=(1, 2, 2),
                                                   padding=(0, 1, 1), bias=False))
            if drop_rate > 0:
                self.add_module('dropout2', nn.Dropout3d(p=drop_rate))
        else:
            self.add_module('conv1', nn.Conv3d(in_features, out_features,
                                               kernel_size=1, bias=False))
            if drop_rate > 0:
                self.add_module('dropout1', nn.Dropout3d(p=drop_rate))
            self.add_module('norm2', nn.BatchNorm3d(out_features))
            self.add_module('act2', nn.SiLU(inplace=True))
            self.add_module('upsample2', nn.Upsample(scale_factor=(1, 2, 2),
                                                     mode='trilinear',
                                                     align_corners=False))
            self.add_module('conv2', nn.Conv3d(out_features, out_features,
                                               kernel_size=(1, 3, 3), stride=1,
                                               padding=(0, 1, 1), bias=False))
            if drop_rate > 0:
                self.add_module('dropout2', nn.Dropout3d(p=drop_rate))


def last_decoding(in_features, out_channels, drop_rate=0., bias=False):
    last_up = nn.Sequential()
    last_up.add_module('norm1', nn.BatchNorm3d(in_features))
    last_up.add_module('act1', nn.SiLU(inplace=True))
    last_up.add_module('conv1', nn.Conv3d(in_features, in_features // 2,
                                          kernel_size=1, bias=False))
    if drop_rate > 0.:
        last_up.add_module('dropout1', nn.Dropout3d(p=drop_rate))
    last_up.add_module('norm2', nn.BatchNorm3d(in_features // 2))
    last_up.add_module('act2', nn.SiLU(inplace=True))
    last_up.add_module('upsample2', nn.Upsample(scale_factor=(1, 2, 2),
                                                mode='trilinear',
                                                align_corners=False))
    last_up.add_module('conv2', nn.Conv3d(in_features // 2, out_channels,
                                          kernel_size=(1, 3, 3), stride=1,
                                          padding=(0, 1, 1), bias=bias))
    return last_up


class DenseED3D(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, growth_rate=16,
                 init_features=48, bn_size=8, drop_rate=0, bottleneck=False,
                 out_activation=None, output_size=None, num_heads=4,
                 use_positional_encoding=True, attn_max_tokens=8192,
                 verbose=True):
        super(DenseED3D, self).__init__()
        blocks = list(blocks)
        if len(blocks) > 1 and len(blocks) % 2 == 0:
            raise ValueError('blocks 长度必须是奇数，当前为 {}'.format(len(blocks)))
        enc_block_layers = blocks[:len(blocks) // 2]
        dec_block_layers = blocks[len(blocks) // 2:]

        self.in_conv = nn.Conv3d(in_channels, init_features, kernel_size=(1, 7, 7),
                                 stride=(1, 2, 2), padding=(0, 3, 3), bias=False)

        self.enc_blocks = nn.ModuleList()
        self.trans_downs = nn.ModuleList()
        self.skip_channels = []

        num_features = init_features
        for num_layers in enc_block_layers:
            self.enc_blocks.append(_DenseBlock3D(
                num_layers=num_layers, in_features=num_features, bn_size=bn_size,
                growth_rate=growth_rate, drop_rate=drop_rate, bottleneck=bottleneck))
            num_features += num_layers * growth_rate
            self.skip_channels.append(num_features)

            self.trans_downs.append(_Transition3D(
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
                self.trans_ups.append(_Transition3D(
                    in_features=num_features, out_features=num_features // 2,
                    down=False, drop_rate=drop_rate))
                num_features //= 2
                in_features = num_features + self.skip_channels[-i]
            self.dec_blocks.append(_DenseBlock3D(
                num_layers=num_layers, in_features=in_features, bn_size=bn_size,
                growth_rate=growth_rate, drop_rate=drop_rate, bottleneck=bottleneck))
            num_features = in_features + num_layers * growth_rate

        self.spatial_attention = SpatialAttention(kernel_size=7, ndim=3)
        self.last_trans_up = last_decoding(num_features, out_channels,
                                           drop_rate=drop_rate)

        if out_activation in ('tanh', 'Tanh'):
            self.out_activation = nn.Tanh()
        elif out_activation in ('silu', 'SiLU', 'swish'):
            self.out_activation = nn.SiLU(inplace=True)
        elif out_activation in ('relu', 'ReLU'):
            self.out_activation = nn.ReLU(inplace=True)
        elif out_activation in ('sigmoid', 'Sigmoid'):
            self.out_activation = nn.Sigmoid()
        else:
            self.out_activation = None

        self.output_size = tuple(output_size) if output_size is not None else None
        if verbose:
            print('# params {}, # conv layers {}'.format(
                *self._num_parameters_convlayers()))

    def forward(self, x):
        input_size = tuple(x.shape[-3:])
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
                if x.shape[-3:] != skip.shape[-3:]:
                    x = F.interpolate(x, size=skip.shape[-3:], mode='trilinear',
                                      align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        x = self.spatial_attention(x)
        x = self.last_trans_up(x)
        if self.out_activation is not None:
            x = self.out_activation(x)

        target = self.output_size if self.output_size is not None else input_size
        if tuple(x.shape[-3:]) != tuple(target):
            x = F.interpolate(x, size=tuple(target), mode='trilinear',
                              align_corners=False)
        return x

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


DenseED = DenseED3D


if __name__ == '__main__':
    net = DenseED3D(1, 3, blocks=(3, 6, 3), growth_rate=8, init_features=16)
    net.forward_test(torch.zeros(1, 1, 2, 65, 65))
