"""
Dense Convolutional Encoder-Decoder Networks (3D variant)
"""

import torch
import torch.nn as nn

class _DenseLayer3D(nn.Sequential):
    def __init__(self, in_features, growth_rate, drop_rate=0., bn_size=8,
                 bottleneck=False):
        super(_DenseLayer3D, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm3d(in_features))
            self.add_module('relu1', nn.LeakyReLU(0.1, inplace=True))
            self.add_module('conv1', nn.Conv3d(in_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
            self.add_module('relu2', nn.LeakyReLU(0.1, inplace=True))
            self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                            kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm3d(in_features))
            self.add_module('relu1', nn.LeakyReLU(0.1, inplace=True))
            self.add_module('conv1', nn.Conv3d(in_features, growth_rate,
                            kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False))
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
            layer = _DenseLayer3D(in_features + i * growth_rate, growth_rate,
                                  drop_rate=drop_rate, bn_size=bn_size,
                                  bottleneck=bottleneck)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition3D(nn.Sequential):
    def __init__(self, in_features, out_features, down, bottleneck=True,
                 drop_rate=0):
        super(_Transition3D, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(in_features))
        self.add_module('relu1', nn.LeakyReLU(0.1, inplace=True))
        if down:
            if bottleneck:
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm3d(out_features))
                self.add_module('relu2', nn.LeakyReLU(0.1, inplace=True))
                self.add_module('conv2', nn.Conv3d(out_features, out_features,
                    kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout3d(p=drop_rate))
            else:
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                    kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))
        else:
            if bottleneck:
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))

                self.add_module('norm2', nn.BatchNorm3d(out_features))
                self.add_module('relu2', nn.LeakyReLU(0.1, inplace=True))
                self.add_module('upsample2', nn.Upsample(
                    scale_factor=(1, 2, 2), mode='trilinear', align_corners=True))
                self.add_module('conv2', nn.Conv3d(
                    out_features, out_features, kernel_size=(1,3,3),
                    stride=1, padding=(0,1,1), bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout3d(p=drop_rate))
            else:
                self.add_module('upsample1', nn.Upsample(
                    scale_factor=(1, 2, 2), mode='trilinear', align_corners=True))
                self.add_module('conv1', nn.Conv3d(
                    out_features, out_features, kernel_size=(1,3,3),
                    stride=1, padding=(0,1,1), bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))

def last_decoding(in_features, out_channels, kernel_size, stride, padding,
                  output_padding=0, bias=False, drop_rate=0.):
    last_up = nn.Sequential()
    last_up.add_module('norm1', nn.BatchNorm3d(in_features))
    last_up.add_module('relu1', nn.LeakyReLU(0.1, inplace=True))
    last_up.add_module('conv1', nn.Conv3d(in_features, in_features // 2,
                    kernel_size=1, stride=1, padding=0, bias=False))
    if drop_rate > 0.:
        last_up.add_module('dropout1', nn.Dropout3d(p=drop_rate))
    last_up.add_module('norm2', nn.BatchNorm3d(in_features // 2))
    last_up.add_module('relu2', nn.LeakyReLU(0.1, inplace=True))
    
    def _tuplize(x):
        if isinstance(x, int):
            return (1, x, x)
        return x
    k, s, p, op = _tuplize(kernel_size), _tuplize(stride), _tuplize(padding), _tuplize(output_padding)
    last_up.add_module('upsample2', nn.Upsample(
        scale_factor=s, mode='trilinear', align_corners=True))
    last_up.add_module('conv2', nn.Conv3d(
        in_features // 2, out_channels, kernel_size=k,
        stride=1, padding=p, bias=bias))
    return last_up

class DenseED3D(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, growth_rate=16,
                 init_features=48, bn_size=8, drop_rate=0, bottleneck=False,
                 out_activation=None, output_size=None):
        super(DenseED3D, self).__init__()
        if len(blocks) > 1 and len(blocks) % 2 == 0:
            raise ValueError('length of blocks must be an odd number, but got {}'
                            .format(len(blocks)))
        enc_block_layers = blocks[: len(blocks) // 2]
        dec_block_layers = blocks[len(blocks) // 2:]

        self.in_conv = nn.Conv3d(in_channels, init_features,
                      kernel_size=(1,7,7), stride=(1,2,2), padding=(0,2,2), bias=False)

        self.enc_blocks = nn.ModuleList()
        self.trans_downs = nn.ModuleList()
        self.skip_channels = []

        num_features = init_features
        for num_layers in enc_block_layers:
            block = _DenseBlock3D(num_layers=num_layers,
                                  in_features=num_features,
                                  bn_size=bn_size,
                                  growth_rate=growth_rate,
                                  drop_rate=drop_rate,
                                  bottleneck=bottleneck)
            self.enc_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            self.skip_channels.append(num_features)

            trans_down = _Transition3D(in_features=num_features,
                                     out_features=num_features // 2,
                                     down=True,
                                     drop_rate=drop_rate)
            self.trans_downs.append(trans_down)
            num_features = num_features // 2

        self.dec_blocks = nn.ModuleList()
        self.trans_ups = nn.ModuleList()

        for i, num_layers in enumerate(dec_block_layers):
            if i == 0:
                in_features = num_features
            else:
                trans_up = _Transition3D(in_features=num_features,
                                        out_features=num_features // 2,
                                        down=False,
                                        drop_rate=drop_rate)
                self.trans_ups.append(trans_up)
                num_features = num_features // 2
                skip_features = self.skip_channels[-i]
                in_features = num_features + skip_features

            block = _DenseBlock3D(num_layers=num_layers,
                                  in_features=in_features,
                                  bn_size=bn_size,
                                  growth_rate=growth_rate,
                                  drop_rate=drop_rate,
                                  bottleneck=bottleneck)
            self.dec_blocks.append(block)
            num_features = in_features + num_layers * growth_rate

        self.last_trans_up = last_decoding(num_features, out_channels,
                            kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1),
                            output_padding=(0,1,1), bias=False, drop_rate=drop_rate)
        if out_activation is not None:
            if out_activation in ['tanh', 'Tanh']:
                self.out_activation = nn.Tanh()
            elif out_activation in ['relu', 'ReLU']:
                self.out_activation = nn.LeakyReLU(0.1, inplace=True)
            elif out_activation in ['sigmoid', 'Sigmoid']:
                self.out_activation = nn.Sigmoid()
            else:
                self.out_activation = None
        else:
            self.out_activation = None

        self.output_size = output_size
        print('# params {}, # conv layers {}'.format(
            *self._num_parameters_convlayers()))

    def forward(self, x):
        input_size = x.shape[-3:]
        x = self.in_conv(x)

        skip_features = []
        for enc_block, trans_down in zip(self.enc_blocks, self.trans_downs):
            x = enc_block(x)
            skip_features.append(x)
            x = trans_down(x)

        for i, dec_block in enumerate(self.dec_blocks):
            if i > 0:
                x = self.trans_ups[i - 1](x)
                skip = skip_features[-i]
                if x.shape[-3:] != skip.shape[-3:]:
                    x = nn.functional.interpolate(
                        x, size=skip.shape[-3:],
                        mode='trilinear', align_corners=False
                    )
                x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        x = self.last_trans_up(x)
        if self.out_activation is not None:
            x = self.out_activation(x)

        target_size = self.output_size if self.output_size is not None else input_size
        if x.shape[-3:] != target_size:
            x = nn.functional.interpolate(
                x, size=target_size,
                mode='trilinear', align_corners=False
            )
        return x

    def forward_test(self, x):
        pass # 省略测试代码，保持主体一致

    def _num_parameters_convlayers(self):
        n_params, n_conv_layers = 0, 0
        for name, param in self.named_parameters():
            if 'conv' in name:
                n_conv_layers += 1
            n_params += param.numel()
        return n_params, n_conv_layers

DenseED = DenseED3D