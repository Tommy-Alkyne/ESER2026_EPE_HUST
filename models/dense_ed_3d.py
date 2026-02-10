
"""
Dense Convolutional Encoder-Decoder Networks (3D variant)

Adapted from dense_ed.py to work with volumetric inputs where depth may be 1.
Key changes:
- Conv2d/ConvTranspose2d -> Conv3d/ConvTranspose3d
- BatchNorm2d -> BatchNorm3d
- Dropout2d -> Dropout3d
- Kernel/stride/padding adjusted to (1, k, k) when we don't want to downsample depth

Expected input shape: (N, C_in, D, H, W)
Expected output shape: (N, C_out, D, H, W)

This implementation preserves the depth dimension (assumes small D, often 1)
"""

import torch
import torch.nn as nn


class _DenseLayer3D(nn.Sequential):
    def __init__(self, in_features, growth_rate, drop_rate=0., bn_size=8,
                 bottleneck=False):
        super(_DenseLayer3D, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm3d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv3d(in_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=True))
            # keep depth kernel 1 to avoid shrinking depth
            self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                            kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm3d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
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
        self.add_module('relu1', nn.ReLU(inplace=True))
        if down:
            if bottleneck:
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm3d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # downsample H,W by 2, keep depth unchanged (stride 1 on depth)
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
            # transition up: upsample H,W by factor 2, keep depth unchanged
            if bottleneck:
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))

                self.add_module('norm2', nn.BatchNorm3d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
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
    last_up.add_module('relu1', nn.ReLU(True))
    last_up.add_module('conv1', nn.Conv3d(in_features, in_features // 2,
                    kernel_size=1, stride=1, padding=0, bias=False))
    if drop_rate > 0.:
        last_up.add_module('dropout1', nn.Dropout3d(p=drop_rate))
    last_up.add_module('norm2', nn.BatchNorm3d(in_features // 2))
    last_up.add_module('relu2', nn.ReLU(True))
    # kernel_size/stride/padding are expected to be 3-tuples; allow scalar for convenience
    def _tuplize(x):
        if isinstance(x, int):
            return (1, x, x)
        return x
    k = _tuplize(kernel_size)
    s = _tuplize(stride)
    p = _tuplize(padding)
    op = _tuplize(output_padding)
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

        self.features = nn.Sequential()

        # First conv: keep depth kernel 1 so depth dimension unchanged
        self.features.add_module('In_conv', nn.Conv3d(in_channels, init_features,
                      kernel_size=(1,7,7), stride=(1,2,2), padding=(0,2,2), bias=False))
        num_features = init_features
        for i, num_layers in enumerate(enc_block_layers):
            block = _DenseBlock3D(num_layers=num_layers,
                                  in_features=num_features,
                                  bn_size=bn_size,
                                  growth_rate=growth_rate,
                                  drop_rate=drop_rate,
                                  bottleneck=bottleneck)
            self.features.add_module('EncBlock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            trans_down = _Transition3D(in_features=num_features,
                                     out_features=num_features // 2,
                                     down=True,
                                     drop_rate=drop_rate)
            self.features.add_module('TransDown%d' % (i + 1), trans_down)
            num_features = num_features // 2

        for i, num_layers in enumerate(dec_block_layers):
            block = _DenseBlock3D(num_layers=num_layers,
                                  in_features=num_features,
                                  bn_size=bn_size,
                                  growth_rate=growth_rate,
                                  drop_rate=drop_rate,
                                  bottleneck=bottleneck)
            self.features.add_module('DecBlock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            if i < len(dec_block_layers) - 1:
                trans_up = _Transition3D(in_features=num_features,
                                        out_features=num_features // 2,
                                        down=False,
                                        drop_rate=drop_rate)
                self.features.add_module('TransUp%d' % (i + 1), trans_up)
                num_features = num_features // 2

        last_trans_up = last_decoding(num_features, out_channels,
                            kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1),
                            output_padding=(0,1,1), bias=False, drop_rate=drop_rate)
        self.features.add_module('LastTransUp', last_trans_up)
        if out_activation is not None:
            if out_activation in ['tanh', 'Tanh']:
                self.features.add_module(out_activation, nn.Tanh())
            elif out_activation in ['relu', 'ReLU']:
                self.features.add_module(out_activation, nn.ReLU(inplace=True))
            elif out_activation in ['sigmoid', 'Sigmoid']:
                self.features.add_module(out_activation, nn.Sigmoid())

        # optionally align output size to target spatial size
        self.output_size = output_size

        print('# params {}, # conv layers {}'.format(
            *self._num_parameters_convlayers()))

    def forward(self, x):
        x = self.features(x)
        if self.output_size is not None:
            d_out, h_out, w_out = self.output_size
            if x.shape[-3:] != (d_out, h_out, w_out):
                x = nn.functional.interpolate(
                    x, size=(d_out, h_out, w_out),
                    mode='trilinear', align_corners=False
                )
        return x

    def forward_test(self, x):
        print('input: {}'.format(x.data.size()))
        for name, module in self.features._modules.items():
            x = module(x)
            print('{}: {}'.format(name, x.data.size()))
        return x

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
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))


if __name__ == '__main__':
    model = DenseED3D(1, 3, blocks=(3, 6, 3), growth_rate=16,
                      init_features=48, drop_rate=0, bn_size=8,
                      bottleneck=False, out_activation='Tanh')
    print(model)
    x = torch.Tensor(16, 1, 1, 65, 65)
    model.forward_test(x)
    print(model._num_parameters_convlayers())

# Backwards-compatible alias: import as `DenseED` from other scripts
DenseED = DenseED3D
