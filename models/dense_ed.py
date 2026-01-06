"""
Dense Convolutional Encoder-Decoder Networks (3D Version)
Reference:
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
"""
import torch
import torch.nn as nn


class _DenseLayer(nn.Sequential):
    """One dense layer within dense block, with bottleneck design. (3D版)
    Args:
        in_features (int):
        growth_rate (int): # out feature maps of every dense layer
        drop_rate (float): 
        bn_size (int): Specifies maximum # features is `bn_size` * 
            `growth_rate`
        bottleneck (bool, False): If True, enable bottleneck design
    """
    def __init__(self, in_features, growth_rate, drop_rate=0., bn_size=8,
                 bottleneck=False):
        super(_DenseLayer, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm3d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv3d(in_features, bn_size *
                            growth_rate, kernel_size=(1,1,1), stride=(1,1,1), bias=False))
            self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                            kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm3d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv3d(in_features, growth_rate,
                            kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout3d(p=drop_rate))
        
    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        return torch.cat([x, y], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_features, growth_rate, drop_rate,
                 bn_size=4, bottleneck=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate,
                                drop_rate=drop_rate, bn_size=bn_size,
                                bottleneck=bottleneck)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, in_features, out_features, down, bottleneck=True, 
                 drop_rate=0):
        """Transition layer, either downsampling or upsampling (3D版)
        Args:
            in_features (int):
            out_features (int):
            down (bool): If True, downsampling, else upsampling
            bottleneck (bool, True): If True, enable bottleneck design
            drop_rate (float, 0.):
        """
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(in_features))  # BatchNorm2d → BatchNorm3d
        self.add_module('relu1', nn.ReLU(inplace=True))
        if down:
            if bottleneck:
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                    kernel_size=(1,1,1), stride=(1,1,1), padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))  # Dropout2d → Dropout3d
                self.add_module('norm2', nn.BatchNorm3d(out_features))  # BatchNorm2d → BatchNorm3d
                self.add_module('relu2', nn.ReLU(inplace=True))
                self.add_module('conv2', nn.Conv3d(out_features, out_features,
                    kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout3d(p=drop_rate))
            else:
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                    kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))
        else:
            if bottleneck:
                self.add_module('conv1', nn.Conv3d(in_features, out_features,
                    kernel_size=(1,1,1), stride=(1,1,1), padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm3d(out_features))  # BatchNorm2d → BatchNorm3d
                self.add_module('relu2', nn.ReLU(inplace=True))
                self.add_module('convT2', nn.ConvTranspose3d(
                    out_features, out_features, kernel_size=(3,3,3), stride=(2,2,2),
                    padding=(1,1,1), output_padding=(1,1,1), bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout3d(p=drop_rate))
            else:
                self.add_module('convT1', nn.ConvTranspose3d(
                    out_features, out_features, kernel_size=(3,3,3), stride=(2,2,2),
                    padding=(1,1,1), output_padding=(1,1,1), bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout3d(p=drop_rate))


def last_decoding(in_features, out_channels, kernel_size, stride, padding, 
                  output_padding=0, bias=False, drop_rate=0.):
    """Last transition up layer, which outputs directly the predictions. (3D版)
    """
    last_up = nn.Sequential()
    last_up.add_module('norm1', nn.BatchNorm3d(in_features))  # BatchNorm2d → BatchNorm3d
    last_up.add_module('relu1', nn.ReLU(True))
    last_up.add_module('conv1', nn.Conv3d(in_features, in_features // 2, 
                    kernel_size=(1,1,1), stride=(1,1,1), padding=0, bias=False))
    if drop_rate > 0.:
        last_up.add_module('dropout1', nn.Dropout3d(p=drop_rate))  # Dropout2d → Dropout3d
    last_up.add_module('norm2', nn.BatchNorm3d(in_features // 2))  # BatchNorm2d → BatchNorm3d
    last_up.add_module('relu2', nn.ReLU(True))
    last_up.add_module('convT2', nn.ConvTranspose3d(in_features // 2, 
                       out_channels, kernel_size=kernel_size if isinstance(kernel_size, tuple) else (kernel_size,kernel_size,kernel_size), 
                       stride=stride if isinstance(stride, tuple) else (stride,stride,stride), 
                       padding=padding if isinstance(padding, tuple) else (padding,padding,padding), 
                       output_padding=output_padding if isinstance(output_padding, tuple) else (output_padding,output_padding,output_padding), 
                       bias=bias))
    return last_up


def activation(name, *args):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    else:
        raise ValueError('Unknown activation function')


class DenseED(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, growth_rate=16,
                 init_features=48, bn_size=8, drop_rate=0, bottleneck=False,
                 out_activation=None):
        """Dense Convolutional Encoder-Decoder Networks (3D版)
        Args:
            in_channels (int): number of input channels (also include time if
                time enters in the input)
            out_channels (int): number of output channels
            blocks (list-like): A list (of odd size) of integers
            growth_rate (int): K
            init_features (int): the number of feature maps after the first
                conv layer
            bn_size: bottleneck size for number of feature maps
            bottleneck (bool): use bottleneck for dense block or not
            drop_rate (float): dropout rate
            out_activation: Output activation function, choices=[None, 'tanh',
                'sigmoid', 'softplus']
        """
        super(DenseED, self).__init__()
        if len(blocks) > 1 and len(blocks) % 2 == 0:
            raise ValueError('length of blocks must be an odd number, but got {}'
                            .format(len(blocks)))
        enc_block_layers = blocks[: len(blocks) // 2]
        dec_block_layers = blocks[len(blocks) // 2:]
        self.features = nn.Sequential()
        self.features.add_module('In_conv', nn.Conv3d(in_channels, init_features, 
                              kernel_size=(7,7,7), stride=(2,2,2), padding=(2,2,2), bias=False))
        
        num_features = init_features
        for i, num_layers in enumerate(enc_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, 
                                growth_rate=growth_rate,
                                drop_rate=drop_rate, 
                                bottleneck=bottleneck)
            self.features.add_module('EncBlock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            trans_down = _Transition(in_features=num_features,
                                     out_features=num_features // 2,
                                     down=True, 
                                     drop_rate=drop_rate)
            self.features.add_module('TransDown%d' % (i + 1), trans_down)
            num_features = num_features // 2
        
        for i, num_layers in enumerate(dec_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, 
                                growth_rate=growth_rate,
                                drop_rate=drop_rate, 
                                bottleneck=bottleneck)
            self.features.add_module('DecBlock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            if i < len(dec_block_layers) - 1:
                trans_up = _Transition(in_features=num_features,
                                    out_features=num_features // 2,
                                    down=False, 
                                    drop_rate=drop_rate)
                self.features.add_module('TransUp%d' % (i + 1), trans_up)
                num_features = num_features // 2
        
        last_trans_up = last_decoding(num_features, out_channels, 
                            kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), 
                            output_padding=(1,1,1), bias=False, drop_rate=drop_rate)
        self.features.add_module('LastTransUp', last_trans_up)
        if out_activation is not None:
            self.features.add_module(out_activation, activation(out_activation))
        
        print('# params {}, # conv layers {}'.format(
            *self._num_parameters_convlayers()))

    def forward(self, x):
        return self.features(x)

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

    def _count_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(param.numel())
            n_params += param.numel()
            print('num of parameters so far: {}'.format(n_params))

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
    dense_ed = DenseED(1, 3, blocks=(3, 6, 3), growth_rate=16, 
                      init_features=48, drop_rate=0, bn_size=8, 
                      bottleneck=False, out_activation='Tanh')
    print(dense_ed)
    x = torch.Tensor(16, 1, 65, 65, 65)
    dense_ed.forward_test(x)
    print(dense_ed._num_parameters_convlayers())