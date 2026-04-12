"""
Dense Convolutional Encoder-Decoder Networks

Reference:
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

"""

import torch
import torch.nn as nn

# ========================================================
# [新增] 空间注意力模块：用于聚焦 T 型管道的高梯度/流场突变区域
# ========================================================
class SpatialAttention(nn.Module):
    """提取空间关键特征的注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上进行最大池化和平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 生成空间注意力权重图 (0~1之间)
        attn_map = self.sigmoid(self.conv(x_cat))
        # 将权重施加回原特征图
        return x * attn_map
# ========================================================

# ========================================================
# [新增] 结合 PINN 的 Transformer 瓶颈层模块：捕获流体全局连通性
# ========================================================
class TransformerBottleneck(nn.Module):
    """
    在编解码器结构最深处提取流体的全局拓扑依赖 (Global Dependency)
    """
    def __init__(self, channels, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super(TransformerBottleneck, self).__init__()
        # 1. 自注意力机制前的标准层归一化
        self.norm1 = nn.LayerNorm(channels)
        # 多头自注意力机制 (batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, 
                                          dropout=dropout, batch_first=True)
        
        # 2. 前馈神经网络 (FFN / MLP) 
        self.norm2 = nn.LayerNorm(channels)
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden_dim),
            nn.SiLU(), 
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 1. 展平并对换维度以适应 Transformer: [B, C, H, W] -> [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)
        
        # 2. 计算多头自注意力并执行残差连接 (Residual)
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out  
        
        # 3. 计算前馈 MLP 层并执行残差连接
        x_norm2 = self.norm2(x_flat)
        mlp_out = self.mlp(x_norm2)
        x_flat = x_flat + mlp_out   
        
        # 4. 将序列平滑地还原回图像特征: [B, H*W, C] -> [B, C, H, W]
        out = x_flat.transpose(1, 2).contiguous().view(B, C, H, W)
        return out
# ========================================================

class _DenseLayer(nn.Sequential):
    """One dense layer within dense block, with bottleneck design.

    Args:
        in_features (int):
        growth_rate (int): # out feature maps of every dense layer
        drop_rate (float): 
        bn_size (int): Specifies maximum # features is `bn_size` * `growth_rate`
        bottleneck (bool, False): If True, enable bottleneck design
    """
    def __init__(self, in_features, growth_rate, drop_rate=0., bn_size=8,
                 bottleneck=False):
        super(_DenseLayer, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.SiLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('relu2', nn.SiLU(inplace=True))
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.SiLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
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
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate,
                                drop_rate=drop_rate, bn_size=bn_size,
                                bottleneck=bottleneck)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, in_features, out_features, down, bottleneck=True, 
                 drop_rate=0):
        """Transition layer, either downsampling or upsampling, both reduce
        number of feature maps, i.e. `out_features` should be less than 
        `in_features`.
        """
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('relu1', nn.SiLU(inplace=True))
        if down:
            # half feature resolution, reduce # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.SiLU(inplace=True))
                # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
                # not using pooling, fully convolutional...
                self.add_module('conv2', nn.Conv2d(out_features, out_features,
                    kernel_size=3, stride=2, padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=3, stride=2, padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
        else:
            # transition up, increase feature resolution, half # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', nn.Conv2d(in_features, out_features,
                    kernel_size=1, stride=1, padding=0, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.SiLU(inplace=True))
                # output_padding=0, or 1 depends on the image size
                # if image size is of the power of 2, then 1 is good
                self.add_module('convT2', nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('convT1', nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=3, stride=2,
                    padding=1, output_padding=1, bias=False))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))


def last_decoding(in_features, out_channels, kernel_size, stride, padding, 
                  output_padding=0, bias=False, drop_rate=0.):
    """Last transition up layer, which outputs directly the predictions.
    """
    last_up = nn.Sequential()
    last_up.add_module('norm1', nn.BatchNorm2d(in_features))
    last_up.add_module('relu1', nn.SiLU(True))
    last_up.add_module('conv1', nn.Conv2d(in_features, in_features // 2, 
                    kernel_size=1, stride=1, padding=0, bias=False))
    if drop_rate > 0.:
        last_up.add_module('dropout1', nn.Dropout2d(p=drop_rate))
    last_up.add_module('norm2', nn.BatchNorm2d(in_features // 2))
    last_up.add_module('relu2', nn.SiLU(True))
    last_up.add_module('convT2', nn.ConvTranspose2d(in_features // 2, 
                       out_channels, kernel_size=kernel_size, stride=stride, 
                       padding=padding, output_padding=output_padding, bias=bias))
    return last_up


def activation(name, *args):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.SiLU(inplace=True)
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
                 out_activation=None, output_size=None):
        super(DenseED, self).__init__()
        if len(blocks) > 1 and len(blocks) % 2 == 0:
            raise ValueError('length of blocks must be an odd number, but got {}'
                            .format(len(blocks)))
        enc_block_layers = blocks[: len(blocks) // 2]
        dec_block_layers = blocks[len(blocks) // 2:]

        self.features = nn.Sequential()

        # First convolution, half image size ================
        self.features.add_module('In_conv', nn.Conv2d(in_channels, init_features, 
                              kernel_size=7, stride=2, padding=2, bias=False))
        # Encoding / transition down ================
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
            
        # ===============================================================
        # [新增] 在降维最深处、受压最极致的地方插入 Transformer (Bottleneck)
        self.features.add_module('TransformerBottleneck', 
                                 TransformerBottleneck(channels=num_features, num_heads=4))
        # ===============================================================

        # Decoding / transition up ==============
        for i, num_layers in enumerate(dec_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, 
                                growth_rate=growth_rate,
                                drop_rate=drop_rate, 
                                bottleneck=bottleneck)
            self.features.add_module('DecBlock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            # the last decoding layer has different convT parameters
            if i < len(dec_block_layers) - 1:
                trans_up = _Transition(in_features=num_features,
                                    out_features=num_features // 2,
                                    down=False, 
                                    drop_rate=drop_rate)
                self.features.add_module('TransUp%d' % (i + 1), trans_up)
                num_features = num_features // 2
        
        # ===============================================================
        # [新增] 在深层特征图映射到最后物理场输出前，无缝插入空间注意力层！
        self.features.add_module('SpatialAttention', SpatialAttention(kernel_size=7))
        # ===============================================================

        # The last decoding layer =======
        last_trans_up = last_decoding(num_features, out_channels, 
                            kernel_size=4, stride=2, padding=1, 
                            output_padding=1, bias=False, drop_rate=drop_rate)
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
            # pass self, otherwise infinite loop
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
    x = torch.Tensor(16, 1, 65, 65)
    dense_ed.forward_test(x)
    print(dense_ed._num_parameters_convlayers())