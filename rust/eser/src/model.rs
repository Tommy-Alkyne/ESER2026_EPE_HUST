//! 2D Dense Convolutional Encoder-Decoder（TransUNet 风格）。移植自 `models/dense_ed.py`。
//!
//! 继承 Python 版相对旧版的三处修正：
//!
//! 1. **有 U-Net skip connection**。旧版 2D 模型整个是一条 Sequential，编码器特征
//!    从不送到解码器，却被 README 称作 TransUNet。缺 skip 对壁面边界层、分离点这类
//!    高频细节的重建非常吃亏。
//! 2. **输出尺寸兜底**。旧版只在奇数分辨率下输入输出等尺寸（实测 128 -> 129），
//!    偶数网格直接在损失处崩掉。
//! 3. **Transformer 瓶颈补了位置编码**（见 [`crate::attention`]）。
//!
//! 上采样一律走 [`crate::interp`] 自己实现的 `align_corners=False` 双线性插值——
//! Burn 自带的 `interpolate` 用的是另一套坐标映射，实测 2 倍上采样最大偏差 0.35。

use burn::module::{Ignored, Module};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, PaddingConfig2d};
use burn::tensor::activation::{leaky_relu, relu, sigmoid, silu, softplus, tanh};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::attention::{SpatialAttention2d, TransformerBottleneck};
use crate::interp::resize_bilinear;
use crate::store::{join, LoadState, StateDict};

/// 输出层激活。Python 版把 `'relu'` 映射到 SiLU，命名有误导性；这里名字与实现
/// 一一对应，需要平滑二阶导时显式选 [`Activation::Silu`]。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Tanh,
    Relu,
    Silu,
    LeakyRelu,
    Sigmoid,
    Softplus,
}

impl Activation {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "tanh" => Ok(Self::Tanh),
            "relu" => Ok(Self::Relu),
            "silu" | "swish" => Ok(Self::Silu),
            "lrelu" => Ok(Self::LeakyRelu),
            "sigmoid" => Ok(Self::Sigmoid),
            "softplus" => Ok(Self::Softplus),
            other => Err(format!("未知的激活函数 {:?}", other)),
        }
    }

    pub fn apply<B: Backend, const D: usize>(self, x: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Self::Tanh => tanh(x),
            Self::Relu => relu(x),
            Self::Silu => silu(x),
            Self::LeakyRelu => leaky_relu(x, 0.1),
            Self::Sigmoid => sigmoid(x),
            Self::Softplus => softplus(x, 4.0),
        }
    }
}

fn conv2d<B: Backend>(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    device: &B::Device,
) -> Conv2d<B> {
    Conv2dConfig::new([in_ch, out_ch], [kernel, kernel])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(pad, pad))
        .with_bias(false)
        .init(device)
}

fn dropout(drop_rate: f64) -> Option<Dropout> {
    (drop_rate > 0.0).then(|| DropoutConfig::new(drop_rate).init())
}

fn maybe_drop<B: Backend, const D: usize>(d: &Option<Dropout>, x: Tensor<B, D>) -> Tensor<B, D> {
    match d {
        Some(d) => d.forward(x),
        None => x,
    }
}

/// 2 倍双线性上采样。
fn upsample2<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    resize_bilinear(x, [h * 2, w * 2])
}

// ---------------------------------------------------------------- Dense block

/// dense block 内的单层。SiLU 保证二阶导连续，配合 PINN 使用。
#[derive(Module, Debug)]
pub struct DenseLayer<B: Backend> {
    norm1: BatchNorm<B, 2>,
    conv1: Conv2d<B>,
    /// bottleneck 分支：1x1 压缩后再 3x3
    norm2: Option<BatchNorm<B, 2>>,
    conv2: Option<Conv2d<B>>,
    drop: Option<Dropout>,
}

impl<B: Backend> DenseLayer<B> {
    fn new(
        in_features: usize,
        growth_rate: usize,
        drop_rate: f64,
        bn_size: usize,
        bottleneck: bool,
        device: &B::Device,
    ) -> Self {
        // 与 Python 一致：只有通道数确实多到值得压缩时才走 bottleneck
        if bottleneck && in_features > bn_size * growth_rate {
            let mid = bn_size * growth_rate;
            Self {
                norm1: BatchNormConfig::new(in_features).init(device),
                conv1: conv2d(in_features, mid, 1, 1, 0, device),
                norm2: Some(BatchNormConfig::new(mid).init(device)),
                conv2: Some(conv2d(mid, growth_rate, 3, 1, 1, device)),
                drop: dropout(drop_rate),
            }
        } else {
            Self {
                norm1: BatchNormConfig::new(in_features).init(device),
                conv1: conv2d(in_features, growth_rate, 3, 1, 1, device),
                norm2: None,
                conv2: None,
                drop: dropout(drop_rate),
            }
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.conv1.forward(silu(self.norm1.forward(x.clone())));
        let y = match (&self.norm2, &self.conv2) {
            (Some(n2), Some(c2)) => c2.forward(silu(n2.forward(y))),
            _ => y,
        };
        // dense connectivity：输出是「输入 + 新特征」的拼接
        Tensor::cat(vec![x, maybe_drop(&self.drop, y)], 1)
    }
}

impl<B: Backend> LoadState for DenseLayer<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        self.norm1.load_state(sd, &join(prefix, "norm1"))?;
        self.conv1.load_state(sd, &join(prefix, "conv1"))?;
        if let (Some(n2), Some(c2)) = (self.norm2.as_mut(), self.conv2.as_mut()) {
            n2.load_state(sd, &join(prefix, "norm2"))?;
            c2.load_state(sd, &join(prefix, "conv2"))?;
        }
        Ok(())
    }
}

#[derive(Module, Debug)]
pub struct DenseBlock<B: Backend> {
    layers: Vec<DenseLayer<B>>,
}

impl<B: Backend> DenseBlock<B> {
    fn new(
        num_layers: usize,
        in_features: usize,
        growth_rate: usize,
        drop_rate: f64,
        bn_size: usize,
        bottleneck: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|i| {
                    DenseLayer::new(
                        in_features + i * growth_rate,
                        growth_rate,
                        drop_rate,
                        bn_size,
                        bottleneck,
                        device,
                    )
                })
                .collect(),
        }
    }

    pub fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}

impl<B: Backend> LoadState for DenseBlock<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state(sd, &join(prefix, &format!("denselayer{}", i + 1)))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- Transition

/// 过渡层：下采样或上采样，同时压缩特征图数量。
///
/// 上采样用「插值 + 卷积」而不是 ConvTranspose：不产生棋盘伪影，且尺寸变换是纯
/// 2 倍，不依赖 `output_padding` 的奇偶性技巧。
///
/// Python 版还有一条 `bottleneck=False` 的下采样分支，但所有调用点都用默认值
/// `True`，那条路是死代码，这里不实现。
#[derive(Module, Debug)]
pub struct Transition<B: Backend> {
    norm1: BatchNorm<B, 2>,
    conv1: Conv2d<B>,
    norm2: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    drop1: Option<Dropout>,
    drop2: Option<Dropout>,
    down: bool,
}

impl<B: Backend> Transition<B> {
    fn new(
        in_features: usize,
        out_features: usize,
        down: bool,
        drop_rate: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            norm1: BatchNormConfig::new(in_features).init(device),
            conv1: conv2d(in_features, out_features, 1, 1, 0, device),
            norm2: BatchNormConfig::new(out_features).init(device),
            conv2: conv2d(
                out_features,
                out_features,
                3,
                if down { 2 } else { 1 },
                1,
                device,
            ),
            drop1: dropout(drop_rate),
            drop2: dropout(drop_rate),
            down,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.conv1.forward(silu(self.norm1.forward(x)));
        let h = maybe_drop(&self.drop1, h);
        let h = silu(self.norm2.forward(h));
        let h = if self.down { h } else { upsample2(h) };
        maybe_drop(&self.drop2, self.conv2.forward(h))
    }
}

impl<B: Backend> LoadState for Transition<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        self.norm1.load_state(sd, &join(prefix, "norm1"))?;
        self.conv1.load_state(sd, &join(prefix, "conv1"))?;
        self.norm2.load_state(sd, &join(prefix, "norm2"))?;
        self.conv2.load_state(sd, &join(prefix, "conv2"))
    }
}

/// 最后一级上采样，直接输出物理场。
#[derive(Module, Debug)]
pub struct LastDecoding<B: Backend> {
    norm1: BatchNorm<B, 2>,
    conv1: Conv2d<B>,
    norm2: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    drop1: Option<Dropout>,
}

impl<B: Backend> LastDecoding<B> {
    fn new(in_features: usize, out_channels: usize, drop_rate: f64, device: &B::Device) -> Self {
        let mid = in_features / 2;
        Self {
            norm1: BatchNormConfig::new(in_features).init(device),
            conv1: conv2d(in_features, mid, 1, 1, 0, device),
            norm2: BatchNormConfig::new(mid).init(device),
            conv2: conv2d(mid, out_channels, 3, 1, 1, device),
            drop1: dropout(drop_rate),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.conv1.forward(silu(self.norm1.forward(x)));
        let h = maybe_drop(&self.drop1, h);
        let h = upsample2(silu(self.norm2.forward(h)));
        self.conv2.forward(h)
    }
}

impl<B: Backend> LoadState for LastDecoding<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        self.norm1.load_state(sd, &join(prefix, "norm1"))?;
        self.conv1.load_state(sd, &join(prefix, "conv1"))?;
        self.norm2.load_state(sd, &join(prefix, "norm2"))?;
        self.conv2.load_state(sd, &join(prefix, "conv2"))
    }
}

// ---------------------------------------------------------------- 整网

/// 网络超参。默认值与 Python 版的关键字参数一致。
#[derive(Debug, Clone)]
pub struct DenseEdConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    /// 长度必须是奇数：前一半是编码器，后一半（含中间那个）是解码器
    pub blocks: Vec<usize>,
    pub growth_rate: usize,
    pub init_features: usize,
    pub bn_size: usize,
    pub drop_rate: f64,
    pub bottleneck: bool,
    pub out_activation: Option<Activation>,
    /// 缺省即输入尺寸
    pub output_size: Option<[usize; 2]>,
    pub num_heads: usize,
    pub use_positional_encoding: bool,
    pub attn_max_tokens: usize,
}

impl DenseEdConfig {
    pub fn new(in_channels: usize, out_channels: usize, blocks: Vec<usize>) -> Self {
        Self {
            in_channels,
            out_channels,
            blocks,
            growth_rate: 16,
            init_features: 48,
            bn_size: 8,
            drop_rate: 0.0,
            bottleneck: false,
            out_activation: None,
            output_size: None,
            num_heads: 4,
            use_positional_encoding: true,
            attn_max_tokens: 8192,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<DenseEd<B>, String> {
        DenseEd::new(self, device)
    }
}

/// 2D DenseED。
#[derive(Module, Debug)]
pub struct DenseEd<B: Backend> {
    in_conv: Conv2d<B>,
    enc_blocks: Vec<DenseBlock<B>>,
    trans_downs: Vec<Transition<B>>,
    transformer_bottleneck: TransformerBottleneck<B>,
    dec_blocks: Vec<DenseBlock<B>>,
    trans_ups: Vec<Transition<B>>,
    spatial_attention: SpatialAttention2d<B>,
    last_trans_up: LastDecoding<B>,
    out_activation: Ignored<Option<Activation>>,
    output_size: Ignored<Option<[usize; 2]>>,
}

impl<B: Backend> DenseEd<B> {
    fn new(cfg: &DenseEdConfig, device: &B::Device) -> Result<Self, String> {
        let n = cfg.blocks.len();
        if n > 1 && n.is_multiple_of(2) {
            return Err(format!("blocks 长度必须是奇数，当前为 {}", n));
        }
        let (enc_layers, dec_layers) = cfg.blocks.split_at(n / 2);

        let in_conv = conv2d(cfg.in_channels, cfg.init_features, 7, 2, 3, device);

        let mut enc_blocks = Vec::new();
        let mut trans_downs = Vec::new();
        let mut skip_channels: Vec<usize> = Vec::new();
        let mut num_features = cfg.init_features;

        for &num_layers in enc_layers {
            enc_blocks.push(DenseBlock::new(
                num_layers,
                num_features,
                cfg.growth_rate,
                cfg.drop_rate,
                cfg.bn_size,
                cfg.bottleneck,
                device,
            ));
            num_features += num_layers * cfg.growth_rate;
            skip_channels.push(num_features);

            trans_downs.push(Transition::new(
                num_features,
                num_features / 2,
                true,
                cfg.drop_rate,
                device,
            ));
            num_features /= 2;
        }

        let transformer_bottleneck = TransformerBottleneck::new(
            num_features,
            cfg.num_heads,
            cfg.use_positional_encoding,
            cfg.attn_max_tokens,
            device,
        );

        let mut dec_blocks = Vec::new();
        let mut trans_ups = Vec::new();
        for (i, &num_layers) in dec_layers.iter().enumerate() {
            let in_features = if i == 0 {
                num_features
            } else {
                trans_ups.push(Transition::new(
                    num_features,
                    num_features / 2,
                    false,
                    cfg.drop_rate,
                    device,
                ));
                num_features /= 2;
                // Python 的 skip_channels[-i]：从后往前数第 i 个
                num_features + skip_channels[skip_channels.len() - i]
            };
            dec_blocks.push(DenseBlock::new(
                num_layers,
                in_features,
                cfg.growth_rate,
                cfg.drop_rate,
                cfg.bn_size,
                cfg.bottleneck,
                device,
            ));
            num_features = in_features + num_layers * cfg.growth_rate;
        }

        Ok(Self {
            in_conv,
            enc_blocks,
            trans_downs,
            transformer_bottleneck,
            dec_blocks,
            trans_ups,
            spatial_attention: SpatialAttention2d::new(7, device),
            last_trans_up: LastDecoding::new(num_features, cfg.out_channels, cfg.drop_rate, device),
            out_activation: Ignored(cfg.out_activation),
            output_size: Ignored(cfg.output_size),
        })
    }

    /// Transformer 瓶颈实际生效的注意力头数。
    pub fn attention_heads(&self) -> usize {
        self.transformer_bottleneck.num_heads()
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, in_h, in_w] = x.dims();
        let mut x = self.in_conv.forward(x);

        let mut skips = Vec::with_capacity(self.enc_blocks.len());
        for (block, down) in self.enc_blocks.iter().zip(self.trans_downs.iter()) {
            x = block.forward(x);
            skips.push(x.clone());
            x = down.forward(x);
        }

        x = self.transformer_bottleneck.forward(x);

        for (i, block) in self.dec_blocks.iter().enumerate() {
            if i > 0 {
                x = self.trans_ups[i - 1].forward(x);
                let skip = skips[skips.len() - i].clone();
                let [_, _, sh, sw] = skip.dims();
                let [_, _, xh, xw] = x.dims();
                if (xh, xw) != (sh, sw) {
                    x = resize_bilinear(x, [sh, sw]);
                }
                x = Tensor::cat(vec![x, skip], 1);
            }
            x = block.forward(x);
        }

        x = self.spatial_attention.forward(x);
        x = self.last_trans_up.forward(x);
        if let Some(act) = self.out_activation.0 {
            x = act.apply(x);
        }

        // 输出尺寸兜底：偶数网格上卷积/上采样凑不出原尺寸，必须显式插回去
        let target = self.output_size.0.unwrap_or([in_h, in_w]);
        let [_, _, h, w] = x.dims();
        if [h, w] != target {
            x = resize_bilinear(x, target);
        }
        x
    }
}

impl<B: Backend> LoadState for DenseEd<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        self.in_conv.load_state(sd, &join(prefix, "in_conv"))?;
        for (i, b) in self.enc_blocks.iter_mut().enumerate() {
            b.load_state(sd, &join(prefix, &format!("enc_blocks.{}", i)))?;
        }
        for (i, t) in self.trans_downs.iter_mut().enumerate() {
            t.load_state(sd, &join(prefix, &format!("trans_downs.{}", i)))?;
        }
        self.transformer_bottleneck
            .load_state(sd, &join(prefix, "transformer_bottleneck"))?;
        for (i, b) in self.dec_blocks.iter_mut().enumerate() {
            b.load_state(sd, &join(prefix, &format!("dec_blocks.{}", i)))?;
        }
        for (i, t) in self.trans_ups.iter_mut().enumerate() {
            t.load_state(sd, &join(prefix, &format!("trans_ups.{}", i)))?;
        }
        self.spatial_attention
            .load_state(sd, &join(prefix, "spatial_attention"))?;
        self.last_trans_up
            .load_state(sd, &join(prefix, "last_trans_up"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn rejects_even_length_blocks() {
        let cfg = DenseEdConfig::new(1, 1, vec![2, 2]);
        assert!(
            cfg.init::<B>(&Default::default()).is_err(),
            "偶数长度的 blocks 分不出对称的编解码器，必须报错"
        );
    }

    #[test]
    fn output_matches_input_size_on_even_and_odd_grids() {
        // 旧版只在奇数分辨率下等尺寸（128 -> 129），偶数网格直接在损失处崩掉
        let device = Default::default();
        let mut cfg = DenseEdConfig::new(2, 3, vec![2, 2, 2]);
        cfg.growth_rate = 4;
        cfg.init_features = 8;
        for (h, w) in [(32usize, 32usize), (33, 29), (40, 24)] {
            let model = cfg.init::<B>(&device).unwrap();
            let y = model.forward(Tensor::<B, 4>::zeros([1, 2, h, w], &device));
            assert_eq!(y.dims(), [1, 3, h, w], "{}x{} 的输入输出尺寸不等", h, w);
        }
    }

    #[test]
    fn num_heads_reaches_the_attention_block() {
        // **整网的黄金对照验不到 num_heads**：那几组配置里瓶颈 C=8、权重 ±0.1，
        // softmax 退化成近似均匀分布，切几个头几乎不改结果（实测把 4 换成 1/2/8，
        // 四组整网参考数据最大只到 0.036 倍容差，差 28 倍才够红）。
        // 于是这条接线只能直接验：配置里的头数必须原样出现在注意力块上。
        let device = Default::default();
        let mut cfg = DenseEdConfig::new(1, 1, vec![2, 2, 2]);
        cfg.growth_rate = 4;
        // 瓶颈通道数 =（init_features + 2*growth_rate）/ 2 = 32，够被 1/2/4/8 整除
        cfg.init_features = 56;
        for heads in [1usize, 2, 4, 8] {
            let mut c = cfg.clone();
            c.num_heads = heads;
            let m = c.init::<B>(&device).unwrap();
            assert_eq!(
                m.attention_heads(),
                heads,
                "配置里的 num_heads={} 没有传到注意力块",
                heads
            );
        }
    }

    #[test]
    fn num_heads_falls_back_to_a_divisor() {
        // 通道数不能被头数整除时向下退让，与 Python 版同样的策略
        let device = Default::default();
        let mut cfg = DenseEdConfig::new(1, 1, vec![2, 2, 2]);
        cfg.growth_rate = 4;
        cfg.init_features = 8; // 瓶颈通道数 8
        cfg.num_heads = 5; // 8 % 5 != 0
        let m = cfg.init::<B>(&device).unwrap();
        let h = m.attention_heads();
        assert!(
            h < 5 && 8_usize.is_multiple_of(h),
            "退让后的头数 {} 不整除 8",
            h
        );
    }

    #[test]
    fn output_size_override_wins() {
        let device = Default::default();
        let mut cfg = DenseEdConfig::new(1, 1, vec![2, 2, 2]);
        cfg.growth_rate = 4;
        cfg.init_features = 8;
        cfg.output_size = Some([17, 21]);
        let model = cfg.init::<B>(&device).unwrap();
        let y = model.forward(Tensor::<B, 4>::zeros([1, 1, 32, 32], &device));
        assert_eq!(y.dims(), [1, 1, 17, 21]);
    }
}
