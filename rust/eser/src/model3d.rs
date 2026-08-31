//! 3D Dense Convolutional Encoder-Decoder（体素版）。移植自 `models/dense_ed_3d.py`。
//!
//! 与 2D 版是**结构相同、算子不同**的两份代码，Python 侧也是这样两个文件。合并成
//! 一份泛型实现要给 Conv2d/Conv3d、BatchNorm<2>/BatchNorm<3> 造一整套 trait，
//! 换来的是更难对照 Python 原版——这笔买卖不划算。
//!
//! 3D 版的关键点：**所有卷积核在深度方向都是 1**（`(1, 3, 3)` / `(1, 7, 7)`），
//! 下采样与上采样也只作用在 H/W 上（`stride=(1, 2, 2)`）。深度方向纯粹是并列的
//! 切片，不做任何混合——这是原模型的设计，不是简化。
//!
//! 激活全部是 SiLU：旧版 3D 分支还留着 `LeakyReLU(0.1)`，一阶导在 0 处不连续、
//! 二阶导处处为 0，恰好破坏 PINN 需要的高阶光滑性。

use burn::module::{Ignored, Module};
use burn::nn::conv::{Conv3d, Conv3dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Dropout, PaddingConfig3d};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::attention::{SpatialAttention3d, TransformerBottleneck};
use crate::interp::resize_trilinear;
use crate::model::Activation;
use crate::store::{join, LoadState, StateDict};

/// 深度方向恒为 1 的 3D 卷积。`kernel` / `stride` / `pad` 只给 H/W。
fn conv3d<B: Backend>(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    device: &B::Device,
) -> Conv3d<B> {
    Conv3dConfig::new([in_ch, out_ch], [1, kernel, kernel])
        .with_stride([1, stride, stride])
        .with_padding(PaddingConfig3d::Explicit(0, pad, pad))
        .with_bias(false)
        .init(device)
}

fn dropout(drop_rate: f64) -> Option<Dropout> {
    (drop_rate > 0.0).then(|| burn::nn::DropoutConfig::new(drop_rate).init())
}

fn maybe_drop<B: Backend>(d: &Option<Dropout>, x: Tensor<B, 5>) -> Tensor<B, 5> {
    match d {
        Some(d) => d.forward(x),
        None => x,
    }
}

/// H/W 各放大 2 倍，深度不动。
fn upsample2<B: Backend>(x: Tensor<B, 5>) -> Tensor<B, 5> {
    let [_, _, d, h, w] = x.dims();
    resize_trilinear(x, [d, h * 2, w * 2])
}

// ---------------------------------------------------------------- Dense block

#[derive(Module, Debug)]
pub struct DenseLayer3d<B: Backend> {
    norm1: BatchNorm<B, 3>,
    conv1: Conv3d<B>,
    norm2: Option<BatchNorm<B, 3>>,
    conv2: Option<Conv3d<B>>,
    drop: Option<Dropout>,
}

impl<B: Backend> DenseLayer3d<B> {
    fn new(
        in_features: usize,
        growth_rate: usize,
        drop_rate: f64,
        bn_size: usize,
        bottleneck: bool,
        device: &B::Device,
    ) -> Self {
        if bottleneck && in_features > bn_size * growth_rate {
            let mid = bn_size * growth_rate;
            Self {
                norm1: BatchNormConfig::new(in_features).init(device),
                conv1: conv3d(in_features, mid, 1, 1, 0, device),
                norm2: Some(BatchNormConfig::new(mid).init(device)),
                conv2: Some(conv3d(mid, growth_rate, 3, 1, 1, device)),
                drop: dropout(drop_rate),
            }
        } else {
            Self {
                norm1: BatchNormConfig::new(in_features).init(device),
                conv1: conv3d(in_features, growth_rate, 3, 1, 1, device),
                norm2: None,
                conv2: None,
                drop: dropout(drop_rate),
            }
        }
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let y = self.conv1.forward(silu(self.norm1.forward(x.clone())));
        let y = match (&self.norm2, &self.conv2) {
            (Some(n2), Some(c2)) => c2.forward(silu(n2.forward(y))),
            _ => y,
        };
        Tensor::cat(vec![x, maybe_drop(&self.drop, y)], 1)
    }
}

impl<B: Backend> LoadState for DenseLayer3d<B> {
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
pub struct DenseBlock3d<B: Backend> {
    layers: Vec<DenseLayer3d<B>>,
}

impl<B: Backend> DenseBlock3d<B> {
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
                    DenseLayer3d::new(
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

    pub fn forward(&self, mut x: Tensor<B, 5>) -> Tensor<B, 5> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}

impl<B: Backend> LoadState for DenseBlock3d<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state(sd, &join(prefix, &format!("denselayer{}", i + 1)))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- Transition

#[derive(Module, Debug)]
pub struct Transition3d<B: Backend> {
    norm1: BatchNorm<B, 3>,
    conv1: Conv3d<B>,
    norm2: BatchNorm<B, 3>,
    conv2: Conv3d<B>,
    drop1: Option<Dropout>,
    drop2: Option<Dropout>,
    down: bool,
}

impl<B: Backend> Transition3d<B> {
    fn new(
        in_features: usize,
        out_features: usize,
        down: bool,
        drop_rate: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            norm1: BatchNormConfig::new(in_features).init(device),
            conv1: conv3d(in_features, out_features, 1, 1, 0, device),
            norm2: BatchNormConfig::new(out_features).init(device),
            conv2: conv3d(
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

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let h = self.conv1.forward(silu(self.norm1.forward(x)));
        let h = maybe_drop(&self.drop1, h);
        let h = silu(self.norm2.forward(h));
        let h = if self.down { h } else { upsample2(h) };
        maybe_drop(&self.drop2, self.conv2.forward(h))
    }
}

impl<B: Backend> LoadState for Transition3d<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        self.norm1.load_state(sd, &join(prefix, "norm1"))?;
        self.conv1.load_state(sd, &join(prefix, "conv1"))?;
        self.norm2.load_state(sd, &join(prefix, "norm2"))?;
        self.conv2.load_state(sd, &join(prefix, "conv2"))
    }
}

#[derive(Module, Debug)]
pub struct LastDecoding3d<B: Backend> {
    norm1: BatchNorm<B, 3>,
    conv1: Conv3d<B>,
    norm2: BatchNorm<B, 3>,
    conv2: Conv3d<B>,
    drop1: Option<Dropout>,
}

impl<B: Backend> LastDecoding3d<B> {
    fn new(in_features: usize, out_channels: usize, drop_rate: f64, device: &B::Device) -> Self {
        let mid = in_features / 2;
        Self {
            norm1: BatchNormConfig::new(in_features).init(device),
            conv1: conv3d(in_features, mid, 1, 1, 0, device),
            norm2: BatchNormConfig::new(mid).init(device),
            conv2: conv3d(mid, out_channels, 3, 1, 1, device),
            drop1: dropout(drop_rate),
        }
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let h = self.conv1.forward(silu(self.norm1.forward(x)));
        let h = maybe_drop(&self.drop1, h);
        let h = upsample2(silu(self.norm2.forward(h)));
        self.conv2.forward(h)
    }
}

impl<B: Backend> LoadState for LastDecoding3d<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        self.norm1.load_state(sd, &join(prefix, "norm1"))?;
        self.conv1.load_state(sd, &join(prefix, "conv1"))?;
        self.norm2.load_state(sd, &join(prefix, "norm2"))?;
        self.conv2.load_state(sd, &join(prefix, "conv2"))
    }
}

// ---------------------------------------------------------------- 整网

/// 3D 网络超参。与 [`crate::model::DenseEdConfig`] 同构，只是 `output_size` 多一根深度轴。
#[derive(Debug, Clone)]
pub struct DenseEd3dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub blocks: Vec<usize>,
    pub growth_rate: usize,
    pub init_features: usize,
    pub bn_size: usize,
    pub drop_rate: f64,
    pub bottleneck: bool,
    pub out_activation: Option<Activation>,
    pub output_size: Option<[usize; 3]>,
    pub num_heads: usize,
    pub use_positional_encoding: bool,
    pub attn_max_tokens: usize,
}

impl DenseEd3dConfig {
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

    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<DenseEd3d<B>, String> {
        DenseEd3d::new(self, device)
    }
}

#[derive(Module, Debug)]
pub struct DenseEd3d<B: Backend> {
    in_conv: Conv3d<B>,
    enc_blocks: Vec<DenseBlock3d<B>>,
    trans_downs: Vec<Transition3d<B>>,
    transformer_bottleneck: TransformerBottleneck<B>,
    dec_blocks: Vec<DenseBlock3d<B>>,
    trans_ups: Vec<Transition3d<B>>,
    spatial_attention: SpatialAttention3d<B>,
    last_trans_up: LastDecoding3d<B>,
    out_activation: Ignored<Option<Activation>>,
    output_size: Ignored<Option<[usize; 3]>>,
}

impl<B: Backend> DenseEd3d<B> {
    fn new(cfg: &DenseEd3dConfig, device: &B::Device) -> Result<Self, String> {
        let n = cfg.blocks.len();
        if n > 1 && n.is_multiple_of(2) {
            return Err(format!("blocks 长度必须是奇数，当前为 {}", n));
        }
        let (enc_layers, dec_layers) = cfg.blocks.split_at(n / 2);

        let in_conv = conv3d(cfg.in_channels, cfg.init_features, 7, 2, 3, device);

        let mut enc_blocks = Vec::new();
        let mut trans_downs = Vec::new();
        let mut skip_channels: Vec<usize> = Vec::new();
        let mut num_features = cfg.init_features;

        for &num_layers in enc_layers {
            enc_blocks.push(DenseBlock3d::new(
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

            trans_downs.push(Transition3d::new(
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
                trans_ups.push(Transition3d::new(
                    num_features,
                    num_features / 2,
                    false,
                    cfg.drop_rate,
                    device,
                ));
                num_features /= 2;
                num_features + skip_channels[skip_channels.len() - i]
            };
            dec_blocks.push(DenseBlock3d::new(
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
            spatial_attention: SpatialAttention3d::new(7, device),
            last_trans_up: LastDecoding3d::new(
                num_features,
                cfg.out_channels,
                cfg.drop_rate,
                device,
            ),
            out_activation: Ignored(cfg.out_activation),
            output_size: Ignored(cfg.output_size),
        })
    }

    /// Transformer 瓶颈实际生效的注意力头数。
    pub fn attention_heads(&self) -> usize {
        self.transformer_bottleneck.num_heads()
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [_, _, in_d, in_h, in_w] = x.dims();
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
                let [_, _, sd, sh, sw] = skip.dims();
                let [_, _, xd, xh, xw] = x.dims();
                if [xd, xh, xw] != [sd, sh, sw] {
                    x = resize_trilinear(x, [sd, sh, sw]);
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

        let target = self.output_size.0.unwrap_or([in_d, in_h, in_w]);
        let [_, _, d, h, w] = x.dims();
        if [d, h, w] != target {
            x = resize_trilinear(x, target);
        }
        x
    }
}

impl<B: Backend> LoadState for DenseEd3d<B> {
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
    fn depth_axis_is_preserved() {
        // 所有卷积核在深度方向都是 1、上下采样也只动 H/W，深度必须原样传下去
        let device = Default::default();
        let mut cfg = DenseEd3dConfig::new(2, 3, vec![2, 2, 2]);
        cfg.growth_rate = 4;
        cfg.init_features = 8;
        let model = cfg.init::<B>(&device).unwrap();
        for d in [1usize, 2, 5] {
            let y = model.forward(Tensor::<B, 5>::zeros([1, 2, d, 24, 24], &device));
            assert_eq!(y.dims(), [1, 3, d, 24, 24], "深度 {} 没有原样保留", d);
        }
    }
}
