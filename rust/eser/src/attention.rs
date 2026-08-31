//! Transformer 瓶颈与空间注意力，2D / 3D 共用。移植自 `models/attention.py`。
//!
//! # 为什么要自己搭注意力
//!
//! Burn 的 `MultiHeadAttention` 是一个完整的注意力块，内部写死 Gelu，而这里要的是
//! **PyTorch `nn.MultiheadAttention` 的等价物**：只做投影 + 缩放点积 + 输出投影，
//! 归一化和 MLP 由外层 pre-norm 结构自己管。权重也得能按 PyTorch 的
//! `in_proj_weight` / `out_proj` 命名对上，所以用 `Linear` 自己搭。
//!
//! # 位置编码
//!
//! 自注意力对 token 排列是等变的。旧版把 (B,C,H,W) 直接铺平送进注意力、**没有任何
//! 位置编码**，于是这一层只能告诉网络「某处有高速区」，说不清在哪——对流场这种强
//! 空间结构的任务是实质性缺陷。这里补的是无参数的 sin-cos 编码，对分辨率自适应，
//! 换网格尺寸不用重训。
//!
//! 编码是**常量**，不参与求导，所以直接在主机侧算好再上传，比用张量算子拼快也简单。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use burn::module::{Ignored, Module};
use burn::nn::conv::{Conv2d, Conv2dConfig, Conv3d, Conv3dConfig};
use burn::nn::{
    LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d, PaddingConfig3d,
};
use burn::tensor::activation::{sigmoid, silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::store::{join, load_linear, LoadState, StateDict};

// ------------------------------------------------------------ 位置编码

/// 一维 sin-cos 编码，返回按 `[len, dim]` 行优先展平的缓冲。
///
/// `dim` 为奇数时先按 `dim + 1` 算再裁掉最后一列，与 Python 版一致。
fn sincos_1d(len: usize, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let eff = if dim % 2 == 1 { dim + 1 } else { dim };
    let half = eff / 2;
    let scale = -(10000.0f64.ln()) / half.max(1) as f64;
    let freq: Vec<f64> = (0..half).map(|j| (j as f64 * scale).exp()).collect();

    let mut out = vec![0.0f32; len * dim];
    for p in 0..len {
        for (k, f) in freq.iter().enumerate() {
            let ang = p as f64 * f;
            // 前 half 列是 sin，后 half 列是 cos（cat 的顺序，不是交错）
            if k < dim {
                out[p * dim + k] = ang.sin() as f32;
            }
            if half + k < dim {
                out[p * dim + half + k] = ang.cos() as f32;
            }
        }
    }
    out
}

/// 任意维度网格的可分离 sin-cos 位置编码，形状 `(1, C, *spatial)`。
///
/// 通道按轴均分，余数全给最后一根轴——与 Python 版逐位一致。
pub fn sincos_positional_encoding<B: Backend, const D: usize>(
    channels: usize,
    spatial: &[usize],
    device: &B::Device,
) -> Tensor<B, D> {
    let n_axes = spatial.len();
    assert_eq!(
        n_axes + 2,
        D,
        "spatial 有 {} 根轴，但目标是 {} 维张量",
        n_axes,
        D
    );
    let per_axis = channels / n_axes;
    let remainder = channels - per_axis * n_axes;

    let dims: Vec<usize> = (0..n_axes)
        .map(|a| per_axis + if a == n_axes - 1 { remainder } else { 0 })
        .collect();
    let encs: Vec<Vec<f32>> = (0..n_axes)
        .map(|a| sincos_1d(spatial[a], dims[a]))
        .collect();

    let n_spatial: usize = spatial.iter().product();
    let mut out = vec![0.0f32; channels * n_spatial];

    let mut base = 0usize; // 当前轴占用的起始通道
    for a in 0..n_axes {
        let d = dims[a];
        if d == 0 {
            continue;
        }
        // 这根轴的下标在展平后的步长
        let stride: usize = spatial[a + 1..].iter().product();
        for (flat, slot) in (0..n_spatial).map(|i| (i, i)) {
            let pos = (flat / stride) % spatial[a];
            for k in 0..d {
                out[(base + k) * n_spatial + slot] = encs[a][pos * d + k];
            }
        }
        base += d;
    }

    let mut shape = [1usize; D];
    shape[1] = channels;
    shape[2..].copy_from_slice(spatial);
    Tensor::<B, 1>::from_data(TensorData::from(out.as_slice()), device).reshape(shape)
}

// ------------------------------------------------------------ 多头自注意力

/// PyTorch `nn.MultiheadAttention(batch_first=True)` 的自注意力等价物。
///
/// 权重按 PyTorch 的布局存：`in_proj` 是一个 `[3C, C]` 的大矩阵，Q/K/V 各占一段。
#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    in_proj: Linear<B>,
    out_proj: Linear<B>,
    num_heads: usize,
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(channels: usize, num_heads: usize, device: &B::Device) -> Self {
        Self {
            in_proj: LinearConfig::new(channels, 3 * channels).init(device),
            out_proj: LinearConfig::new(channels, channels).init(device),
            num_heads,
        }
    }

    /// 实际生效的头数（构造时会向下退让到能整除通道数为止）。
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// `x`: (B, N, C)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, c] = x.dims();
        let h = self.num_heads;
        let hd = c / h;

        let qkv = self.in_proj.forward(x); // (B, N, 3C)
        let split = |i: usize| {
            qkv.clone()
                .slice([0..b, 0..n, i * c..(i + 1) * c])
                .reshape([b, n, h, hd])
                .swap_dims(1, 2) // (B, H, N, hd)
        };
        let (q, k, v) = (split(0), split(1), split(2));

        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar((hd as f64).sqrt());
        let attn = softmax(scores, 3);
        let context = attn
            .matmul(v)
            .swap_dims(1, 2) // (B, N, H, hd)
            .reshape([b, n, c]);
        // 输出投影不能省。少了它权重照样加载得上、形状照样对、前向照样出数——
        // 这个缺陷被削弱过的黄金数据整整盖住了一轮
        self.out_proj.forward(context)
    }
}

impl<B: Backend> LoadState for SelfAttention<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        // PyTorch 把 Q/K/V 的投影合成一个 in_proj_weight/in_proj_bias，
        // 命名不带 `.weight` 后缀，所以只能显式给键名
        load_linear(
            &mut self.in_proj,
            sd,
            &join(prefix, "in_proj_weight"),
            &join(prefix, "in_proj_bias"),
        )?;
        self.out_proj.load_state(sd, &join(prefix, "out_proj"))
    }
}

// ------------------------------------------------------------ Transformer 瓶颈

/// 编解码器最深处的全局自注意力块，2D / 3D 通用（`forward` 对张量阶数泛型）。
///
/// 结构是 pre-norm + SiLU MLP，与 Burn 自带的注意力块不是一回事。
/// Python 版的 dropout 参数在本项目里恒为 0（没有命令行开关能改它），所以这里
/// 不做这一层——真需要时再加，YAGNI。
#[derive(Module, Debug)]
pub struct TransformerBottleneck<B: Backend> {
    norm1: LayerNorm<B>,
    attn: SelfAttention<B>,
    norm2: LayerNorm<B>,
    mlp_in: Linear<B>,
    mlp_out: Linear<B>,
    use_positional_encoding: bool,
    max_tokens: usize,
    /// 只提醒一次，别把训练日志刷满
    warned: Ignored<Arc<AtomicBool>>,
}

impl<B: Backend> TransformerBottleneck<B> {
    pub fn new(
        channels: usize,
        num_heads: usize,
        use_positional_encoding: bool,
        max_tokens: usize,
        device: &B::Device,
    ) -> Self {
        // 保证 embed_dim 能被 num_heads 整除，与 Python 版同样的退让策略
        let mut heads = num_heads.max(1);
        while !channels.is_multiple_of(heads) && heads > 1 {
            heads -= 1;
        }
        let hidden = (channels as f64 * 4.0) as usize;
        Self {
            norm1: LayerNormConfig::new(channels).init(device),
            attn: SelfAttention::new(channels, heads, device),
            norm2: LayerNormConfig::new(channels).init(device),
            mlp_in: LinearConfig::new(channels, hidden).init(device),
            mlp_out: LinearConfig::new(hidden, channels).init(device),
            use_positional_encoding,
            max_tokens,
            warned: Ignored(Arc::new(AtomicBool::new(false))),
        }
    }

    /// 实际生效的注意力头数。供上层验证「配置里的 num_heads 确实传到了这里」——
    /// 整网的黄金对照验不到它（瓶颈的 softmax 近似均匀，切几个头结果几乎不变）。
    pub fn num_heads(&self) -> usize {
        self.attn.num_heads()
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let shape = x.dims();
        let spatial: Vec<usize> = shape[2..].to_vec();
        let n_tokens: usize = spatial.iter().product();

        if n_tokens > self.max_tokens {
            // O(N²) 注意力在大网格上会爆显存。跳过并提示，而不是让它崩掉——
            // 但必须提示：静默跳过等于这一层白建了
            if !self.warned.0.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[TransformerBottleneck] token 数 {} 超过上限 {}，本层跳过（可调 attn_max_tokens）",
                    n_tokens, self.max_tokens
                );
            }
            return x;
        }

        let (b, c) = (shape[0], shape[1]);
        let x = if self.use_positional_encoding {
            let pe = sincos_positional_encoding::<B, D>(c, &spatial, &x.device());
            x + pe
        } else {
            x
        };

        let flat = x.reshape([b, c, n_tokens]).swap_dims(1, 2); // (B, N, C)
        let h = self.norm1.forward(flat.clone());
        let flat = flat + self.attn.forward(h);
        let h = self.norm2.forward(flat.clone());
        let flat = flat + self.mlp_out.forward(silu(self.mlp_in.forward(h)));

        flat.swap_dims(1, 2).reshape(shape)
    }
}

impl<B: Backend> LoadState for TransformerBottleneck<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        self.norm1.load_state(sd, &join(prefix, "norm1"))?;
        self.attn.load_state(sd, &join(prefix, "attn"))?;
        self.norm2.load_state(sd, &join(prefix, "norm2"))?;
        // Python 侧 mlp 是 Sequential[Linear, SiLU, Dropout, Linear, Dropout]，
        // 所以两个线性层的下标是 0 和 3
        self.mlp_in.load_state(sd, &join(prefix, "mlp.0"))?;
        self.mlp_out.load_state(sd, &join(prefix, "mlp.3"))
    }
}

// ------------------------------------------------------------ 空间注意力

/// 通道维 avg/max 池化 -> 卷积 -> sigmoid 权重图（2D）。
#[derive(Module, Debug)]
pub struct SpatialAttention2d<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> SpatialAttention2d<B> {
    pub fn new(kernel_size: usize, device: &B::Device) -> Self {
        let pad = kernel_size / 2;
        Self {
            conv: Conv2dConfig::new([2, 1], [kernel_size, kernel_size])
                .with_padding(PaddingConfig2d::Explicit(pad, pad))
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let stats = Tensor::cat(vec![x.clone().mean_dim(1), channel_max_4d(x.clone())], 1);
        x * sigmoid(self.conv.forward(stats))
    }
}

/// 沿通道维取最大值，返回 (B, 1, H, W)。
///
/// **不能直接用 `x.max_dim(1)`**：它的反向传播走 gather/scatter，而 Burn 的 ndarray
/// 后端要求索引张量「只有最后一维可以不同」，通道维在第 1 位，一反向就 panic
/// （`Unsupported dimension, only the last dimension can differ`）。前向完全正常，
/// 只有训练时才炸——又是一个「不训练就发现不了」的坑。
/// 先把通道维换到最后再规约，绕开这条限制，代价只是两次换轴。
fn channel_max_4d<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    x.swap_dims(1, 3).max_dim(3).swap_dims(1, 3)
}

/// 3D 版，同样的理由。
fn channel_max_5d<B: Backend>(x: Tensor<B, 5>) -> Tensor<B, 5> {
    x.swap_dims(1, 4).max_dim(4).swap_dims(1, 4)
}

impl<B: Backend> LoadState for SpatialAttention2d<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        self.conv.load_state(sd, &join(prefix, "conv"))
    }
}

/// 3D 版。
#[derive(Module, Debug)]
pub struct SpatialAttention3d<B: Backend> {
    conv: Conv3d<B>,
}

impl<B: Backend> SpatialAttention3d<B> {
    pub fn new(kernel_size: usize, device: &B::Device) -> Self {
        let pad = kernel_size / 2;
        Self {
            conv: Conv3dConfig::new([2, 1], [kernel_size; 3])
                .with_padding(PaddingConfig3d::Explicit(pad, pad, pad))
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let stats = Tensor::cat(vec![x.clone().mean_dim(1), channel_max_5d(x.clone())], 1);
        x * sigmoid(self.conv.forward(stats))
    }
}

impl<B: Backend> LoadState for SpatialAttention3d<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        self.conv.load_state(sd, &join(prefix, "conv"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn positional_encoding_splits_channels_by_axis() {
        // C=8, 2 根轴 -> 每根 4 个通道。前 4 个只随 H 变，后 4 个只随 W 变
        let pe: Tensor<B, 4> = sincos_positional_encoding(8, &[5, 7], &Default::default());
        assert_eq!(pe.dims(), [1, 8, 5, 7]);
        let v: Vec<f32> = pe.into_data().to_vec().unwrap();
        let at = |c: usize, y: usize, x: usize| v[(c * 5 + y) * 7 + x];

        for x in 1..7 {
            assert_eq!(at(0, 3, x), at(0, 3, 0), "前半通道不应随 W 变化");
        }
        for y in 1..5 {
            assert_eq!(at(5, y, 2), at(5, 0, 2), "后半通道不应随 H 变化");
        }
        // 位置 0 的 sin 分量为 0、cos 分量为 1
        assert_eq!(at(0, 0, 0), 0.0);
        assert!((at(2, 0, 0) - 1.0).abs() < 1e-6, "cos(0) 应为 1");
    }

    #[test]
    fn positional_encoding_breaks_permutation_equivariance() {
        // 位置编码存在的全部意义：两个位置拿到不同的向量
        let pe: Tensor<B, 4> = sincos_positional_encoding(8, &[6, 6], &Default::default());
        let v: Vec<f32> = pe.into_data().to_vec().unwrap();
        let col = |y: usize, x: usize| (0..8).map(|c| v[(c * 6 + y) * 6 + x]).collect::<Vec<_>>();
        assert_ne!(col(0, 0), col(1, 0), "不同行必须拿到不同编码");
        assert_ne!(col(0, 0), col(0, 1), "不同列必须拿到不同编码");
    }

    #[test]
    fn channel_max_survives_backward() {
        // 直接用 `max_dim(1)` 的话，前向没事、反向 panic。这条测试就是把那个坑钉住
        use burn::backend::Autodiff;
        type AB = Autodiff<NdArray<f32>>;
        let device = Default::default();
        let x = Tensor::<AB, 4>::ones([2, 5, 4, 4], &device).require_grad();
        let m = SpatialAttention2d::<AB>::new(7, &device);
        let loss = m.forward(x.clone()).sum();
        let grads = loss.backward();
        assert!(x.grad(&grads).is_some(), "空间注意力的梯度回传断了");
    }

    #[test]
    fn channel_max_picks_the_largest_channel() {
        let device = Default::default();
        let data: Vec<f32> = (0..3 * 4).map(|i| (i / 4) as f32).collect(); // 通道 i 全等于 i
        let x = Tensor::<B, 1>::from_data(TensorData::from(data.as_slice()), &device)
            .reshape([1, 3, 2, 2]);
        let got: Vec<f32> = channel_max_4d(x).into_data().to_vec().unwrap();
        assert_eq!(got, vec![2.0; 4], "换轴之后规约的不是通道维");
    }

    #[test]
    fn attention_skips_when_tokens_exceed_limit() {
        let m = TransformerBottleneck::<B>::new(4, 2, true, 8, &Default::default());
        let x = Tensor::<B, 4>::ones([1, 4, 5, 5], &Default::default()); // 25 > 8
        let y = m.forward(x.clone());
        assert_eq!(
            y.into_data().to_vec::<f32>().unwrap(),
            x.into_data().to_vec::<f32>().unwrap(),
            "超限时应原样返回"
        );
    }
}
