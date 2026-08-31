//! 按 PyTorch `state_dict` 的名字往 Burn 模块里灌权重。
//!
//! 为什么不用现成的：Burn 0.18 没有 `burn-store`（那是更新版本的东西），
//! `burn-import` 又主要是 build 期代码生成。而这里真正要解决的问题很小——把
//! 「名字 -> 张量」塞进已经建好的模块——自己写一层比引一个新依赖便宜得多。
//!
//! # 两个必须防的静默错误
//!
//! 1. **Linear 的权重要转置**。PyTorch 的 `Linear.weight` 是 `[out, in]`，
//!    Burn 的是 `[in, out]`。形状恰好是方阵时（本项目的 `out_proj` 就是）不转置
//!    也不会报错，只会算出另一个矩阵。
//! 2. **没被用上的键必须报出来**。前缀写错一个字母，那一层就悄悄留着随机初始化，
//!    模型照样前向、照样出数——这类错误只有「所有键都被消费掉」这条断言能抓。

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv3d};
use burn::nn::{BatchNorm, LayerNorm, Linear};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// 一份扁平的「参数名 -> 数据」表，等价于 PyTorch 的 `state_dict()`。
#[derive(Debug, Default)]
pub struct StateDict {
    entries: HashMap<String, (Vec<usize>, Vec<f32>)>,
    used: RefCell<HashSet<String>>,
}

impl StateDict {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) {
        let name = name.into();
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            data.len(),
            "{}：形状 {:?} 与数据长度 {} 不符",
            name,
            shape,
            data.len()
        );
        self.entries.insert(name, (shape, data));
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 取出一个 D 维张量，并把这个键记为已用。
    pub fn tensor<B: Backend, const D: usize>(&self, name: &str) -> Result<Tensor<B, D>, String> {
        let (shape, data) = self
            .entries
            .get(name)
            .ok_or_else(|| format!("权重表里没有 {:?}", name))?;
        if shape.len() != D {
            return Err(format!(
                "{:?} 是 {} 维（形状 {:?}），期望 {} 维",
                name,
                shape.len(),
                shape,
                D
            ));
        }
        self.used.borrow_mut().insert(name.to_string());
        let mut dims = [1usize; D];
        dims.copy_from_slice(shape);
        Ok(
            Tensor::<B, 1>::from_data(TensorData::from(data.as_slice()), &Default::default())
                .reshape(dims),
        )
    }

    /// 还没被任何模块取走的键。**非空就说明有一层没接上线**。
    pub fn unused(&self) -> Vec<String> {
        let used = self.used.borrow();
        let mut left: Vec<String> = self
            .entries
            .keys()
            .filter(|k| !used.contains(*k))
            .cloned()
            .collect();
        left.sort();
        left
    }

    /// 加载完必须调这个。留下没用过的键 = 有模块保持着随机初始化，模型照样跑，
    /// 只是算的不是你以为的那个东西。
    pub fn ensure_fully_used(&self) -> Result<(), String> {
        let left = self.unused();
        if left.is_empty() {
            return Ok(());
        }
        Err(format!(
            "{} 个权重没有被加载（前缀写错？）：{}",
            left.len(),
            left.join(", ")
        ))
    }
}

/// 拼接模块路径。前缀为空时直接用字段名——顶层模块（单独验一个子模块时）
/// 的键名不带前缀，`format!("{}.{}", "", name)` 会拼出 `.name` 这种对不上的键。
pub fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", prefix, name)
    }
}

/// 能从 `state_dict` 里按前缀取权重的模块。
pub trait LoadState {
    /// `prefix` 是 PyTorch 侧的模块路径，如 `"enc_blocks.0.denselayer1"`。
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String>;
}

fn set<B: Backend, const D: usize>(
    slot: &mut Param<Tensor<B, D>>,
    sd: &StateDict,
    name: &str,
) -> Result<(), String> {
    let t: Tensor<B, D> = sd.tensor(name)?;
    let want = slot.val().dims();
    if t.dims() != want {
        return Err(format!(
            "{:?} 形状 {:?}，模块期望 {:?}",
            name,
            t.dims(),
            want
        ));
    }
    *slot = Param::from_tensor(t);
    Ok(())
}

/// 可选参数：权重表里没有就保持原样（例如推理用不到的 running 统计量）。
fn set_optional<B: Backend, const D: usize>(
    slot: &mut Option<Param<Tensor<B, D>>>,
    sd: &StateDict,
    name: &str,
) -> Result<(), String> {
    match (slot.as_mut(), sd.contains(name)) {
        (Some(p), true) => set(p, sd, name),
        (None, true) => Err(format!("{:?} 在权重表里，但模块没有这个参数", name)),
        (Some(_), false) => Err(format!("模块需要 {:?}，权重表里没有", name)),
        (None, false) => Ok(()),
    }
}

impl<B: Backend> LoadState for Conv2d<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        set(&mut self.weight, sd, &join(prefix, "weight"))?;
        set_optional(&mut self.bias, sd, &join(prefix, "bias"))
    }
}

impl<B: Backend> LoadState for Conv3d<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        set(&mut self.weight, sd, &join(prefix, "weight"))?;
        set_optional(&mut self.bias, sd, &join(prefix, "bias"))
    }
}

impl<B: Backend, const D: usize> LoadState for BatchNorm<B, D> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        // PyTorch 叫 weight/bias，Burn 叫 gamma/beta，是同一对仿射参数
        set(&mut self.gamma, sd, &join(prefix, "weight"))?;
        set(&mut self.beta, sd, &join(prefix, "bias"))?;
        // running 统计量是 buffer 不是 parameter，导出时可能没有；缺了就保持
        // Burn 的默认初值（mean=0 / var=1），推理结果与 PyTorch 侧同样的初值一致
        for (name, is_mean) in [("running_mean", true), ("running_var", false)] {
            let key = format!("{}.{}", prefix, name);
            if sd.contains(&key) {
                let t: Tensor<B, 1> = sd.tensor(&key)?;
                if is_mean {
                    self.running_mean = burn::module::RunningState::new(t);
                } else {
                    self.running_var = burn::module::RunningState::new(t);
                }
            }
        }
        Ok(())
    }
}

/// 按显式键名加载线性层。
///
/// PyTorch 里有些线性层的参数不叫 `<prefix>.weight`——`nn.MultiheadAttention` 的
/// 就叫 `in_proj_weight`，所以键名要能单独指定。
pub fn load_linear<B: Backend>(
    layer: &mut Linear<B>,
    sd: &StateDict,
    weight_key: &str,
    bias_key: &str,
) -> Result<(), String> {
    // PyTorch [out, in] -> Burn [in, out]。方阵时不转置也不会报错，只会算错
    let w: Tensor<B, 2> = sd.tensor(weight_key)?;
    let w = w.transpose();
    let want = layer.weight.val().dims();
    if w.dims() != want {
        return Err(format!(
            "{:?} 转置后 {:?}，模块期望 {:?}",
            weight_key,
            w.dims(),
            want
        ));
    }
    layer.weight = Param::from_tensor(w);
    set_optional(&mut layer.bias, sd, bias_key)
}

impl<B: Backend> LoadState for Linear<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        load_linear(self, sd, &join(prefix, "weight"), &join(prefix, "bias"))
    }
}

impl<B: Backend> LoadState for LayerNorm<B> {
    fn load_state(&mut self, sd: &StateDict, prefix: &str) -> Result<(), String> {
        set(&mut self.gamma, sd, &join(prefix, "weight"))?;
        set(&mut self.beta, sd, &join(prefix, "bias"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::LinearConfig;
    use burn::tensor::Tensor;

    type B = NdArray<f32>;

    #[test]
    fn linear_weight_is_transposed() {
        // 非方阵才能暴露转置问题：PyTorch [3, 2] -> Burn [2, 3]
        let mut sd = StateDict::new();
        sd.insert("fc.weight", vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        sd.insert("fc.bias", vec![3], vec![0.0, 0.0, 0.0]);

        let mut fc = LinearConfig::new(2, 3).init::<B>(&Default::default());
        fc.load_state(&sd, "fc").unwrap();
        assert_eq!(fc.weight.val().dims(), [2, 3]);

        // x = [1, 0] 应当取出 PyTorch 权重的第一列 [1, 3, 5]
        let x = Tensor::<B, 2>::from_data(TensorData::from([[1.0f32, 0.0]]), &Default::default());
        let y: Vec<f32> = fc.forward(x).into_data().to_vec().unwrap();
        assert_eq!(y, vec![1.0, 3.0, 5.0], "转置方向反了");
    }

    #[test]
    fn running_statistics_are_loaded_and_not_swapped() {
        // **这段分支此前一次都没执行过**：所有黄金/参照权重表都只导出
        // `named_parameters()`，running_mean 恒为 0、running_var 恒为 1，
        // 于是推理模式的 BatchNorm 退化成纯仿射 `gamma*x/sqrt(1+eps)+beta`——
        // 把这段代码整段删掉、或者把 mean 和 var 装反，502 项黄金数据全都照样绿。
        //
        // 这里用**非平凡且互不相同**的 mean/var：装反了输出会明显不同。
        use burn::nn::BatchNormConfig;

        let device = Default::default();
        let mut sd = StateDict::new();
        sd.insert("bn.weight", vec![2], vec![1.0, 1.0]);
        sd.insert("bn.bias", vec![2], vec![0.0, 0.0]);
        sd.insert("bn.running_mean", vec![2], vec![3.0, -1.0]);
        sd.insert("bn.running_var", vec![2], vec![4.0, 0.25]);

        let mut bn: BatchNorm<B, 2> = BatchNormConfig::new(2).init(&device);
        bn.load_state(&sd, "bn").unwrap();
        sd.ensure_fully_used().unwrap();

        // 推理模式（非 autodiff 后端）：y = (x - mean) / sqrt(var + eps) * gamma + beta
        let x = Tensor::<B, 4>::from_data(TensorData::from([[[[3.0f32]], [[-1.0f32]]]]), &device);
        let y: Vec<f32> = bn.forward(x).into_data().to_vec().unwrap();

        // 两个通道都恰好喂了自己的均值，输出应当都是 0
        for (i, v) in y.iter().enumerate() {
            assert!(
                v.abs() < 1e-3,
                "通道 {} 的输出是 {}，说明 running_mean 没装上或装错了通道",
                i,
                v
            );
        }

        // 再喂一个偏离均值的点，检查用的是各自的 var（4.0 与 0.25，差 4 倍标准差）
        let x2 = Tensor::<B, 4>::from_data(TensorData::from([[[[5.0f32]], [[1.0f32]]]]), &device);
        let y2: Vec<f32> = bn.forward(x2).into_data().to_vec().unwrap();
        // ch0: (5-3)/sqrt(4) = 1.0 ；ch1: (1-(-1))/sqrt(0.25) = 4.0
        assert!((y2[0] - 1.0).abs() < 1e-3, "ch0 实得 {}，期望 1.0", y2[0]);
        assert!(
            (y2[1] - 4.0).abs() < 1e-3,
            "ch1 实得 {}，期望 4.0——mean 与 var 很可能装反了",
            y2[1]
        );
    }

    #[test]
    fn running_statistics_are_optional() {
        // 只有 parameter、没有 buffer 的权重表（黄金数据就是这样）必须能正常加载，
        // running 统计量保持 Burn 的默认初值
        use burn::nn::BatchNormConfig;
        let device = Default::default();
        let mut sd = StateDict::new();
        sd.insert("bn.weight", vec![2], vec![2.0, 3.0]);
        sd.insert("bn.bias", vec![2], vec![0.5, -0.5]);

        let mut bn: BatchNorm<B, 2> = BatchNormConfig::new(2).init(&device);
        bn.load_state(&sd, "bn").expect("缺 running 统计量不该报错");
        sd.ensure_fully_used().unwrap();
        assert_eq!(
            bn.gamma.val().into_data().to_vec::<f32>().unwrap(),
            vec![2.0, 3.0]
        );
    }

    #[test]
    fn unused_keys_are_reported() {
        let mut sd = StateDict::new();
        sd.insert("fc.weight", vec![3, 2], vec![0.0; 6]);
        sd.insert("fc.bias", vec![3], vec![0.0; 3]);
        sd.insert("typo.weight", vec![2], vec![0.0; 2]);

        let mut fc = LinearConfig::new(2, 3).init::<B>(&Default::default());
        fc.load_state(&sd, "fc").unwrap();
        let err = sd.ensure_fully_used().unwrap_err();
        assert!(err.contains("typo.weight"), "没报出漏掉的键：{}", err);
    }

    #[test]
    fn shape_mismatch_is_an_error_not_a_silent_reshape() {
        let mut sd = StateDict::new();
        sd.insert("fc.weight", vec![4, 2], vec![0.0; 8]);
        let mut fc = LinearConfig::new(2, 3).init::<B>(&Default::default());
        assert!(fc.load_state(&sd, "fc").is_err(), "形状不符必须报错");
    }
}
