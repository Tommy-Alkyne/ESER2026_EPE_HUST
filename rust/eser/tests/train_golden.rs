//! P5 验收：一个完整训练步逐元素对上黄金数据（第 7 层）。
//!
//! 这是整套参考数据里最上层的一项，前六层都对上了它才有意义。它一次钉住四件事：
//!
//! 1. **训练模式的前向**（BatchNorm 走 batch 统计，不是 running 统计）
//! 2. **组合损失**（数据项 + 物理项 + 权重斜坡）
//! 3. **裁剪后的梯度**——反向传播全对，且梯度裁剪是 PyTorch 的**全局范数**语义。
//!    Burn 自带的裁剪是逐张量的，用了它这一项会红
//! 4. **Adam 更新后的全部参数**（含 L2-in-gradient 的 weight decay 与偏差修正）
//!
//! 判据是 manifest 声明的 `|a-b| <= tol + tol*|b|`，tol = 1e-5。

mod common;

use burn::backend::{Autodiff, NdArray};
use burn::module::{Module, ModuleVisitor, ParamId};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

use common::{assert_close4, open_npz, read, read_scalar, state_dict, tensor4};
use eser::model::{DenseEd, DenseEdConfig};
use eser::physics::{scalar, ChannelOrder, PhysicsLoss};
use eser::store::LoadState;
use eser::train::clip_grad_norm;

/// 训练步要走反向传播，所以后端必须叠上 autodiff。
/// 顺带一提：Burn 靠 `B::ad_enabled()` 区分 BatchNorm 的训练/推理模式，
/// 用这个后端就自动是训练模式，正好对上 Python 侧的 `model.train()`。
type AD = Autodiff<NdArray<f32>>;
type Plain = NdArray<f32>;

/// 用 autodiff 的测试必须串行跑。
///
/// **这是 Burn 0.18 的问题，不是测试写法问题**：`AutodiffServer` 的图内存管理
/// （`burn-autodiff/src/runtime/server.rs`）在多个图并存时会把别的图的节点当成
/// 不可达而回收掉，随后 `backward` 里 `self.steps.remove(&node_id).expect(...)`
/// 直接 panic：`Node should have a step registered`。cargo 默认并行跑测试，
/// 于是这类失败随机出现（实测 12 次里 2 次）。
///
/// 用一把进程内的锁把它们串起来。锁中毒（前一个测试 panic）时继续用，
/// 否则一个失败会级联成一片失败，掩盖真正的首因。
static AUTODIFF_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn autodiff_guard() -> std::sync::MutexGuard<'static, ()> {
    AUTODIFF_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// 与 export_golden.py 的 layer7_train_step 一一对应
const LR: f64 = 3e-3;
const WEIGHT_DECAY: f32 = 5e-4;
const GRAD_CLIP: f64 = 1.0;
const DX: f64 = 0.03;
const DY: f64 = 0.05;

fn config() -> DenseEdConfig {
    let mut cfg = DenseEdConfig::new(2, 3, vec![2, 2, 2]);
    cfg.growth_rate = 4;
    cfg.init_features = 8;
    cfg.output_size = Some([32, 32]);
    cfg
}

/// 把模型里指定 id 的参数值收出来。
///
/// **必须按 id 过滤**：`ModuleVisitor` 会把 BatchNorm 的 running_mean / running_var
/// 也当浮点参数遍历（本模型 50 个可训练参数 + 24 个 running 统计量 = 74 个张量），
/// 而 PyTorch 的 `named_parameters()` 不含 buffer，参考数据只有 50 项。
///
/// Burn 的 visitor 只给 `ParamId` 不给名字，所以比对方式是：逐个参数张量与参考数据
/// 做集合匹配——每个参考张量都要在模型里找到元素数相同且数值一致的那一个。
/// 代价是「张量对错了位但数值恰好都对」抓不到；而位置对错在前向、损失、梯度三层
/// 已经被逐元素钉死了，这里只是最后一道确认。
struct ParamCollector<'a, Bk: AutodiffBackend> {
    keep: &'a [ParamId],
    values: Vec<Vec<f32>>,
    _p: std::marker::PhantomData<Bk>,
}

impl<Bk: AutodiffBackend> ModuleVisitor<Bk> for ParamCollector<'_, Bk> {
    fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<Bk, D>) {
        if self.keep.contains(&id) {
            self.values
                .push(tensor.clone().into_data().to_vec().unwrap());
        }
    }
}

fn collect_params(model: &DenseEd<AD>, keep: &[ParamId]) -> Vec<Vec<f32>> {
    let mut v = ParamCollector::<AD> {
        keep,
        values: Vec::new(),
        _p: std::marker::PhantomData,
    };
    model.visit(&mut v);
    v.values
}

/// 每个参考张量都要在实际参数里找到数值一致的对应项。
fn assert_params_match(tag: &str, got: &[Vec<f32>], want: &[(String, Vec<f32>)]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{}：模型有 {} 个参数张量，参考数据有 {} 个",
        tag,
        got.len(),
        want.len()
    );
    let mut used = vec![false; got.len()];
    for (name, w) in want {
        let hit = got.iter().enumerate().position(|(i, g)| {
            !used[i]
                && g.len() == w.len()
                && g.iter()
                    .zip(w.iter())
                    .all(|(a, b)| (*a as f64 - *b as f64).abs() <= 1e-5 + 1e-5 * (*b as f64).abs())
        });
        match hit {
            Some(i) => used[i] = true,
            None => panic!(
                "{}：参考参数 {:?}（{} 个元素）在模型里找不到数值一致的对应项",
                tag,
                name,
                w.len()
            ),
        }
    }
}

/// 读参考张量。`prefix` 是完整前缀——参数是 `step/before/param/`，
/// 梯度是 `step/grad/`（导出脚本对这两组用了不同的命名，不能想当然拼 `param`）。
fn reference_params(prefix: &str) -> Vec<(String, Vec<f32>)> {
    let mut npz = open_npz();
    let full = prefix.to_string();
    let names: Vec<String> = npz
        .names()
        .unwrap()
        .into_iter()
        .filter(|n| n.starts_with(&full))
        .collect();
    names
        .into_iter()
        .map(|n| {
            let key = n
                .trim_end_matches(".npy")
                .trim_start_matches(&full[..])
                .to_string();
            let a: ndarray::ArrayD<f32> = npz.by_name(&n).unwrap();
            // 本模型里所有 2D 参数都是线性层权重。PyTorch 存 [out, in]，Burn 存
            // [in, out]，**梯度和更新后的参数同样是转置关系**，比对前统一转置过来
            let data: Vec<f32> = if a.shape().len() == 2 {
                let (r, c) = (a.shape()[0], a.shape()[1]);
                let flat: Vec<f32> = a.iter().copied().collect();
                (0..c)
                    .flat_map(|j| (0..r).map(move |i| (i, j)))
                    .map(|(i, j)| flat[i * c + j])
                    .collect()
            } else {
                a.iter().copied().collect()
            };
            (key, data)
        })
        .collect()
}

#[test]
fn one_training_step_matches_reference() {
    let _serial = autodiff_guard();
    let device = Default::default();
    let mut npz = open_npz();

    // ---- 1. 建模并装入训练前的参数 ----
    let sd = state_dict(&mut npz, "step/before");
    let mut model: DenseEd<AD> = config().init(&device).expect("建模失败");
    model.load_state(&sd, "").expect("装入训练前参数失败");
    sd.ensure_fully_used().expect("有参数没被装入");

    let x: Tensor<Plain, 4> = tensor4(&mut npz, "step/input");
    let y: Tensor<Plain, 4> = tensor4(&mut npz, "step/target");
    let mask: Tensor<Plain, 4> = tensor4(&mut npz, "step/mask");
    let to_ad = |t: Tensor<Plain, 4>| Tensor::<AD, 4>::from_data(t.into_data(), &device);
    let (x, y, mask) = (to_ad(x), to_ad(y), to_ad(mask));

    // ---- 2. 训练模式前向 ----
    let out = model.forward(x);
    assert_close4(
        "step/forward",
        out.clone().inner(),
        &read(&mut npz, "step/forward"),
    );

    // ---- 3. 组合损失 ----
    let (loss, _) = PhysicsLoss::<AD> {
        epoch: 50,
        total_epochs: 200,
        channel_order: ChannelOrder::Pvv,
        dx: DX,
        dy: DY,
        physics_warmup: 0.1,
        mask: Some(mask),
        y_min: Some(vec![-10.0, -1.0, -0.4]),
        y_max: Some(vec![30.0, 2.0, 0.6]),
        ..Default::default()
    }
    .compute(out, y);
    let got_loss = scalar(loss.clone());
    let want_loss = read_scalar(&mut npz, "step/loss");
    assert!(
        (got_loss - want_loss).abs() <= 1e-5 + 1e-5 * want_loss.abs(),
        "step/loss: got={} want={}",
        got_loss,
        want_loss
    );

    // ---- 4. 反向 + 全局梯度裁剪 ----
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let (grads, total_norm) = clip_grad_norm(&model, grads, GRAD_CLIP);
    assert!(
        total_norm.is_finite() && total_norm > 0.0,
        "全局梯度范数应为有限正数，实得 {}",
        total_norm
    );

    let want_grads = reference_params("step/grad/");
    let mut collector = GradCollector::<AD>::new(&grads);
    model.visit(&mut collector);
    assert_params_match("step/grad", &collector.values, &want_grads);
    let trainable = collector.ids.clone();

    // ---- 5. Adam 一步 ----
    let mut optim = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        // Burn 的默认 epsilon 不是 1e-8，必须显式对齐 PyTorch
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(WEIGHT_DECAY)))
        .init::<AD, DenseEd<AD>>();
    let model = optim.step(LR, model, grads);

    assert_params_match(
        "step/after",
        &collect_params(&model, &trainable),
        &reference_params("step/after/param/"),
    );
}

#[test]
fn gradient_reference_keeps_its_discriminating_power() {
    // **参考数据本身可能验不动东西**——这个项目已经吃过一次亏：`_fixed_init` 把
    // 归一化层的 gamma 也置成 ±0.1，整网被压平，「漏掉一整级上采样」只差 4.7e-7，
    // 在 1e-5 的容差里跟没改坏分不开。
    //
    // 梯度这一层同理：某个张量的参考值若远小于容差，返回全零也能通过，那一项就是
    // 摆设。这里把「摆设项」的数量钉死在当前值（1 项，
    // `transformer_bottleneck.mlp.3.bias`，max|g| = 3.6e-9——在 f32 下本来就与 0
    // 无法区分，收紧容差只会去测浮点噪声）。数量一旦上升就说明参考数据又被压平了。
    let vacuous: Vec<(String, f32)> = reference_params("step/grad/")
        .into_iter()
        .filter_map(|(name, v)| {
            let worst = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            // 判据与比对函数一致：全零向量能否通过 |0 - w| <= tol + tol*|w|
            let passes_as_zero = v
                .iter()
                .all(|w| (*w as f64).abs() <= 1e-5 + 1e-5 * (*w as f64).abs());
            passes_as_zero.then_some((name, worst))
        })
        .collect();

    assert_eq!(
        vacuous.len(),
        1,
        "对「返回全零」也能通过的梯度参考项有 {} 个（此前是 1 个）：{:?}。\
         数量上升通常意味着参考数据被压平了，整层验收随之失效",
        vacuous.len(),
        vacuous
    );
    assert_eq!(
        vacuous[0].0, "transformer_bottleneck.mlp.3.bias",
        "无鉴别力的那一项换了个张量（{}），说明梯度量级分布变了，值得查一查",
        vacuous[0].0
    );
}

#[test]
fn global_clipping_scales_to_the_threshold() {
    let _serial = autodiff_guard();
    // 这条守的是 P5 最容易踩的坑：Burn 自带的 `GradientClipping::Norm` 逐张量裁剪，
    // PyTorch 的 `clip_grad_norm_` 算全局范数。逐张量裁的话，裁完之后**总范数不等于
    // 阈值**（每个张量各自被压到阈值以下，合起来可以远大于阈值）。
    let device = Default::default();
    let model = loaded_model(&device);

    let total = clip_grad_norm(&model, fresh_grads(&model, &device), f64::INFINITY).1;
    let threshold = total / 2.0;
    let (clipped, reported) = clip_grad_norm(&model, fresh_grads(&model, &device), threshold);
    assert!(
        (reported - total).abs() < 1e-6 * total,
        "两次算出的全局范数应当一致：{} vs {}",
        reported,
        total
    );

    let after = global_norm(&model, &clipped);
    assert!(
        (after - threshold).abs() < 1e-4 * threshold,
        "全局裁剪后的总范数应当等于阈值 {}，实得 {}（逐张量裁的话会明显偏大）",
        threshold,
        after
    );
}

#[test]
fn clipping_leaves_small_gradients_alone() {
    let _serial = autodiff_guard();
    // 阈值远大于实际范数时不该动任何梯度——PyTorch 的判据是 `coef < 1`
    let device = Default::default();
    let model = loaded_model(&device);
    let before = global_norm(&model, &fresh_grads(&model, &device));
    let (untouched, _) = clip_grad_norm(&model, fresh_grads(&model, &device), before * 10.0);
    let after = global_norm(&model, &untouched);
    assert!(
        (after - before).abs() < 1e-9 * before,
        "阈值远大于范数时不该缩放，{} -> {}",
        before,
        after
    );
}

// ------------------------------------------------------------------ 测试工具

fn loaded_model(device: &<AD as burn::tensor::backend::Backend>::Device) -> DenseEd<AD> {
    let mut npz = open_npz();
    let sd = state_dict(&mut npz, "step/before");
    let mut model: DenseEd<AD> = config().init(device).expect("建模失败");
    model.load_state(&sd, "").expect("装入参数失败");
    model
}

/// 重新跑一遍前向+反向，拿一份新的梯度（`GradientsParams` 会被裁剪消耗掉）。
fn fresh_grads(
    model: &DenseEd<AD>,
    device: &<AD as burn::tensor::backend::Backend>::Device,
) -> GradientsParams {
    let mut npz = open_npz();
    let x: Tensor<Plain, 4> = tensor4(&mut npz, "step/input");
    let y: Tensor<Plain, 4> = tensor4(&mut npz, "step/target");
    let x = Tensor::<AD, 4>::from_data(x.into_data(), device);
    let y = Tensor::<AD, 4>::from_data(y.into_data(), device);
    let (loss, _) = PhysicsLoss::<AD> {
        dx: DX,
        dy: DY,
        ..Default::default()
    }
    .compute(model.forward(x), y);
    GradientsParams::from_grads(loss.backward(), model)
}

fn global_norm(model: &DenseEd<AD>, grads: &GradientsParams) -> f64 {
    let mut acc = GradCollector::<AD>::new(grads);
    model.visit(&mut acc);
    acc.values
        .iter()
        .flat_map(|v| v.iter())
        .map(|&g| (g as f64) * (g as f64))
        .sum::<f64>()
        .sqrt()
}

/// 收集每个参数对应的梯度。
struct GradCollector<'a, Bk: AutodiffBackend> {
    grads: &'a GradientsParams,
    values: Vec<Vec<f32>>,
    /// 有梯度的参数 id——用来把 BatchNorm 的 running 统计量排除在参数比对之外
    ids: Vec<ParamId>,
    _p: std::marker::PhantomData<Bk>,
}

impl<'a, Bk: AutodiffBackend> GradCollector<'a, Bk> {
    fn new(grads: &'a GradientsParams) -> Self {
        Self {
            grads,
            values: Vec::new(),
            ids: Vec::new(),
            _p: std::marker::PhantomData,
        }
    }
}

impl<Bk: AutodiffBackend> ModuleVisitor<Bk> for GradCollector<'_, Bk> {
    fn visit_float<const D: usize>(&mut self, id: ParamId, _tensor: &Tensor<Bk, D>) {
        if let Some(g) = self.grads.get::<Bk::InnerBackend, D>(id) {
            self.values.push(g.into_data().to_vec().unwrap());
            self.ids.push(id);
        }
    }
}
