//! 端到端对照（方案 §4.3）的 Rust 侧。
//!
//! ```text
//! cargo run -p eser --release --example e2e_compare -- tests/golden/e2e.npz
//! ```
//!
//! 从 PyTorch 冻结的**同一组初始权重**、同一份数据出发，训练同样的 20 epoch，
//! 逐 epoch 比对 loss / R² / RMSE。允许浮点累加顺序带来的漂移，但趋势和量级必须一致。
//!
//! 这不是单元测试（跑一次要几十秒），所以做成 example。
//!
//! **参考数据 `tests/golden/e2e.npz` 是冻结的**：产它的 `tests/e2e_compare.py` 随
//! PyTorch 版一起退役了，既不能重新生成，也没有第二个实现能交叉验证这批数值。改坏
//! Rust 训练循环时它照样会报出来，但别把它当成「随时可复现的对照」。

use std::fs::File;

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;

use eser::data::RunningMetrics;
use eser::model::{DenseEd, DenseEdConfig};
use eser::physics::{scalar, ChannelOrder, PhysicsLoss};
use eser::store::{LoadState, StateDict};
use eser::train::clip_grad_norm;

type AD = Autodiff<NdArray<f32>>;

const EPOCHS: usize = 20;
const LR: f64 = 3e-3;
const WEIGHT_DECAY: f32 = 5e-4;
const GRAD_CLIP: f64 = 1.0;
const DX: f64 = 0.03;
const DY: f64 = 0.05;

fn get(npz: &mut NpzReader<File>, key: &str) -> ArrayD<f32> {
    npz.by_name(&format!("{}.npy", key))
        .unwrap_or_else(|e| panic!("参考数据里没有 {:?}: {}", key, e))
}

fn get_f64(npz: &mut NpzReader<File>, key: &str) -> Vec<f64> {
    let a: ArrayD<f64> = npz
        .by_name(&format!("{}.npy", key))
        .unwrap_or_else(|e| panic!("参考数据里没有 {:?}: {}", key, e));
    a.iter().copied().collect()
}

fn tensor4(
    npz: &mut NpzReader<File>,
    key: &str,
    device: &<AD as burn::tensor::backend::Backend>::Device,
) -> Tensor<AD, 4> {
    let a = get(npz, key);
    let s = a.shape().to_vec();
    let flat: Vec<f32> = a.iter().copied().collect();
    Tensor::<AD, 1>::from_data(TensorData::from(flat.as_slice()), device)
        .reshape([s[0], s[1], s[2], s[3]])
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        panic!("用法：cargo run -p eser --release --example e2e_compare -- <e2e.npz 路径>")
    });
    let mut npz = NpzReader::new(File::open(&path).expect("打不开参考数据")).unwrap();
    let device = Default::default();

    // ---- 装入 Python 冻结的初始权重 ----
    let mut sd = StateDict::new();
    let names: Vec<String> = npz
        .names()
        .unwrap()
        .into_iter()
        .filter(|n| n.starts_with("param/"))
        .collect();
    for full in names {
        let key = full
            .trim_end_matches(".npy")
            .trim_start_matches("param/")
            .to_string();
        let a: ArrayD<f32> = npz.by_name(&full).unwrap();
        sd.insert(key, a.shape().to_vec(), a.iter().copied().collect());
    }

    let mut cfg = DenseEdConfig::new(2, 3, vec![2, 2, 2]);
    cfg.growth_rate = 4;
    cfg.init_features = 8;
    let [_, _, h, w] = {
        let a = get(&mut npz, "input");
        [a.shape()[0], a.shape()[1], a.shape()[2], a.shape()[3]]
    };
    cfg.output_size = Some([h, w]);
    let mut model: DenseEd<AD> = cfg.init(&device).expect("建模失败");
    model.load_state(&sd, "").expect("装入初始权重失败");
    sd.ensure_fully_used().expect("有初始权重没被装入");

    let x = tensor4(&mut npz, "input", &device);
    let y = tensor4(&mut npz, "target", &device);
    let mask = tensor4(&mut npz, "mask", &device);
    let y_min: Vec<f32> = get(&mut npz, "y_min").iter().copied().collect();
    let y_max: Vec<f32> = get(&mut npz, "y_max").iter().copied().collect();
    let n_out = y.dims()[1];

    let mut optim = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(WEIGHT_DECAY)))
        .init::<AD, DenseEd<AD>>();

    let (mut losses, mut r2s, mut rmses) = (Vec::new(), Vec::new(), Vec::new());

    for epoch in 1..=EPOCHS {
        let out = model.forward(x.clone());
        let (loss, _) = PhysicsLoss::<AD> {
            epoch,
            total_epochs: EPOCHS,
            channel_order: ChannelOrder::Pvv,
            dx: DX,
            dy: DY,
            physics_warmup: 0.1,
            mask: Some(mask.clone()),
            y_min: Some(y_min.clone()),
            y_max: Some(y_max.clone()),
            ..Default::default()
        }
        .compute(out, y.clone());
        losses.push(scalar(loss.clone()));

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let (grads, _) = clip_grad_norm(&model, grads, GRAD_CLIP);
        model = optim.step(LR, model, grads);

        // 评估走推理模式：`valid()` 把模型换到非 autodiff 后端，
        // BatchNorm 于是用 running 统计量，与 Python 侧的 model.eval() 一致
        let eval_model = model.valid();
        let pred = eval_model.forward(x.clone().inner());
        let mut m = RunningMetrics::new(n_out);
        m.update(pred, y.clone().inner(), Some(mask.clone().inner()));
        r2s.push(m.r2());
        rmses.push(m.rmse());
    }

    // ---- 比对 ----
    let want_loss = get_f64(&mut npz, "loss");
    let want_r2 = get_f64(&mut npz, "r2");
    let want_rmse = get_f64(&mut npz, "rmse");

    println!(
        "epoch |        loss (rust / python)         |     R² (rust / python)      | RMSE 相对差"
    );
    let mut worst_loss: f64 = 0.0;
    let mut worst_rmse: f64 = 0.0;
    for i in 0..EPOCHS {
        let dl = (losses[i] - want_loss[i]).abs() / want_loss[i].abs().max(1e-12);
        let dr = (rmses[i] - want_rmse[i]).abs() / want_rmse[i].abs().max(1e-12);
        worst_loss = worst_loss.max(dl);
        worst_rmse = worst_rmse.max(dr);
        if i < 3 || i == EPOCHS - 1 {
            println!(
                "{:5} | {:.6e} / {:.6e} | {:+.6} / {:+.6} | {:.2e}",
                i + 1,
                losses[i],
                want_loss[i],
                r2s[i],
                want_r2[i],
                dr
            );
        }
    }
    println!();
    println!(
        "20 个 epoch 上的最大相对偏差：loss {:.3e}，RMSE {:.3e}",
        worst_loss, worst_rmse
    );
    println!(
        "loss 首末：rust {:.6e} -> {:.6e}；python {:.6e} -> {:.6e}",
        losses[0],
        losses[EPOCHS - 1],
        want_loss[0],
        want_loss[EPOCHS - 1]
    );
}
