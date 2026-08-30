# 基于物理信息的 T 型管流场深度学习代理模型

(Physics-Informed T-Pipe Fluid Surrogate Model)

用 CNN 编解码网络代理 T 型管内不可压缩流场（压力 + 速度）的 CFD 求解，
并以 PINN 残差（连续性方程 + 稳态 N-S 动量方程）作为物理正则项。

> **溯源**：本项目 fork 自 [`cics-nd/cnn-surrogate`](https://github.com/cics-nd/cnn-surrogate)
> （Zhu & Zabaras 的贝叶斯卷积编解码代理模型，原用于 KLE 随机渗流场）。
> 代码中的 `--kle` 参数、`images/kle*.png` 等是上游遗留。

---

## 一、开始之前：三个必须确认的配置

代码本身跑通很容易，但下面三件事配错了，指标再漂亮也没有意义。

### 1. 通道顺序（`--channel-order`）

物理损失需要知道哪个通道是压力、哪个是速度分量。**没有任何办法自动推断**，
必须由你指定：

| 取值 | 含义 |
|---|---|
| `pvv`（默认） | `output` 的通道顺序是 (P, Vx, Vy) |
| `vvp` | 通道顺序是 (Vx, Vy, P) |

配错了不会报任何错，只会让散度约束去算 `∂P/∂x + ∂Vx/∂y` 这种无意义的量，
把模型往错误方向拉。**请先打开数据集确认**：

```python
import h5py
with h5py.File('dataset/你的数据.hdf5') as f:
    print(f['output'].shape)                 # (N, C, H, W)
    for c in range(f['output'].shape[1]):
        print(c, f['output'][:, c].min(), f['output'][:, c].max())
# 压力通道的量级通常比速度大 1~2 个数量级
```

### 2. 计算域物理尺寸（`--domain-size Ly Lx`）

物理残差在**有量纲**的场上计算，网格间距由 `dx = Lx / (W-1)` 得到。
默认 `1.0 1.0` 只是占位值 —— 若真实管道是 0.1 m × 0.05 m，就要写
`--domain-size 0.05 0.1`，否则 `dx` 差几个数量级，雷诺数完全对不上。
同理 `--rho`、`--nu` 需要填真实工况值。

### 3. 几何掩码（`--t-geometry`）

默认几何用**归一化比例**描述，从原始 137×137 网格反推得到，任意分辨率下
覆盖同一物理区域（约 44.9%）。若你的管道形状不同，用 JSON 覆盖：

```bash
python train.py --t-geometry '{"vertical":[0.0,0.99,0.0,0.26],"horizontal":[0.37,0.64,0.26,1.0]}'
```

每段是 `[y0, y1, x0, x1]` 的比例（右开区间）。想在整幅网格上训练就加 `--no-mask`。

---

## 二、核心技术特性

- **TransUNet 风格架构**：DenseNet 编解码骨干 + U-Net skip connection +
  最深处的全局多头自注意力瓶颈。自注意力带 **2D/3D sin-cos 位置编码**
  （无参数、对分辨率自适应），否则注意力对 token 排列等变，全局信息里
  不含空间位置。
- **物理约束在有量纲场上计算**：损失内部先用训练集统计量把 [0,1] 的预测
  反归一化回真实压力/速度，再用真实 `dx/dy` 求残差。逐通道归一化会把
  Vx、Vy 缩放到不同尺度，直接在归一化空间求 `div(v)` 是错的。
- **物理项量级自适应**（`--physics-loss-scaling ema`）：残差绝对量级随
  `1/dx`、`1/dx²` 变化好几个数量级，滑动估计把各项拉回 O(1)，
  `--div-weight` 之类的权重才与量纲无关、可跨数据集复用。
- **严格的计算域掩码**：数据项、散度、动量、平滑四项**全部**受掩码约束，
  且分母是掩码内像素数。物理残差额外把掩码向内腐蚀 1 格，保证 3×3 差分
  模板整个落在域内，不会把墙外的数值漏进损失。
- **SiLU 全覆盖**：2D 与 3D 骨干均使用 SiLU，保证二阶导连续 —— 这是 PINN
  能用有限差分近似高阶导数的前提。
- **任意分辨率**：2D / 3D 模型都对输出尺寸做兜底插值，奇偶数网格均可。

---

## 三、目录结构

```text
.
├── dataset/                    # HDF5 数据集（自动生成的 *.stats.npz 缓存也在这里）
├── experiments/                # 实验输出：args.txt / checkpoints / training/
├── models/
│   ├── attention.py            # Transformer 瓶颈 + 空间注意力（2D/3D 共用，含位置编码）
│   ├── dense_ed.py             # 2D DenseED + skip + Transformer
│   └── dense_ed_3d.py          # 3D 体素版本
├── utils/
│   ├── geometry.py             # 计算域几何（矩形段比例 或 .npy 栅格，唯一真源）
│   ├── data_common.py          # 全量统计、归一化/反归一化、数据划分
│   ├── load_data.py            # 2D 加载器（惰性 HDF5，句柄按 worker 缓存）
│   ├── load_data_3d.py         # 3D 加载器（签名与 2D 完全一致）
│   ├── physics_loss.py         # 散度 / N-S 动量 / 平滑 / 重建 + 量级归一化
│   ├── transport.py            # 对流扩散残差（污染物输运，--task dispersion）
│   ├── misc.py                 # 目录工具 + 分通道 R²/RMSE 累计器
│   └── plot.py / plot_3d.py    # 预测对比图与训练曲线（3D 版直接复用 2D 实现）
├── tests/
│   ├── make_fake_data.py       # 生成合成数据集，用于冒烟测试
│   ├── test_regressions.py     # 63 项回归测试，覆盖审查报告每一条问题
│   └── test_transport.py       # 39 项对流扩散 / 栅格掩码测试（含解析解验证）
├── train.py                    # 主训练脚本
├── surrogate_predict.py        # 离线推理（输出反归一化回物理量纲）
├── diagnose.py                 # 诊断报告（含配置合理性检查）
├── gui_trainer.py              # Tkinter 训练控制台
└── args.py / args_3d.py        # 统一参数定义（args_3d 只改 --ndim 默认值）
```

---

## 四、工作流程

### 1. 环境

```bash
pip install torch numpy h5py matplotlib
# GUI 需要 tkinter；Linux 上通常是 python3-tk
```

### 2. 数据

把 HDF5 放进 `dataset/`。文件需要包含：

- `output`：`(N, C, H, W)` 或 `(N, C, D, H, W)` 的目标流场
- `input` 或 `sensor_input`：网络输入
- （3D 可选）`spatial_input`：带空间结构的输入，会与标量向量自动拼接

首次加载时会全量扫描一遍算逐通道极值，并缓存成 `dataset/<file>.output.stats.npz`。
**数据改了记得删缓存**（缓存会按 mtime 自动失效，手动删更保险）。

### 3. 训练

```bash
# 单文件，自动切 20% 做验证
python train.py --train-file dataset/p_v_256.hdf5 \
                --channel-order pvv \
                --domain-size 1.0 1.0 --rho 1.0 --nu 0.01 \
                --epochs 200 --batch-size 8

# 有独立测试集
python train.py --train-file dataset/train.hdf5 --test-file dataset/test.hdf5

# 3D
python train.py --ndim 3 --train-file dataset/10243d.hdf5

# 图形界面
python gui_trainer.py
```

#### 污染物输运（两阶段）

除了流场，代码也支持污染物浓度场的对流扩散代理模型。流场只取决于通风工况、
且是稳态的；浓度场取决于流场 + 源、且是瞬态的。绑在一个网络里等于让不变的
东西陪着变的东西重算，所以分两阶段——换个泄漏点位时**阶段一不用重跑**。

```bash
# 阶段一：工况 -> (P, Vx, Vy, T)。高 Re 湍流下层流动量残差不成立，用 --ns-weight 0
#         关掉，只保留连续性方程（它对不可压缩 RANS 时均场严格成立）
python train.py --task flow --channel-order pvvt --ns-weight 0                 --train-file dataset/stage1.hdf5                 --domain-mask dataset/station_mask.npy                 --domain-size 12.0 60.0 --blocks "[3,4,6,4,3]"

# 阶段二：(Vx, Vy, 源位置图, t) -> C
python train.py --task dispersion --channel-order c                 --train-file dataset/stage2.hdf5                 --diffusivity 2e-5 --velocity-channels 0 1                 --domain-mask dataset/station_mask.npy                 --domain-size 12.0 60.0 --blocks "[3,4,6,4,3]"
```

几个要点：

- **速度场从输入通道来，不在 output 里**（`--velocity-channels` 指定位置）。
  取错通道不报错，只会让物理约束作用在错误的场上。
- **时刻 t 是一个输入通道**（广播成常数图）。稳态就是不给 t，同一套代码覆盖
  稳态与瞬态，没有自回归 rollout 的误差累积。
- **源强不用作为变量采样**：被动标量下方程对 C 是线性的，训练时源强固定为 1，
  使用时线性缩放即可。前提是污染物本身不是浮力源（火灾烟气不适用）。
- **`--blocks` 要够深**：512×512 用默认 `[3,6,3]` 时瓶颈是 128×128=16384 token，
  超过 `--attn-max-tokens` 会被静默跳过，注意力层白加。`[3,4,6,4,3]` 降到 4096。
- 掩码用 `.npy` 栅格（0/1 数组，分辨率不必与训练网格一致，做最近邻重采样），
  站厅/站台这类平面用矩形段拼不出来。

> **测试集会强制复用训练集的归一化统计量**，两边标度一致，R² 才有可比性。

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--val-split` | 0.2 | 训练/测试同文件时切多少做验证；0 表示不切（此时 R² 是样本内指标） |
| `--channel-order` | pvv | 输出通道语义 |
| `--domain-size` | 1.0 1.0 | 计算域物理尺寸 (Ly Lx) |
| `--div-weight` / `--ns-weight` / `--smooth-weight` | 0.05 / 0.005 / 0.005 | 物理项权重（已经过量级归一化） |
| `--physics-warmup` | 0.1 | 前 10% 的 epoch 只拟合数据，不加物理约束 |
| `--grad-clip` | 1.0 | 梯度范数裁剪；PINN 的二阶项容易发散 |
| `--plot-fn` | imshow | `contourf` 也已修好色阶对齐，但 imshow 更快更准 |
| `--no-mask` | 关 | 在整幅网格上训练与评估 |
| `--task` | flow | `flow`=流场（连续性+动量）；`dispersion`=污染物输运（对流扩散） |
| `--domain-mask` | 无 | 计算域掩码，可给 `.npy` 栅格（任意形状）或 JSON 矩形段；`--t-geometry` 的别名 |
| `--diffusivity` | 1e-5 | 扩散系数 D（`dispersion` 用） |
| `--advdiff-weight` | 0.05 | 对流扩散残差权重 |
| `--velocity-channels` | 0 1 | `dispersion` 下速度场在**输入**张量的哪两个通道 |

### 4. 诊断

```bash
python diagnose.py
```

会读取最新实验，先做**配置合理性检查**（训练测试是否同源、通道顺序、
domain_size 是否还是默认值……），再报 R²、分通道 R²、RMSE 与物理残差，
最后给调优建议。

### 5. 推理

```bash
python surrogate_predict.py \
    --ckpt-file experiments/run01/checkpoints/model_best.pth \
    --data-file dataset/new_case.hdf5 \
    --out-file experiments/run01/pred.hdf5
```

输出文件包含：

- `predictions`：**已反归一化的有量纲流场**（要保留 [0,1] 加 `--keep-normalized`）
- `y_min` / `y_max`：归一化统计量，方便下游自行换算
- attrs：`channel_order`、`ndim`、超参 JSON 等

模型超参从 checkpoint 自身读取，不需要在命令行重复一遍。

### 6. 结果解读

- 图像路径：`experiments/<实验名>/training/predictions/`
- 第 1 行 CFD 真值，第 2 行代理模型预测（**与第 1 行同色阶**），
  第 3 行残差（以 0 为中心的对称 diverging 色阶，接近纯白即拟合良好）
- 曲线：`r2.pdf`、`rmse.pdf`、`r2_per_channel.pdf`、`div_rms.pdf`、`ns_rms.pdf`

---

## 五、测试

```bash
python -m tests.test_regressions          # 63 项回归测试
python -m tests.test_transport            # 39 项对流扩散 / 栅格掩码测试
python tests/make_fake_data.py --out-dir /tmp/fake && \
  python train.py --train-file /tmp/fake/fake2d.hdf5 --epochs 2 \
                  --exp-name smoke --num-workers 0    # 端到端冒烟
```

---

## 六、已知限制

- **物理常数需要人工确认**：`--rho` / `--nu` / `--domain-size` 没有办法从
  HDF5 推断，填错了物理残差的绝对数值就没有参考意义（相对下降趋势仍可用）。
- **注意力显存**：Transformer 瓶颈是 O(N²)。token 数超过 `--attn-max-tokens`
  （默认 8192）时会自动跳过该层并打印提示，而不是崩掉。
- **3D 各向异性**：3D 卷积核是 `(1,3,3)`，深度方向不做卷积；物理残差也只在
  x-y 平面上计算，尚未包含 `∂/∂z` 项。真正的三维 N-S 约束还需要扩展。
- **旧 checkpoint 不兼容**：2D 模型加了 skip connection、上采样从
  `ConvTranspose` 换成 `Upsample+Conv`，权重结构已变。旧权重需要重训。
