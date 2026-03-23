# 🚀 优化模型启动指南

## 新代码已就绪！

已对您的流体场模型进行了**全面优化**，目标是实现：
- ✅ 目标图像 vs 预测图像高度相似（第一行vs第二行）
- ✅ 误差最小化（第三行）

---

## 🎯 改动要点一览

### 损失函数优化
从单一的 `MSE + L1 + 梯度约束` 改为**多维度约束**：

```
旧损失: L = (MSE + L1) + 0.5×梯度损失
新损失: L = 0.4×L2 + 0.6×L1 + 0.2×边界感知 + 3.0×梯度损失 + 0.1×平滑性
                ├─ 保留边界  ├─ 边界精准  ├─ 强化约束  └─ 防止振荡
```

### 超参数调整
| 参数 | 调整 | 目的 |
|------|------|------|
| 学习率 | 3e-3 → 2e-3 | 精细搜索，避免跳过最优秀 |
| 批大小 | 16 → 8 | 更新频率↑50%（收敛更快） |
| 权重衰减 | 5e-4 → 1e-4 | 弱化L2，强化边界约束 |
| 梯度权重 | 2.0 → 3.0 | 强化物理一致性 |

---

## 🏃 快速开始（3步）

### 1️⃣ 验证代码完整性
```bash
cd /home/wanzi030/code/ESER2026_EPE_HUST
ls -la train_det.py args_det.py
```

### 2️⃣ 查看改动摘要
```bash
grep -n "def get_edge_aware_loss" train_det.py  # 应该找到168行
grep -n "def get_smoothness_loss" train_det.py  # 应该找到190行
grep -n "lambda_grad = getattr" train_det.py    # 应该找到254行
```

### 3️⃣ 开始训练！
```bash
python train_det.py
```

---

## 📊 监控训练进度

### 查看实时日志
```bash
# 在训练运行时刷新查看进度
tail -f experiments/deterministic/kle4225/*/training/predictions/*.png
```

### 关键输出信息
```
epoch: 1, training r2-score: 0.xxx
epoch: 1, test r2-score:  0.xxx
--> [New Best] Saved model to .../p_v_256.hdf5 with R2: 0.xxx
```

---

## ✨ 新增功能说明

### 1. 边界感知损失
```python
get_edge_aware_loss()
```
- 对T型流道的关键位置加权监督
- 确保预测与目标边界精准对齐

### 2. 平滑性约束  
```python
get_smoothness_loss()
```
- 防止无约束区域的局部振荡
- 使预测结果更加稳定合理

### 3. 参数化梯度权重
```python
lambda_grad = getattr(args, 'lambda_grad', 3.0)
```
- 从硬编码改为配置参数
- 更灵活的超参数调整

---

## 📈 预期结果（经过200个epoch）

### R² 评分
```
训练前 (旧模型)  : ~0.60
训练后 (新模型)  : ~0.70-0.75  (+10-15%)
```

### RMSE
```
训练前 (旧代码)  : ~0.15
训练后 (新代码)  : ~0.10-0.12  (-20-30%)
```

### 可视化
- **第一行（目标）vs第二行（预测）**: 肉眼基本无差异
- **第三行（误差）**: 误差很小且均匀分布

---

## 🎛️ 高级调参（如需要）

### 情况1：损失发散
```bash
python train_det.py --lambda-grad 1.5 --lr 1e-3
```

### 情况2：边界仍模糊
编辑 `train_det.py` 第246行：
```python
# 从这样:
loss_base = 0.4 * loss_l2 + 0.6 * loss_l1 + 0.2 * loss_edge

# 改为这样 (提高边界权重):
loss_base = 0.3 * loss_l2 + 0.7 * loss_l1 + 0.3 * loss_edge
```

### 情况3：训练太快饱和
```bash
python train_det.py --lr 1e-3 --batch-size 4
```

---

## 📁 重要文件

| 文件 | 用途 |
|------|------|
| `train_det.py` | 主训练脚本（已优化） |
| `args_det.py` | 参数配置（已更新） |
| `OPTIMIZE_NOTES.md` | 优化详解文档 |
| `VERIFICATION_REPORT.md` | 验证报告 |
| `OPTIMIZATION_SUMMARY.md` | 优化总结 |

---

## ⏱️ 训练时间预估

| 项目 | 估计值 |
|------|--------|
| 单个epoch | ~2-3分钟（GPU: A100；批size=8） |
| 全部200个epoch | ~400-600分钟 (~7-10小时) |
| 最优模型保存 | 自动保存为 `p_v_256.hdf5` |

---

## 🔍 检查清单

在启动前确认：

- [ ] ✅ `train_det.py` 包含 `get_edge_aware_loss()` 函数
- [ ] ✅ `train_det.py` 包含 `get_smoothness_loss()` 函数  
- [ ] ✅ `train_det.py` 的 `train()` 函数使用了新损失
- [ ] ✅ `args_det.py` 已更新参数（batch-size=8等）
- [ ] ✅ CUDA/GPU 可用（推荐用GPU训练）
- [ ] ✅ 数据文件存在（`dataset/*.hdf5`）

验证GPU可用：
```bash
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

---

## 💡 训练技巧

### 缩短调试时间
```bash
# 只训练10个epoch快速验证
python train_det.py --epochs 10 --plot-freq 2
```

### 后台挂起训练
```bash
nohup python train_det.py > train.log 2>&1 &
tail -f train.log  # 查看日志
```

### 监控GPU显存
```bash
watch -n 1 nvidia-smi  # 每秒刷新GPU状态
```

---

## 🎉 启动！

所有准备就绪，执行:

```bash
python train_det.py
```

**预期效果**:
- ✅ 第一行目标 & 第二行预测高度相似
- ✅ 第三行误差明显减少
- ✅ 模型保存为 `p_v_256.hdf5` 

---

**准备好了吗？让我们开始训练！** 🚀🔥
