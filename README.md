# ESER2026_EPE_HUST (3D-only)

本仓库已整理为 3D-only 工作流，核心入口如下：

- 训练: `train_det.py`
- 预测: `surrogate_predict.py`

## 训练

示例命令:

```bash
python3 train_det.py \
	--exp-name p_v_pred_tesst \
	--train-file ./dataset/p_v_256.hdf5 \
	--test-file ./dataset/p_v_256.hdf5 \
	--epochs 50 \
	--batch-size 16 \
	--growth-rate 16
```

说明:

- 当前默认使用 `args_det_3d.py`。
- 训练输出会保存到 `./experiments/<exp-name>/...`。

## 预测

示例命令:

```bash
python3 surrogate_predict.py \
	--ckpt-file ./experiments/p_v_pred_tesst/kle4225/ntrain512_blocks[3, 6, 3]_growth16_nif48_drop0.0_batch16_lr0.003_wd0.0005_epochs50/checkpoints/model_epoch50.pth \
	--data-file ./dataset/p_v_256.hdf5 \
	--batch-size 64 \
	--out-file ./dataset/prediction/p_v_256.pred.hdf5
```

说明:

- 仅支持 3D 输入 `(N, C, D, H, W)`。
- 支持 `logits/class/reg` 三种输出模式（`--save-mode`）。
- 默认会对预测结果应用 T-mask（与训练一致）。

## 已清理内容

以下旧组件已移除:

- 2D 模型与 2D 数据/绘图工具
- 旧 Bayesian UQ/SVGD 代码
- 多份历史模型备份文件
- 旧 KLE 下载脚本
