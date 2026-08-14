"""
Surrogate inference script: load a .pth checkpoint and predict on a
specified HDF5 file, saving the outputs.

Usage example (3D, my_run_3d checkpoint, predict on 10243d.hdf5):

	python surrogate_predict.py \
	  --ckpt-file ./experiments/my_run_3d/kle4225/ntrain512_blocks[3, 6, 3]_growth16_nif48_drop0.0_batch16_lr0.003_wd0.0005_epochs200/checkpoints/model_epoch100.pth \
	  --data-file ./dataset/10243d.hdf5 \
	  --ndim 3 \
	  --batch-size 128 \
	  --out-file ./experiments/my_run_3d/predictions_10243d_epoch100.pred.hdf5

This script does not train; it only performs forward prediction.
"""

import argparse
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import h5py
from utils.hybrid_bins import (
	DEFAULT_CLS_THRESHOLDS,
	bin_center_ratios_from_thresholds,
	validate_cls_thresholds,
)


def parse_args():
	p = argparse.ArgumentParser(description="Surrogate inference: load .pth and predict on HDF5")
	# Required inputs
	p.add_argument('--ckpt-file', type=str, required=True, help='Path to checkpoint .pth file')
	p.add_argument('--data-file', type=str, required=True, help='Path to HDF5 file with datasets: input/output')

	# Dimensionality and channels
	p.add_argument('--ndim', type=int, default=None, choices=[2, 3], help='Data dimensionality (auto if omitted)')
	p.add_argument('--nic', type=int, default=1, help='Number of input channels')
	p.add_argument('--noc', type=int, default=3, help='Number of output channels')

	# Model hyperparameters (defaults match repo experiments)
	p.add_argument('--blocks', type=str, default='[3, 6, 3]', help='Dense block layout as a Python list')
	p.add_argument('--growth-rate', type=int, default=16)
	p.add_argument('--init-features', type=int, default=48)
	p.add_argument('--drop-rate', type=float, default=0.0)
	p.add_argument('--bn-size', type=int, default=8)
	p.add_argument('--bottleneck', action='store_true', default=False)

	# Inference options
	p.add_argument('--batch-size', type=int, default=64)
	p.add_argument('--limit', type=int, default=None, help='Limit number of samples to predict (debug)')
	p.add_argument('--out-file', type=str, default=None, help='Output file (.hdf5 or .npz) to save predictions')
	p.add_argument('--device', type=str, default=None, help='Device override, e.g. cuda:0 or cpu')
	p.add_argument('--save-mode', type=str, default='auto', choices=['auto', 'logits', 'class', 'reg'],
			   help='How to save predictions: reg (continuous regression), logits (raw model output), class (argmax labels), or auto')
	p.add_argument('--cls-thresholds', nargs='+', type=float, default=None,
			   help='Training class thresholds. If omitted, read them from the checkpoint run args.txt')
	p.add_argument('--bin-center-ratios', nargs='+', type=float, default=None,
			   help='Explicit center-ratio override; by default centers are derived from --cls-thresholds exactly as in training')
	p.add_argument('--center-max-source', type=str, default='target', choices=['fixed', 'target'],
			   help='target exactly matches training; fixed supports inference without using the target maximum')
	p.add_argument('--center-max-value', type=float, default=1.0,
			   help='Absolute max value used when --center-max-source=fixed')
	p.add_argument('--target-channel', type=int, default=None,
			   help='Target channel used to compute target maximum; defaults to the training args.txt value')
	p.add_argument('--pad-multiple', type=int, default=16,
			   help='For 3D inference: pad H/W to this multiple before model forward, then crop back')

	args = p.parse_args()

	# Parse blocks string safely
	try:
		args.blocks = json.loads(args.blocks.replace("'", '"'))
		if not isinstance(args.blocks, list):
			raise ValueError
	except Exception:
		raise ValueError('--blocks must be a JSON-like list, e.g. "[3, 6, 3]"')

	training_args_path = os.path.join(
		os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt_file))),
		'args.txt',
	)
	training_args = {}
	if os.path.isfile(training_args_path):
		try:
			with open(training_args_path, 'r') as args_file:
				training_args = json.load(args_file)
		except (OSError, ValueError) as exc:
			print('Warning: failed to read training args from {}: {}'.format(training_args_path, exc))

	if args.cls_thresholds is None:
		args.cls_thresholds = training_args.get(
			'cls_thresholds', list(DEFAULT_CLS_THRESHOLDS)
		)
	args.cls_thresholds = list(validate_cls_thresholds(args.cls_thresholds))
	if args.bin_center_ratios is None:
		args.bin_center_ratios = list(
			bin_center_ratios_from_thresholds(args.cls_thresholds)
		)
	if args.target_channel is None:
		args.target_channel = int(training_args.get('target_channel', 0))

	return args


def _auto_detect_ndim(data_file):
	with h5py.File(data_file, 'r') as f:
		if 'spatial_input' in f:
			x_shape = f['spatial_input'].shape
		elif 'output' in f:
			x_shape = f['output'].shape
		else:
			x_shape = f['input'].shape
	# 2D: (N, C, H, W); 3D: (N, C, D, H, W)
	ndim = 3 if len(x_shape) == 5 else 2
	return ndim, x_shape


def _build_model(args, device):
	if args.ndim == 3:
		from models.dense_ed_3d import DenseED as DenseED
	else:
		from models.dense_ed import DenseED
	model_kwargs = {
		'in_channels': args.nic,
		'out_channels': args.noc,
		'blocks': args.blocks,
		'growth_rate': args.growth_rate,
		'init_features': args.init_features,
		'drop_rate': args.drop_rate,
		'bn_size': args.bn_size,
		'bottleneck': args.bottleneck,
		'out_activation': None,
	}
	if args.ndim == 3:
		model_kwargs['output_size'] = None
	model = DenseED(**model_kwargs).to(device)
	return model


def _pad_hw_to_multiple_torch(x, multiple=16, mode='replicate'):
	"""Pad tensor H/W to next multiple and return crop slices for restoration."""
	if x.dim() not in (4, 5):
		raise ValueError('Unsupported tensor dim {}'.format(x.dim()))
	h, w = x.shape[-2], x.shape[-1]
	pad_h_total = (multiple - h % multiple) % multiple
	pad_w_total = (multiple - w % multiple) % multiple
	pad_top = pad_h_total // 2
	pad_bottom = pad_h_total - pad_top
	pad_left = pad_w_total // 2
	pad_right = pad_w_total - pad_left
	if x.dim() == 5:
		pad_dims = (pad_left, pad_right, pad_top, pad_bottom, 0, 0)
	else:
		pad_dims = (pad_left, pad_right, pad_top, pad_bottom)
	x_pad = F.pad(x, pad_dims, mode=mode)
	h_slice = slice(pad_top, pad_top + h)
	w_slice = slice(pad_left, pad_left + w)
	return x_pad, h_slice, w_slice


def _model_forward_with_optional_pad(model, x, ndim, pad_multiple=16):
	"""Keep 2D path unchanged; 3D path uses pad->forward->crop for alignment."""
	if ndim != 3:
		return model(x)
	x_pad, h_slice, w_slice = _pad_hw_to_multiple_torch(x, multiple=pad_multiple, mode='replicate')
	y_pad = model(x_pad)
	return y_pad[..., h_slice, w_slice]


def _load_data(args):
	# Use repo loaders to keep preprocessing consistent
	hybrid_kwargs = {
		'prepare_hybrid_targets': True,
		'cls_thresholds': args.cls_thresholds,
		'num_classes': len(args.bin_center_ratios),
		'target_channel': args.target_channel,
		'center_max_source': args.center_max_source,
		'center_max_value': args.center_max_value,
		'bin_center_ratios': args.bin_center_ratios,
	}
	if args.ndim == 3:
		from utils.load_data_3d import load_data
		loader, stats = load_data(
			args.data_file,
			args.batch_size,
			input_key=None,
			prefer_sensor_input=False,
			align_input_to_output=True,
			shuff=False,
			**hybrid_kwargs
		)
	else:
		from utils.load_data import load_data
		loader, stats = load_data(
			args.data_file,
			args.batch_size,
			shuff=False,
			**hybrid_kwargs
		)
	return loader, stats


def _validate_preprocessed_batch(batch):
	required = {'input', 'target_continuous', 'bin_centers', 'mask'}
	if not isinstance(batch, dict):
		raise TypeError('Expected a dict batch from load_data, got {}'.format(type(batch)))
	missing = required.difference(batch)
	if missing:
		raise KeyError('Batch is missing fields: {}'.format(sorted(missing)))
	return (
		batch['input'],
		batch['target_continuous'],
		batch['bin_centers'],
		batch['mask'],
	)


def main():
	args = parse_args()

	# Auto-detect dimensionality from file if not provided
	if args.ndim is None:
		args.ndim, in_shape = _auto_detect_ndim(args.data_file)
	else:
		_, in_shape = _auto_detect_ndim(args.data_file)

	# Device
	if args.device is not None:
		device = torch.device(args.device)
	else:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load checkpoint state early to align channels
	state = torch.load(args.ckpt_file, map_location=device)
	# 2D vs 3D checkpoints use different module names
	if 'last_trans_up.conv2.weight' in state:
		args.noc = state['last_trans_up.conv2.weight'].shape[0]
	elif 'features.LastTransUp.convT2.weight' in state:
		args.noc = state['features.LastTransUp.convT2.weight'].shape[1]
	elif 'features.LastTransUp.conv2.weight' in state:
		args.noc = state['features.LastTransUp.conv2.weight'].shape[0]
	if 'in_conv.weight' in state:
		args.nic = state['in_conv.weight'].shape[1]
	elif 'features.In_conv.weight' in state:
		args.nic = state['features.In_conv.weight'].shape[1]

	# Load data
	loader, stats = _load_data(args)
	n_total = len(loader.dataset)
	if args.limit is not None:
		n_total = min(n_total, args.limit)

	# Build and load model
	model = _build_model(args, device)
	model.load_state_dict(state)
	model.eval()

	# Prepare output array preallocation using model output shape
	# (robust to discrete models where out_channels may differ from dataset output channels)
	sample_x, sample_y, _, sample_mask = _validate_preprocessed_batch(loader.dataset[0])
	with torch.no_grad():
		sample_pred = _model_forward_with_optional_pad(
			model,
			sample_x.unsqueeze(0).to(device),
			args.ndim,
			pad_multiple=args.pad_multiple,
		).detach().cpu().numpy()
	model_c = sample_pred.shape[1]
	target_c = sample_y.shape[0] if sample_y.ndim >= 1 else 1
	if args.save_mode == 'auto':
		if target_c == 1 and model_c > 1:
			save_mode = 'reg'
		else:
			save_mode = 'logits'
	else:
		save_mode = args.save_mode

	if save_mode == 'class':
		pred_shape = (n_total, 1) + tuple(sample_pred.shape[2:])
		preds = np.zeros(pred_shape, dtype=np.int64)
	elif save_mode == 'reg':
		pred_shape = (n_total, 1) + tuple(sample_pred.shape[2:])
		preds = np.zeros(pred_shape, dtype=np.float32)
	else:
		pred_shape = (n_total,) + tuple(sample_pred.shape[1:])
		preds = np.zeros(pred_shape, dtype=np.float32)
	masks = np.zeros((n_total, 1) + tuple(sample_pred.shape[2:]), dtype=np.uint8)

	# Iterate and predict
	idx_write = 0
	with torch.no_grad():
		for batch_idx, batch in enumerate(loader):
			if args.limit is not None and idx_write >= args.limit:
				break
			x, _, bin_centers_tensor, domain_mask = _validate_preprocessed_batch(batch)
			x = x.to(device)
			# 拿到包含 5 个通道的完整输出 (B, 5, D, H, W)
			y_pred_full = _model_forward_with_optional_pad(
				model,
				x,
				args.ndim,
				pad_multiple=args.pad_multiple,
			).detach().cpu().numpy()

			if save_mode == 'class':
				y_pred = np.argmax(y_pred_full, axis=1, keepdims=True)
			elif save_mode == 'reg':
				if y_pred_full.shape[1] > 1:
					out_class = y_pred_full[:, :-1, ...]  # 前四个通道是分类 logits
					out_reg = y_pred_full[:, -1:, ...]    # 最后一个是残差回归通道
					num_classes = out_class.shape[1]
					bin_centers = bin_centers_tensor.detach().cpu().numpy()
					if bin_centers.shape[1] != num_classes:
						raise ValueError(
							'Preprocessed bin center count ({}) must equal number of class channels ({})'.format(
								bin_centers.shape[1], num_classes
							)
						)

					# Soft-argmax style assembly: center = sum(p_k * c_k)
					# stable softmax over class channel
					logits = out_class - np.max(out_class, axis=1, keepdims=True)
					exp_logits = np.exp(logits)
					probs = exp_logits / np.maximum(np.sum(exp_logits, axis=1, keepdims=True), 1e-12)

					# bin_centers: (B, C) -> (B, C, 1, 1[, 1]) for broadcasting
					expand_shape = [bin_centers.shape[0], bin_centers.shape[1]] + [1] * (out_class.ndim - 2)
					bin_centers_expanded = bin_centers.reshape(expand_shape)

					pred_center_soft = np.sum(probs * bin_centers_expanded, axis=1, keepdims=True).astype(np.float32)
					y_pred = pred_center_soft + out_reg
				else:
					y_pred = y_pred_full
			else:
				y_pred = y_pred_full

			if domain_mask is None:
				mask_np = np.ones(
					(y_pred.shape[0], 1) + tuple(y_pred.shape[2:]), dtype=np.uint8
				)
			else:
				mask_np = domain_mask.detach().cpu().numpy().astype(np.uint8)
			if mask_np.shape[1] != 1 or tuple(mask_np.shape[2:]) != tuple(y_pred.shape[2:]):
				raise ValueError(
					'Mask shape {} does not match prediction shape {}'.format(
						mask_np.shape, y_pred.shape
					)
				)
			y_pred = np.where(mask_np.astype(bool), y_pred, 0)
			bs = y_pred.shape[0]
			end = min(idx_write + bs, n_total)
			preds[idx_write:end] = y_pred[:(end - idx_write)]
			masks[idx_write:end] = mask_np[:(end - idx_write)]
			idx_write = end

	# Determine output file path
	if args.out_file is None:
		base = os.path.splitext(os.path.basename(args.data_file))[0]
		out_dir = os.path.dirname(args.data_file)
		args.out_file = os.path.join(out_dir, base + '.pred.hdf5')

	# Save predictions
	if args.out_file.lower().endswith('.hdf5') or args.out_file.lower().endswith('.h5'):
		with h5py.File(args.out_file, 'w') as f:
			f.create_dataset('predictions', data=preds, compression='gzip')
			f.create_dataset('mask', data=masks, compression='gzip')
			f.attrs['source_data'] = args.data_file
			f.attrs['ckpt_file'] = args.ckpt_file
			f.attrs['save_mode'] = save_mode
			f.attrs['ndim'] = args.ndim
			f.attrs['nic'] = args.nic
			f.attrs['noc'] = args.noc
			f.attrs['blocks'] = json.dumps(args.blocks)
			f.attrs['growth_rate'] = args.growth_rate
			f.attrs['init_features'] = args.init_features
			f.attrs['drop_rate'] = args.drop_rate
			f.attrs['bn_size'] = args.bn_size
			f.attrs['bottleneck'] = int(bool(args.bottleneck))
			f.attrs['cls_thresholds'] = json.dumps(args.cls_thresholds)
			f.attrs['bin_center_ratios'] = json.dumps(args.bin_center_ratios)
			f.attrs['center_max_source'] = args.center_max_source
			f.attrs['target_channel'] = args.target_channel
			f.attrs['concentration_normalization'] = stats.get(
				'concentration_normalization', 'none'
			)
	else:
		# Fallback: npz
		np.savez_compressed(args.out_file, predictions=preds, mask=masks, meta={
			'source_data': args.data_file,
			'ckpt_file': args.ckpt_file,
			'save_mode': save_mode,
			'ndim': args.ndim,
			'nic': args.nic,
			'noc': args.noc,
			'blocks': args.blocks,
			'growth_rate': args.growth_rate,
			'init_features': args.init_features,
			'drop_rate': args.drop_rate,
			'bn_size': args.bn_size,
			'bottleneck': bool(args.bottleneck),
			'cls_thresholds': args.cls_thresholds,
			'bin_center_ratios': args.bin_center_ratios,
			'center_max_source': args.center_max_source,
			'target_channel': args.target_channel,
			'concentration_normalization': stats.get('concentration_normalization', 'none'),
		})

	print('Predictions saved to:', args.out_file)
	print('Pred shape:', preds.shape)


if __name__ == '__main__':
	main()
