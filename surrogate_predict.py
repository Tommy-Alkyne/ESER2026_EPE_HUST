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
import numpy as np
import h5py


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

	args = p.parse_args()

	# Parse blocks string safely
	try:
		args.blocks = json.loads(args.blocks.replace("'", '"'))
		if not isinstance(args.blocks, list):
			raise ValueError
	except Exception:
		raise ValueError('--blocks must be a JSON-like list, e.g. "[3, 6, 3]"')

	return args


def _auto_detect_ndim(data_file):
	with h5py.File(data_file, 'r') as f:
		x = f['input'][()]
	# 2D: (N, C, H, W); 3D: (N, C, D, H, W)
	ndim = 3 if x.ndim == 5 else 2
	return ndim, x.shape


def _build_model(args, device):
	if args.ndim == 3:
		from models.dense_ed_3d import DenseED as DenseED
	else:
		from models.dense_ed import DenseED
	model = DenseED(
		in_channels=args.nic,
		out_channels=args.noc,
		blocks=args.blocks,
		growth_rate=args.growth_rate,
		init_features=args.init_features,
		drop_rate=args.drop_rate,
		bn_size=args.bn_size,
		bottleneck=args.bottleneck,
		out_activation=None,
		output_size=getattr(args, 'output_size', None) if args.ndim == 3 else None,
	).to(device)
	return model


def _load_data(args):
	# Use repo loaders to keep preprocessing consistent
	if args.ndim == 3:
		from utils.load_data_3d import load_data
	else:
		from utils.load_data import load_data
	loader, stats = load_data(
		args.data_file,
		args.batch_size,
		input_key=None,
		prefer_sensor_input=False,
		align_input_to_output=True,
		shuff=False
	)
	return loader, stats


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

	# Align output size for 3D to match targets
	if args.ndim == 3:
		_, sample_y = loader.dataset[0]
		args.output_size = tuple(sample_y.shape[-3:])

	# Build and load model
	model = _build_model(args, device)
	model.load_state_dict(state)
	model.eval()

	# Prepare output array preallocation using model output shape
	# (robust to discrete models where out_channels may differ from dataset output channels)
	sample_x, sample_y = loader.dataset[0]
	with torch.no_grad():
		sample_pred = model(sample_x.unsqueeze(0).to(device)).detach().cpu().numpy()
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

	# Iterate and predict
	idx_write = 0
	with torch.no_grad():
		for batch_idx, (x, _) in enumerate(loader):
			if args.limit is not None and idx_write >= args.limit:
				break
			x = x.to(device)
			y_pred = model(x).detach().cpu().numpy()
			if save_mode == 'class':
				y_pred = np.argmax(y_pred, axis=1, keepdims=True)
			elif save_mode == 'reg':
				if y_pred.shape[1] > 1:
					y_pred = y_pred[:, -1:, ...]
				else:
					y_pred = y_pred[:, :1, ...]
				y_pred = np.maximum(y_pred, 0.0)
			bs = y_pred.shape[0]
			end = min(idx_write + bs, n_total)
			preds[idx_write:end] = y_pred[:(end - idx_write)]
			idx_write = end

	# Apply T-mask: zero-out regions outside the two rectangles
	if preds.ndim in (4, 5):
		h, w = preds.shape[-2:]
		mask = np.zeros((h, w), dtype=bool)
		mask[0:136, 0:35] = True
		mask[51:87, 35:137] = True
		if preds.ndim == 4:
			preds[..., ~mask] = 0.0
		else:
			preds[..., :, ~mask] = 0.0

	# Determine output file path
	if args.out_file is None:
		base = os.path.splitext(os.path.basename(args.data_file))[0]
		out_dir = os.path.dirname(args.data_file)
		args.out_file = os.path.join(out_dir, base + '.pred.hdf5')

	# Save predictions
	if args.out_file.lower().endswith('.hdf5') or args.out_file.lower().endswith('.h5'):
		with h5py.File(args.out_file, 'w') as f:
			f.create_dataset('predictions', data=preds, compression='gzip')
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
	else:
		# Fallback: npz
		np.savez_compressed(args.out_file, predictions=preds, meta={
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
		})

	print('Predictions saved to:', args.out_file)
	print('Pred shape:', preds.shape)


if __name__ == '__main__':
	main()

