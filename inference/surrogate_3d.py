import argparse
import json
import os

import h5py
import numpy as np
import torch

from models.dense_ed_3d import DenseED
from training.preprocess import crop_model_output, maybe_pad_model_input
from utils.load_data_3d import load_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D surrogate inference: load .pth checkpoint and predict on HDF5"
    )
    parser.add_argument("--ckpt-file", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--data-file", type=str, required=True, help="Path to HDF5 file")

    parser.add_argument("--nic", type=int, default=1, help="Input channels")
    parser.add_argument("--noc", type=int, default=3, help="Output channels")

    parser.add_argument("--blocks", type=str, default="[3, 6, 3]", help="Dense block layout as list")
    parser.add_argument("--growth-rate", type=int, default=16)
    parser.add_argument("--init-features", type=int, default=48)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--bn-size", type=int, default=8)
    parser.add_argument("--bottleneck", action="store_true", default=False)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for debug")
    parser.add_argument("--out-file", type=str, default=None, help="Output .hdf5 or .npz")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 or cpu")
    parser.add_argument(
        "--save-mode",
        type=str,
        default="auto",
        choices=["auto", "logits", "class", "reg"],
        help="How to save outputs",
    )
    parser.add_argument(
        "--bin-center-ratios",
        nargs="+",
        type=float,
        default=[0.05, 0.25, 0.55, 0.85],
        help="Relative class center ratios for reg reconstruction",
    )
    parser.add_argument(
        "--center-max-source",
        type=str,
        default="fixed",
        choices=["fixed", "target"],
        help="How to get center max value in reg mode",
    )
    parser.add_argument(
        "--center-max-value",
        type=float,
        default=1.0,
        help="Absolute max value when center-max-source=fixed",
    )
    parser.add_argument("--pad-multiple", type=int, default=16)

    args = parser.parse_args()
    try:
        args.blocks = json.loads(args.blocks.replace("'", '"'))
        if not isinstance(args.blocks, list):
            raise ValueError
    except Exception as error:
        raise ValueError("--blocks must be a JSON-like list, e.g. '[3, 6, 3]'") from error
    return args


def _validate_3d_input(data_file):
    with h5py.File(data_file, "r") as handle:
        x = handle["input"][()]
    if x.ndim != 5:
        raise ValueError(
            "3D-only pipeline expects input shape (N, C, D, H, W), got {}".format(x.shape)
        )


def _resolve_device(device_override):
    if device_override is not None:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_channels_from_checkpoint(state_dict, args):
    if "last_trans_up.conv2.weight" in state_dict:
        args.noc = state_dict["last_trans_up.conv2.weight"].shape[0]
    elif "features.LastTransUp.conv2.weight" in state_dict:
        args.noc = state_dict["features.LastTransUp.conv2.weight"].shape[0]

    if "in_conv.weight" in state_dict:
        args.nic = state_dict["in_conv.weight"].shape[1]
    elif "features.In_conv.weight" in state_dict:
        args.nic = state_dict["features.In_conv.weight"].shape[1]


def _build_model(args, device):
    return DenseED(
        in_channels=args.nic,
        out_channels=args.noc,
        blocks=args.blocks,
        growth_rate=args.growth_rate,
        init_features=args.init_features,
        drop_rate=args.drop_rate,
        bn_size=args.bn_size,
        bottleneck=args.bottleneck,
        out_activation=None,
        output_size=None,
    ).to(device)


def _forward_with_padding(model, x, pad_multiple):
    x_pad, h_slice, w_slice = maybe_pad_model_input(x, pad_multiple=pad_multiple)
    y_pad = model(x_pad)
    return crop_model_output(y_pad, h_slice, w_slice)


def _load_dataset(args):
    return load_data(
        args.data_file,
        args.batch_size,
        input_key=None,
        prefer_sensor_input=False,
        align_input_to_output=True,
        shuff=False,
    )


def _choose_save_mode(args, sample_pred, sample_target):
    model_channels = sample_pred.shape[1]
    target_channels = sample_target.shape[0] if sample_target.ndim >= 1 else 1
    if args.save_mode == "auto":
        if target_channels == 1 and model_channels > 1:
            return "reg"
        return "logits"
    return args.save_mode


def _allocate_preds(mode, n_total, sample_pred):
    if mode in ("class", "reg"):
        shape = (n_total, 1) + tuple(sample_pred.shape[2:])
        dtype = np.int64 if mode == "class" else np.float32
        return np.zeros(shape, dtype=dtype)
    shape = (n_total,) + tuple(sample_pred.shape[1:])
    return np.zeros(shape, dtype=np.float32)


def _assemble_regression(out_full, y_true, args):
    if out_full.shape[1] <= 1:
        return out_full

    out_class = out_full[:, :-1, ...]
    out_reg = out_full[:, -1:, ...]
    num_classes = out_class.shape[1]
    if len(args.bin_center_ratios) != num_classes:
        raise ValueError(
            "--bin-center-ratios length ({}) must equal class channels ({})".format(
                len(args.bin_center_ratios), num_classes
            )
        )

    if args.center_max_source == "target":
        y_true_np = y_true.detach().cpu().numpy()
        y_true_first = y_true_np[:, :1, ...]
        batch_max = np.maximum(y_true_first.reshape(y_true_first.shape[0], -1).max(axis=1), 1e-8)
    else:
        batch_max = np.full((out_reg.shape[0],), max(float(args.center_max_value), 1e-8), dtype=np.float32)

    ratios = np.array(args.bin_center_ratios, dtype=np.float32)
    centers = batch_max[:, None] * ratios[None, :]

    logits = out_class - np.max(out_class, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.maximum(np.sum(exp_logits, axis=1, keepdims=True), 1e-12)

    expand_shape = [centers.shape[0], centers.shape[1]] + [1] * (out_class.ndim - 2)
    centers_expanded = centers.reshape(expand_shape)
    center_soft = np.sum(probs * centers_expanded, axis=1, keepdims=True).astype(np.float32)
    return center_soft + out_reg


def _apply_t_mask(preds):
    if preds.ndim < 2:
        return preds
    h, w = preds.shape[-2], preds.shape[-1]
    mask = np.zeros((h, w), dtype=bool)
    mask[0:136, 0:35] = True
    mask[51:87, 35:137] = True
    preds[..., ~mask] = 0.0
    return preds


def _resolve_out_file(args):
    if args.out_file is not None:
        return args.out_file
    base = os.path.splitext(os.path.basename(args.data_file))[0]
    out_dir = os.path.dirname(args.data_file)
    return os.path.join(out_dir, base + ".pred.hdf5")


def _save_outputs(out_file, preds, args, save_mode):
    if out_file.lower().endswith((".hdf5", ".h5")):
        with h5py.File(out_file, "w") as handle:
            handle.create_dataset("predictions", data=preds, compression="gzip")
            handle.attrs["source_data"] = args.data_file
            handle.attrs["ckpt_file"] = args.ckpt_file
            handle.attrs["save_mode"] = save_mode
            handle.attrs["ndim"] = 3
            handle.attrs["nic"] = args.nic
            handle.attrs["noc"] = args.noc
            handle.attrs["blocks"] = json.dumps(args.blocks)
            handle.attrs["growth_rate"] = args.growth_rate
            handle.attrs["init_features"] = args.init_features
            handle.attrs["drop_rate"] = args.drop_rate
            handle.attrs["bn_size"] = args.bn_size
            handle.attrs["bottleneck"] = int(bool(args.bottleneck))
        return

    np.savez_compressed(
        out_file,
        predictions=preds,
        meta={
            "source_data": args.data_file,
            "ckpt_file": args.ckpt_file,
            "save_mode": save_mode,
            "ndim": 3,
            "nic": args.nic,
            "noc": args.noc,
            "blocks": args.blocks,
            "growth_rate": args.growth_rate,
            "init_features": args.init_features,
            "drop_rate": args.drop_rate,
            "bn_size": args.bn_size,
            "bottleneck": bool(args.bottleneck),
        },
    )


def run(args):
    _validate_3d_input(args.data_file)
    device = _resolve_device(args.device)

    state = torch.load(args.ckpt_file, map_location=device)
    _infer_channels_from_checkpoint(state, args)

    loader, _ = _load_dataset(args)
    n_total = len(loader.dataset)
    if args.limit is not None:
        n_total = min(n_total, args.limit)

    model = _build_model(args, device)
    model.load_state_dict(state)
    model.eval()

    sample_x, sample_y = loader.dataset[0]
    with torch.no_grad():
        sample_pred = _forward_with_padding(model, sample_x.unsqueeze(0).to(device), args.pad_multiple)
        sample_pred = sample_pred.detach().cpu().numpy()

    save_mode = _choose_save_mode(args, sample_pred, sample_y)
    preds = _allocate_preds(save_mode, n_total, sample_pred)

    idx_write = 0
    with torch.no_grad():
        for x, y_true in loader:
            if args.limit is not None and idx_write >= args.limit:
                break
            out_full = _forward_with_padding(model, x.to(device), args.pad_multiple).detach().cpu().numpy()
            if save_mode == "class":
                y_pred = np.argmax(out_full, axis=1, keepdims=True)
            elif save_mode == "reg":
                y_pred = _assemble_regression(out_full, y_true, args)
            else:
                y_pred = out_full

            batch_size = y_pred.shape[0]
            end = min(idx_write + batch_size, n_total)
            preds[idx_write:end] = y_pred[: end - idx_write]
            idx_write = end

    preds = _apply_t_mask(preds)
    out_file = _resolve_out_file(args)
    _save_outputs(out_file, preds, args, save_mode)

    print("Predictions saved to:", out_file)
    print("Pred shape:", preds.shape)


def main():
    args = parse_args()
    run(args)
