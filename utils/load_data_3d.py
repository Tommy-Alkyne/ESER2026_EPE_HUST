import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils.hybrid_targets import build_hybrid_dataset


def _reshape_to_5d(arr):
    if arr.ndim == 3:
        return arr[:, None, None, :, :]
    if arr.ndim == 4:
        return arr[:, :, None, :, :]
    if arr.ndim == 5:
        return arr
    raise ValueError("Unsupported input ndim {}".format(arr.ndim))


def _vector_to_5d(arr, target_shape):
    n_samples = arr.shape[0]
    vector = arr.reshape(n_samples, -1)[:, :, None, None, None]
    d, h, w = target_shape
    return np.tile(vector, (1, 1, d, h, w))


def _nonbatch_feature_dim(arr):
    return int(np.prod(arr.shape[1:]))


def _attr_text(value, default="none"):
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_concentration(datasets, attrs):
    """Apply only the normalization explicitly declared by the HDF5 file."""
    mode = _attr_text(attrs.get("concentration_normalization"))
    if mode in ("none", "", "already_normalized"):
        return datasets, mode
    if mode != "divide_by_particle_flow":
        raise ValueError("Unsupported concentration normalization: {}".format(mode))
    if "input" not in datasets:
        raise KeyError("particle-flow normalization requires the 'input' dataset")

    input_data = np.asarray(datasets["input"], dtype=np.float32).copy()
    flow_column = int(attrs.get("particle_flow_column", 3))
    if input_data.ndim != 2 or not 0 <= flow_column < input_data.shape[1]:
        raise ValueError(
            "particle_flow_column {} is invalid for input shape {}".format(
                flow_column, input_data.shape
            )
        )
    particle_flow = input_data[:, flow_column]
    if not np.all(np.isfinite(particle_flow)) or np.any(np.abs(particle_flow) < 1e-12):
        raise ValueError("particle_flow contains non-finite or zero values")

    output = np.asarray(datasets["output"], dtype=np.float32)
    output_scale = particle_flow.reshape((len(particle_flow),) + (1,) * (output.ndim - 1))
    datasets["output"] = output / output_scale

    sensor_start = int(attrs.get("input_concentration_start", input_data.shape[1]))
    sensor_count = int(attrs.get("input_concentration_count", 0))
    sensor_end = sensor_start + sensor_count
    if sensor_count:
        if sensor_start < 0 or sensor_end > input_data.shape[1]:
            raise ValueError(
                "input concentration slice [{}:{}] is invalid for shape {}".format(
                    sensor_start, sensor_end, input_data.shape
                )
            )
        input_data[:, sensor_start:sensor_end] /= particle_flow[:, None]
    datasets["input"] = input_data

    if "sensor_input" in datasets:
        sensor_data = np.asarray(datasets["sensor_input"], dtype=np.float32)
        sensor_scale = particle_flow.reshape(
            (len(particle_flow),) + (1,) * (sensor_data.ndim - 1)
        )
        datasets["sensor_input"] = sensor_data / sensor_scale
    return datasets, mode


def _prepare_mask(mask_data, n_samples, target_shape):
    """Return a bool mask shaped (N, 1, D, H, W).

    Accepted layouts are shared (H,W), (D,H,W), (1,D,H,W) and
    per-sample (N,H,W), (N,D,H,W), (N,1,D,H,W). Geometry is unrestricted,
    but its spatial shape must match the output field.
    """
    if mask_data is None:
        return np.ones((n_samples, 1) + tuple(target_shape), dtype=bool)

    mask = np.asarray(mask_data)
    target_shape = tuple(target_shape)
    if mask.ndim == 2 and target_shape[0] == 1 and tuple(mask.shape) == target_shape[1:]:
        mask = mask[None, None, None, :, :]
    elif mask.ndim == 3 and tuple(mask.shape) == target_shape:
        mask = mask[None, None, :, :, :]
    elif (
        mask.ndim == 3
        and target_shape[0] == 1
        and mask.shape[0] in (1, n_samples)
        and tuple(mask.shape[1:]) == target_shape[1:]
    ):
        mask = mask[:, None, None, :, :]
    elif mask.ndim == 4 and tuple(mask.shape[1:]) == target_shape:
        mask = mask[:, None, :, :, :]
    elif (
        mask.ndim == 4
        and target_shape[0] == 1
        and mask.shape[1] == 1
        and tuple(mask.shape[2:]) == target_shape[1:]
    ):
        mask = mask[:, :, None, :, :]
    elif (
        mask.ndim == 5
        and mask.shape[1] == 1
        and tuple(mask.shape[2:]) == target_shape
    ):
        pass
    else:
        raise ValueError(
            "Mask shape {} cannot be aligned with output spatial shape {}".format(
                mask.shape, target_shape
            )
        )

    if mask.shape[0] == 1 and n_samples > 1:
        mask = np.broadcast_to(mask, (n_samples,) + mask.shape[1:])
    elif mask.shape[0] != n_samples:
        raise ValueError(
            "Mask batch dimension {} does not match sample count {}".format(
                mask.shape[0], n_samples
            )
        )
    return mask.astype(bool, copy=False)


def load_data(data_dir, batch_size, input_key=None, prefer_sensor_input=False,
              align_input_to_output=True, shuff=True, combine_inputs=True,
              test_batch_size=None, split_ratio=None, split_seed=None,
              prepare_hybrid_targets=False, cls_thresholds=None,
              num_classes=None, target_channel=0, ignore_index=-100,
              center_max_source="target", center_max_value=1.0,
              bin_center_ratios=None):
    """Load 3D fields, applying file-declared normalization and domain masks.

    Raw HDF5 arrays are first copied into memory. File-declared normalization,
    mask alignment and optional hybrid-target preparation then happen before a
    DataLoader is created.

    With ``prepare_hybrid_targets=False`` the dataset yields the legacy
    ``(input, output, mask)`` tuple. With it enabled, each sample is a dict whose
    target fields are ready for the loss calculation.
    """
    with h5py.File(data_dir, "r") as h5_file:
        dataset_names = ("input", "output", "sensor_input", "spatial_input")
        datasets = {
            name: h5_file[name][()]
            for name in dataset_names
            if name in h5_file
        }
        if "output" not in datasets:
            raise KeyError("No 'output' dataset found in HDF5 file")
        mask_data = h5_file["mask"][()] if "mask" in h5_file else None
        attrs = dict(h5_file.attrs)

    datasets, normalization_mode = _normalize_concentration(datasets, attrs)
    y_data = datasets["output"]
    if y_data.ndim == 4:
        target_shape = (1, y_data.shape[-2], y_data.shape[-1])
    elif y_data.ndim == 5:
        target_shape = tuple(y_data.shape[-3:])
    else:
        raise ValueError("Unsupported output ndim {}".format(y_data.ndim))

    if input_key is not None:
        if input_key not in datasets:
            raise KeyError("Input dataset '{}' not found".format(input_key))
        x_data = datasets[input_key]
    elif combine_inputs and "spatial_input" in datasets:
        spatial_data = _reshape_to_5d(datasets["spatial_input"])
        spatial_shape = tuple(spatial_data.shape[-3:])
        parts = [spatial_data]
        if "input" in datasets:
            parts.append(_vector_to_5d(datasets["input"], spatial_shape))
        if "sensor_input" in datasets:
            parts.append(_vector_to_5d(datasets["sensor_input"], spatial_shape))
        x_data = np.concatenate(parts, axis=1)
    elif combine_inputs and ("input" in datasets or "sensor_input" in datasets):
        parts = []
        if "input" in datasets:
            input_arr = datasets["input"]
            if input_arr.ndim in (2, 3) and _nonbatch_feature_dim(input_arr) > 1:
                parts.append(_vector_to_5d(input_arr, target_shape))
            else:
                parts.append(_reshape_to_5d(input_arr))
        if "sensor_input" in datasets:
            parts.append(_vector_to_5d(datasets["sensor_input"], target_shape))
        x_data = np.concatenate(parts, axis=1)
    elif "input" in datasets and (not prefer_sensor_input or "sensor_input" not in datasets):
        x_data = datasets["input"]
    elif "sensor_input" in datasets:
        x_data = datasets["sensor_input"]
    else:
        raise KeyError("No valid input dataset found in HDF5 file")

    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(y_data.shape))
    print("concentration normalization: {}".format(normalization_mode))

    if x_data.ndim == 2:
        n_samples, n_features = x_data.shape
        side = int(np.sqrt(n_features))
        if side * side != n_features:
            raise ValueError(
                "2D input cannot be reshaped into a square map. "
                "Provide 'spatial_input' or input_key with spatial data."
            )
        x_data = x_data.reshape(n_samples, 1, 1, side, side)
    elif x_data.ndim in (3, 4, 5):
        x_data = _reshape_to_5d(x_data)
    else:
        raise ValueError("Unsupported input ndim {}".format(x_data.ndim))

    if y_data.ndim == 4:
        y_data = y_data[:, :, None, :, :]
    elif y_data.ndim != 5:
        raise ValueError("Unsupported output ndim {}".format(y_data.ndim))

    mask_data = _prepare_mask(mask_data, y_data.shape[0], y_data.shape[-3:])
    print("mask data shape: {}; valid cells/sample: {}".format(
        mask_data.shape, mask_data[0].sum()
    ))

    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)
    mask_tensor = torch.tensor(mask_data, dtype=torch.bool)

    if align_input_to_output:
        output_shape = tuple(y_tensor.shape[-3:])
        input_shape = tuple(x_tensor.shape[-3:])
        if input_shape != output_shape:
            x_tensor = F.interpolate(
                x_tensor,
                size=output_shape,
                mode="trilinear",
                align_corners=False,
            )

    kwargs = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
    if prepare_hybrid_targets:
        preprocess_kwargs = {
            "num_classes": num_classes,
            "target_channel": target_channel,
            "ignore_index": ignore_index,
            "center_max_source": center_max_source,
            "center_max_value": center_max_value,
            "bin_center_ratios": bin_center_ratios,
        }
        if cls_thresholds is not None:
            preprocess_kwargs["cls_thresholds"] = cls_thresholds
        dataset = build_hybrid_dataset(
            x_tensor,
            y_tensor,
            mask=mask_tensor,
            **preprocess_kwargs
        )
    else:
        dataset = TensorDataset(x_tensor, y_tensor, mask_tensor)

    y_data_mean = np.mean(y_data, axis=0)
    stats = {
        "y_mean": y_data_mean,
        "y_var": np.sum((y_data - y_data_mean) ** 2),
        "concentration_normalization": normalization_mode,
        "mask_shape": tuple(mask_tensor.shape[2:]),
        "hybrid_targets_prepared": bool(prepare_hybrid_targets),
    }

    if split_ratio is not None:
        if not 0.0 < split_ratio < 1.0:
            raise ValueError("split_ratio must be in (0, 1), got {}".format(split_ratio))
        n_total = len(dataset)
        n_train = int(n_total * split_ratio)
        n_test = n_total - n_train
        if n_train < 1 or n_test < 1:
            raise ValueError(
                "split_ratio {} leaves an empty train/test set ({} samples)".format(
                    split_ratio, n_total
                )
            )
        if test_batch_size is None:
            test_batch_size = batch_size
        generator = (
            torch.Generator().manual_seed(split_seed)
            if split_seed is not None
            else None
        )
        train_set, test_set = random_split(
            dataset, [n_train, n_test], generator=generator
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=shuff, **kwargs
        )
        test_loader = DataLoader(
            test_set, batch_size=test_batch_size, shuffle=False, **kwargs
        )
        print("split into train({}) / test({}) samples at ratio {}".format(
            n_train, n_test, split_ratio
        ))
        return train_loader, test_loader, stats

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuff, **kwargs)
    return data_loader, stats


if __name__ == "__main__":
    data_path = "/home/tommy/code/ESER2026_EPE_HUST/dataset/10243d.hdf5"
    loader, data_stats = load_data(data_path, 16)
    print("y_mean shape: {}".format(data_stats["y_mean"].shape))
    print("y_var: {}".format(data_stats["y_var"]))
