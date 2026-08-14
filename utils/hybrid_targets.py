import torch
from torch.utils.data import Dataset

from utils.hybrid_bins import (
    DEFAULT_CLS_THRESHOLDS,
    bin_center_ratios_from_thresholds,
    validate_cls_thresholds,
)


HYBRID_TARGET_KEYS = (
    "target_continuous",
    "target_class",
    "target_center",
    "target_residual",
    "bin_centers",
    "mask",
)


class DictTensorDataset(Dataset):
    """Dataset backed by equal-length tensors and yielding named fields."""

    def __init__(self, **tensors):
        if not tensors:
            raise ValueError("At least one tensor is required")
        lengths = {name: len(value) for name, value in tensors.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError("Tensor lengths do not match: {}".format(lengths))
        self.tensors = tensors

    def __len__(self):
        return len(next(iter(self.tensors.values())))

    def __getitem__(self, index):
        return {name: value[index] for name, value in self.tensors.items()}


def _select_target_channel(target, target_channel):
    if target.dim() < 3:
        raise ValueError(
            "Target must have shape (N, C, spatial...), got {}".format(
                tuple(target.shape)
            )
        )
    if target.dim() == 3:
        target = target.unsqueeze(1)
    channel = int(target_channel)
    if channel < 0 or channel >= target.shape[1]:
        raise ValueError(
            "target_channel {} is invalid for target shape {}".format(
                channel, tuple(target.shape)
            )
        )
    return target[:, channel:channel + 1, ...]


def _align_mask(mask, target):
    """Align shared or per-sample masks to ``(N, 1, spatial...)``."""
    if mask is None:
        return torch.ones_like(target, dtype=torch.bool)
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.bool()

    if mask.dim() == target.dim() - 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == target.dim() - 1:
        mask = mask.unsqueeze(1)
    elif mask.dim() != target.dim():
        raise ValueError(
            "Mask shape {} cannot be aligned with target shape {}".format(
                tuple(mask.shape), tuple(target.shape)
            )
        )

    if mask.shape[1] != 1 or tuple(mask.shape[2:]) != tuple(target.shape[2:]):
        raise ValueError(
            "Mask shape {} does not match target shape {}".format(
                tuple(mask.shape), tuple(target.shape)
            )
        )
    if mask.shape[0] == 1 and target.shape[0] > 1:
        mask = mask.expand(target.shape[0], *mask.shape[1:])
    elif mask.shape[0] != target.shape[0]:
        raise ValueError(
            "Mask batch dimension {} does not match target batch dimension {}".format(
                mask.shape[0], target.shape[0]
            )
        )
    return mask


def preprocess_hybrid_targets(
    target,
    mask=None,
    cls_thresholds=DEFAULT_CLS_THRESHOLDS,
    num_classes=None,
    target_channel=0,
    ignore_index=-100,
    center_max_source="target",
    center_max_value=1.0,
    bin_center_ratios=None,
):
    """Prepare all hybrid classification/regression targets in memory.

    The returned tensors are ready for the training or prediction loop. Bin
    centers retain the existing per-sample target-maximum behavior by default.
    """
    if not torch.is_tensor(target):
        target = torch.as_tensor(target, dtype=torch.float32)
    elif not target.is_floating_point():
        target = target.float()
    if not torch.isfinite(target).all():
        raise ValueError("Target contains non-finite values")

    thresholds = validate_cls_thresholds(cls_thresholds)
    inferred_classes = len(thresholds) + 1
    if num_classes is None:
        num_classes = inferred_classes
    if int(num_classes) != inferred_classes:
        raise ValueError(
            "num_classes {} requires {} thresholds, got {}".format(
                num_classes, int(num_classes) - 1, len(thresholds)
            )
        )

    target_continuous = _select_target_channel(target, target_channel)
    aligned_mask = _align_mask(mask, target_continuous)
    batch_size = target_continuous.shape[0]

    target_max = target_continuous.reshape(batch_size, -1).max(dim=1).values
    target_max = target_max.clamp(min=1e-8)
    if center_max_source == "target":
        center_max = target_max
    elif center_max_source == "fixed":
        fixed_max = max(float(center_max_value), 1e-8)
        center_max = target_continuous.new_full((batch_size,), fixed_max)
    else:
        raise ValueError(
            "center_max_source must be 'target' or 'fixed', got {!r}".format(
                center_max_source
            )
        )

    if bin_center_ratios is None:
        bin_center_ratios = bin_center_ratios_from_thresholds(thresholds)
    center_ratios = target_continuous.new_tensor(tuple(bin_center_ratios))
    if center_ratios.numel() != int(num_classes):
        raise ValueError(
            "bin_center_ratios length {} must equal num_classes {}".format(
                center_ratios.numel(), num_classes
            )
        )
    if not torch.isfinite(center_ratios).all():
        raise ValueError("bin_center_ratios contains non-finite values")
    bin_centers = center_max.unsqueeze(1) * center_ratios.unsqueeze(0)

    target_class = torch.zeros_like(target_continuous, dtype=torch.long)
    for class_index, threshold in enumerate(thresholds, start=1):
        threshold_values = target_max * float(threshold)
        view_shape = [batch_size, 1] + [1] * (target_continuous.dim() - 2)
        target_class[
            target_continuous >= threshold_values.view(*view_shape)
        ] = class_index
    target_class = target_class.squeeze(1)

    flat_class = target_class.reshape(batch_size, -1)
    flat_centers = torch.gather(bin_centers, 1, flat_class)
    target_center = flat_centers.reshape(target_class.shape).unsqueeze(1)
    target_residual = target_continuous - target_center

    target_class = target_class.clone()
    target_class[~aligned_mask.squeeze(1)] = int(ignore_index)

    return {
        "target_continuous": target_continuous,
        "target_class": target_class,
        "target_center": target_center,
        "target_residual": target_residual,
        "bin_centers": bin_centers,
        "mask": aligned_mask,
    }


def build_hybrid_dataset(input_tensor, target_tensor, mask=None, **preprocess_kwargs):
    targets = preprocess_hybrid_targets(
        target_tensor, mask=mask, **preprocess_kwargs
    )
    return DictTensorDataset(input=input_tensor, **targets)
