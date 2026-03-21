import torch
import torch.nn.functional as F


def build_t_mask(output):
    h, w = output.shape[-2:]
    mask_2d = torch.zeros((h, w), device=output.device, dtype=output.dtype)
    y1 = min(136, h)
    x1 = min(35, w)
    if y1 > 0 and x1 > 0:
        mask_2d[0:y1, 0:x1] = 1.0

    y0, y1b = min(51, h), min(87, h)
    x0, x1b = min(35, w), min(137, w)
    if y1b > y0 and x1b > x0:
        mask_2d[y0:y1b, x0:x1b] = 1.0

    if output.dim() == 5:
        d = output.shape[-3]
        return mask_2d.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(1, 1, d, 1, 1)

    raise ValueError('Unsupported output dim {}'.format(output.dim()))


def pad_hw_to_multiple(x, multiple=16, mode='replicate'):
    """Pad H/W to the next multiple and return crop slices for restoration."""
    if x.dim() != 5:
        raise ValueError('Unsupported input dim {}'.format(x.dim()))

    h, w = x.shape[-2], x.shape[-1]
    pad_h_total = (multiple - h % multiple) % multiple
    pad_w_total = (multiple - w % multiple) % multiple

    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    pad_dims = (pad_left, pad_right, pad_top, pad_bottom, 0, 0)
    x_padded = F.pad(x, pad_dims, mode=mode)

    h_slice = slice(pad_top, pad_top + h)
    w_slice = slice(pad_left, pad_left + w)
    return x_padded, h_slice, w_slice


def maybe_pad_model_input(x, pad_multiple=16):
    return pad_hw_to_multiple(x, multiple=pad_multiple, mode='replicate')


def crop_model_output(output, h_slice, w_slice):
    return output[..., h_slice, w_slice]


def continuous_to_discrete(target_continuous, thresholds):
    target_discrete = torch.zeros_like(target_continuous, dtype=torch.long)
    batch_size = target_continuous.shape[0]

    for i in range(batch_size):
        max_val = target_continuous[i].max().clamp(min=1e-8)
        for class_idx, threshold in enumerate(thresholds, start=1):
            target_discrete[i, target_continuous[i] >= threshold * max_val] = class_idx

    if target_discrete.dim() >= 2 and target_discrete.shape[1] == 1:
        target_discrete = target_discrete.squeeze(1)

    return target_discrete


def compute_bin_centers(target_continuous, thresholds):
    batch_size = target_continuous.shape[0]
    max_vals = target_continuous.reshape(batch_size, -1).max(dim=1).values.clamp(min=1e-8)
    threshold_ratios = torch.tensor(thresholds, device=target_continuous.device, dtype=target_continuous.dtype)
    abs_thresholds = max_vals.unsqueeze(1) * threshold_ratios.unsqueeze(0)

    left_edges = torch.cat([torch.zeros((batch_size, 1), device=target_continuous.device, dtype=target_continuous.dtype), abs_thresholds], dim=1)
    right_edges = torch.cat([abs_thresholds, max_vals.unsqueeze(1)], dim=1)
    return 0.5 * (left_edges + right_edges)


def labels_to_centers(labels, bin_centers):
    labels_safe = labels.clamp(min=0, max=bin_centers.shape[1] - 1)
    batch_size = labels_safe.shape[0]
    flat_labels = labels_safe.reshape(batch_size, -1)
    flat_centers = torch.gather(bin_centers, 1, flat_labels)
    centers = flat_centers.reshape(labels_safe.shape).unsqueeze(1)
    return centers


def select_target_channel(target, target_channel):
    if target.dim() >= 2 and target.shape[1] > 1:
        channel_idx = max(0, min(int(target_channel), target.shape[1] - 1))
        return target[:, channel_idx:channel_idx + 1, ...]
    return target


def apply_ignore_index_with_t_mask(target_discrete, output):
    t_mask = build_t_mask(output)
    t_mask_bool = t_mask.squeeze(1).bool()
    if t_mask_bool.shape[0] == 1 and target_discrete.shape[0] > 1:
        t_mask_bool = t_mask_bool.expand(target_discrete.shape[0], *t_mask_bool.shape[1:])
    target_discrete[~t_mask_bool] = -100
    return target_discrete, t_mask_bool
