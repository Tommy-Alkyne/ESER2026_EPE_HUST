"""
Train deterministic convolutional encoder-decoder networks
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import sys
import os

# choose args module based on command-line flag --ndim 3 or --use-3d
use_3d = False
if '--ndim' in sys.argv:
    try:
        idx = sys.argv.index('--ndim')
        if sys.argv[idx + 1] == '3':
            use_3d = True
    except Exception:
        pass
if '--use-3d' in sys.argv:
    use_3d = True

if use_3d:
    from args_det_3d import args, device
else:
    from args_det import args, device
from utils.misc import mkdirs

# choose 2D/3D implementations depending on args.ndim (default 2)
if getattr(args, 'ndim', 2) == 3:
    from models.dense_ed_3d import DenseED as DenseED
    from utils.load_data_3d import load_data
    from utils.plot_3d import plot_prediction_det, save_stats
else:
    from models.dense_ed import DenseED
    from utils.load_data import load_data
    from utils.plot import plot_prediction_det, save_stats
import json
from time import time

args.train_dir = args.run_dir + "/training"
args.pred_dir = args.train_dir + "/predictions"
mkdirs([args.train_dir, args.pred_dir])

# load checkpoint if in post mode
# if args.post:
#     if args.ckpt_epoch is not None:
#         checkpoint = args.ckpt_dir + '/model_epoch{}.pth'.format(args.ckpt_epoch)
#     else:
#         checkpoint = args.ckpt_dir + '/model_epoch{}.pth'.format(args.epochs)
#     model.load_state_dict(torch.load(checkpoint))
#     print('Loaded pre-trained model: {}'.format(checkpoint))

# load data
# allow overriding train/test files via command-line --train-file / --test-file
train_file = None
test_file = None
if '--train-file' in sys.argv:
    try:
        train_file = sys.argv[sys.argv.index('--train-file') + 1]
    except Exception:
        train_file = None
if '--test-file' in sys.argv:
    try:
        test_file = sys.argv[sys.argv.index('--test-file') + 1]
    except Exception:
        test_file = None

if train_file is not None:
    # If user passed an absolute path, use it. If user passed a relative path
    # that contains a directory component (e.g. 'dataset/10243d.hdf5'),
    # assume they intended that relative path and use it as-is. Only join
    # with args.data_dir when the user provided a bare filename.
    if os.path.isabs(train_file):
        train_data_dir = train_file
    elif os.path.dirname(train_file):
        train_data_dir = train_file
    else:
        train_data_dir = os.path.join(args.data_dir, train_file)
elif getattr(args, 'ndim', 2) == 3:
    train_data_dir = os.path.join(args.data_dir, '10243d.hdf5')
else:
    train_data_dir = args.data_dir + '/kle{}_lhs{}.hdf5'.format(args.kle, args.ntrain)

if test_file is not None:
    if os.path.isabs(test_file):
        test_data_dir = test_file
    elif os.path.dirname(test_file):
        test_data_dir = test_file
    else:
        test_data_dir = os.path.join(args.data_dir, test_file)
elif getattr(args, 'ndim', 2) == 3:
    test_data_dir = train_data_dir
else:
    test_data_dir = args.data_dir + '/kle{}_mc{}.hdf5'.format(args.kle, args.ntest)

# if only a training file was given (no --test-file), split it into train/test
args.split_train = train_file is not None and test_file is None
if args.split_train:
    print('Only training data provided (no --test-file): splitting {} '
          'into train/test at ratio {}.'.format(train_data_dir, args.split_ratio))

# auto-detect channels and output size for 3D from training file
if getattr(args, 'ndim', 2) == 3:
    try:
        with h5py.File(train_data_dir, 'r') as f:
            y_shape = f['output'].shape
            if len(y_shape) >= 5:
                args.noc = y_shape[1]
                args.image_size = list(y_shape[-3:])
            elif len(y_shape) == 4:
                args.noc = y_shape[1]
                args.image_size = [1] + list(y_shape[-2:])

            if 'spatial_input' in f:
                spatial_shape = f['spatial_input'].shape
                args.nic = spatial_shape[1]
                if 'input' in f:
                    args.nic += f['input'].shape[1]
                if 'sensor_input' in f:
                    args.nic += f['sensor_input'].shape[1]
            elif 'input' in f:
                x_shape = f['input'].shape
                if len(x_shape) in [4, 5]:
                    args.nic = x_shape[1]
                else:
                    args.nic = int(np.prod(x_shape[1:]))
                if 'sensor_input' in f:
                    args.nic += int(np.prod(f['sensor_input'].shape[1:]))
            elif 'sensor_input' in f:
                x_shape = f['sensor_input'].shape
                if len(x_shape) in [4, 5]:
                    args.nic = x_shape[1]
                else:
                    args.nic = int(np.prod(x_shape[1:]))
            else:
                args.nic = 1
    except Exception as e:
        print('Warning: failed to infer 3D data shapes from {}: {}'.format(train_data_dir, e))

# initialize model
num_classes = getattr(args, 'num_classes', 4)
regression_channels = 1
model_kwargs = dict(in_channels=getattr(args, 'nic', 1), out_channels=num_classes + regression_channels,
                    blocks=args.blocks,
                    growth_rate=args.growth_rate,
                    init_features=args.init_features,
                    drop_rate=args.drop_rate,
                    bn_size=args.bn_size,
                    bottleneck=args.bottleneck,
                    out_activation=None)
if getattr(args, 'ndim', 2) == 3:
    model_kwargs['output_size'] = None
model = DenseED(**model_kwargs).to(device)
print(model)

if getattr(args, 'split_train', False) and test_file is None:
    # single training file provided: randomly split it into train/test
    train_loader, test_loader, train_stats = load_data(
        train_data_dir, args.batch_size,
        test_batch_size=args.test_batch_size,
        split_ratio=args.split_ratio,
        split_seed=args.seed)
    test_stats = train_stats
else:
    train_loader, train_stats = load_data(train_data_dir, args.batch_size)
    test_loader, test_stats = load_data(test_data_dir, args.test_batch_size)
print('Loaded data!')

optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10,
                    threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8)

class_weights = torch.tensor(getattr(args, 'class_weights', [1.0, 2.0, 4.0, 8.0]), dtype=torch.float).to(device)
criterion_ce = nn.CrossEntropyLoss(
    weight=class_weights, 
    ignore_index=-100)
lambda_reg = getattr(args, 'lambda_reg', 5.0)
lambda_grad = getattr(args, 'lambda_grad', 0.0)

logger = {}
logger['loss_train'] = []
logger['loss_test'] = []
logger['acc_train'] = []
logger['acc_test'] = []
logger['loss_cls_train'] = []
logger['loss_cls_test'] = []
logger['loss_reg_train'] = []
logger['loss_reg_test'] = []
logger['loss_grad_train'] = []
logger['loss_grad_test'] = []
logger['time'] = []

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
    if output.dim() == 4:
        return mask_2d.unsqueeze(0).unsqueeze(0)
    if output.dim() == 5:
        d = output.shape[-3]
        return mask_2d.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(1, 1, d, 1, 1)
    raise ValueError('Unsupported output dim {}'.format(output.dim()))


def pad_hw_to_multiple(x, multiple=16, mode='replicate'):
    """Pad H/W to the next multiple and return crop slices for restoration."""
    if x.dim() not in (4, 5):
        raise ValueError('Unsupported input dim {}'.format(x.dim()))

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

    x_padded = F.pad(x, pad_dims, mode=mode)
    h_slice = slice(pad_top, pad_top + h)
    w_slice = slice(pad_left, pad_left + w)
    return x_padded, h_slice, w_slice


def maybe_pad_model_input(x):
    """Only pad 3D training inputs; keep 2D path unchanged."""
    if getattr(args, 'ndim', 2) != 3:
        return x, None, None
    return pad_hw_to_multiple(x, multiple=getattr(args, 'pad_multiple', 16), mode='replicate')


def crop_model_output(output, h_slice, w_slice):
    if h_slice is None or w_slice is None:
        return output
    return output[..., h_slice, w_slice]


# def continuous_to_discrete(target_continuous, thresholds=[0.02, 0.08, 0.15]):
#     target_discrete = torch.zeros_like(target_continuous, dtype=torch.long)
#     target_discrete[target_continuous >= thresholds[0]] = 1
#     target_discrete[target_continuous >= thresholds[1]] = 2
#     target_discrete[target_continuous >= thresholds[2]] = 3
#     if target_discrete.dim() >= 2 and target_discrete.shape[1] == 1:
#         target_discrete = target_discrete.squeeze(1)
#     return target_discrete

def continuous_to_discrete(target_continuous, thresholds=None):
    if thresholds is None:
        thresholds = getattr(args, 'cls_thresholds', [0.1, 0.4, 0.7])
    target_discrete = torch.zeros_like(target_continuous, dtype=torch.long)
    B = target_continuous.shape[0]

    for i in range(B):
        max_val = target_continuous[i].max().clamp(min=1e-8)
        for class_idx, threshold in enumerate(thresholds, start=1):
            target_discrete[i, target_continuous[i] >= threshold * max_val] = class_idx

    if target_discrete.dim() >= 2 and target_discrete.shape[1] == 1:
        target_discrete = target_discrete.squeeze(1)

    return target_discrete


def compute_bin_centers(target_continuous, thresholds=None):
    if thresholds is None:
        thresholds = getattr(args, 'cls_thresholds', [0.1, 0.4, 0.7])
    B = target_continuous.shape[0]
    max_vals = target_continuous.reshape(B, -1).max(dim=1).values.clamp(min=1e-8)
    threshold_ratios = torch.tensor(thresholds, device=target_continuous.device, dtype=target_continuous.dtype)
    abs_thresholds = max_vals.unsqueeze(1) * threshold_ratios.unsqueeze(0)

    left_edges = torch.cat([torch.zeros((B, 1), device=target_continuous.device, dtype=target_continuous.dtype), abs_thresholds], dim=1)
    right_edges = torch.cat([abs_thresholds, max_vals.unsqueeze(1)], dim=1)
    return 0.5 * (left_edges + right_edges)


def labels_to_centers(labels, bin_centers):
    labels_safe = labels.clamp(min=0, max=bin_centers.shape[1] - 1)
    B = labels_safe.shape[0]
    flat_labels = labels_safe.reshape(B, -1)
    flat_centers = torch.gather(bin_centers, 1, flat_labels)
    centers = flat_centers.reshape(labels_safe.shape).unsqueeze(1)
    return centers


def select_target_channel(target):
    if target.dim() >= 2 and target.shape[1] > 1:
        channel_idx = int(getattr(args, 'target_channel', 0))
        channel_idx = max(0, min(channel_idx, target.shape[1] - 1))
        return target[:, channel_idx:channel_idx + 1, ...]
    return target


def apply_ignore_index_with_t_mask(target_discrete, output):
    t_mask = build_t_mask(output)
    t_mask_bool = t_mask.squeeze(1).bool()
    if t_mask_bool.shape[0] == 1 and target_discrete.shape[0] > 1:
        t_mask_bool = t_mask_bool.expand(target_discrete.shape[0], *t_mask_bool.shape[1:])
    target_discrete[~t_mask_bool] = -100
    return target_discrete, t_mask_bool


def compute_gradient_loss(output, target, mask=None):
    if output.dim() == 4:
        grad_output_y = output[:, :, :, 1:] - output[:, :, :, :-1]
        grad_output_x = output[:, :, 1:, :] - output[:, :, :-1, :]
        grad_target_y = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_target_x = target[:, :, 1:, :] - target[:, :, :-1, :]
        if mask is not None:
            mask_y = mask[:, :, :, 1:] * mask[:, :, :, :-1]
            mask_x = mask[:, :, 1:, :] * mask[:, :, :-1, :]
            grad_output_y = grad_output_y * mask_y
            grad_output_x = grad_output_x * mask_x
            grad_target_y = grad_target_y * mask_y
            grad_target_x = grad_target_x * mask_x
        loss_y = F.l1_loss(grad_output_y, grad_target_y, reduction='sum')
        loss_x = F.l1_loss(grad_output_x, grad_target_x, reduction='sum')
        return loss_y + loss_x
    if output.dim() == 5:
        grad_output_z = output[:, :, 1:, :, :] - output[:, :, :-1, :, :]
        grad_output_y = output[:, :, :, 1:, :] - output[:, :, :, :-1, :]
        grad_output_x = output[:, :, :, :, 1:] - output[:, :, :, :, :-1]
        grad_target_z = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        grad_target_y = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        grad_target_x = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        if mask is not None:
            mask_z = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
            mask_y = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
            mask_x = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]
            grad_output_z = grad_output_z * mask_z
            grad_output_y = grad_output_y * mask_y
            grad_output_x = grad_output_x * mask_x
            grad_target_z = grad_target_z * mask_z
            grad_target_y = grad_target_y * mask_y
            grad_target_x = grad_target_x * mask_x
        loss_z = F.l1_loss(grad_output_z, grad_target_z, reduction='sum')
        loss_y = F.l1_loss(grad_output_y, grad_target_y, reduction='sum')
        loss_x = F.l1_loss(grad_output_x, grad_target_x, reduction='sum')
        return loss_z + loss_y + loss_x
    raise ValueError('Unsupported output dim {}'.format(output.dim()))

def test(epoch):#epoch is a number
    model.eval()
    total_loss = 0.
    total_loss_cls = 0.
    total_loss_reg = 0.
    total_loss_grad = 0.
    correct_pixels = 0
    total_valid_pixels = 0
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        input_padded, h_slice, w_slice = maybe_pad_model_input(input)
        output_padded = model(input_padded)
        output = crop_model_output(output_padded, h_slice, w_slice)

        out_class = output[:, :num_classes, ...]
        out_reg = output[:, num_classes:num_classes + 1, ...]
        target_continuous = select_target_channel(target)

        target_discrete = continuous_to_discrete(target_continuous)
        bin_centers = compute_bin_centers(target_continuous)
        target_center = labels_to_centers(target_discrete, bin_centers)
        target_residual = target_continuous - target_center
        target_discrete_ce = target_discrete.clone()
        target_discrete_ce, t_mask_bool = apply_ignore_index_with_t_mask(target_discrete_ce, out_class)

        # Soft-argmax center for smoother gradient supervision field
        probs = F.softmax(out_class, dim=1)
        B_size, C_size = bin_centers.shape
        view_shape = [B_size, C_size] + [1] * (out_class.dim() - 2)
        bin_centers_expanded = bin_centers.view(*view_shape)
        pred_center_soft = torch.sum(probs * bin_centers_expanded, dim=1, keepdim=True)

        loss_cls = criterion_ce(out_class, target_discrete_ce)
        t_mask_reg_bool = t_mask_bool.unsqueeze(1).expand_as(out_reg)
        pred_continuous_for_loss = pred_center_soft + out_reg
        if t_mask_reg_bool.any():
            loss_reg = F.l1_loss(out_reg[t_mask_reg_bool], target_residual[t_mask_reg_bool])
            loss_grad = compute_gradient_loss(
                pred_continuous_for_loss,
                target_continuous,
                mask=t_mask_reg_bool.float()
            ) / t_mask_reg_bool.sum().float().clamp(min=1.0)
        else:
            loss_reg = torch.tensor(0.0, device=output.device)
            loss_grad = torch.tensor(0.0, device=output.device)
        loss = loss_cls + lambda_reg * loss_reg + lambda_grad * loss_grad

        total_loss += loss.item()
        total_loss_cls += loss_cls.item()
        total_loss_reg += loss_reg.item()
        total_loss_grad += loss_grad.item()

        # ================== Soft-Argmax smooth assembly ==================
        # 1) Keep argmax for classification accuracy metric
        preds = torch.argmax(out_class, dim=1)

        # 2) Convert logits to probabilities
        probs = F.softmax(out_class, dim=1)

        # 3) Expand bin centers to match feature-map rank
        B_size, C_size = bin_centers.shape
        view_shape = [B_size, C_size] + [1] * (out_class.dim() - 2)
        bin_centers_expanded = bin_centers.view(*view_shape)

        # 4) Probability-weighted center (smooth baseline)
        pred_center_soft = torch.sum(probs * bin_centers_expanded, dim=1, keepdim=True)

        # 5) Add residual branch for final continuous prediction
        pred_continuous = pred_center_soft + out_reg
        # ===============================================================
        valid_mask = (target_discrete_ce != -100)
        correct_pixels += (preds[valid_mask] == target_discrete_ce[valid_mask]).sum().item()
        total_valid_pixels += valid_mask.sum().item()

        # plot predictions
        if epoch % args.plot_freq == 0 and batch_idx == 0:
            n_samples = 6 if epoch == args.epochs else 2
            idx = torch.randperm(input.size(0))[:n_samples]

            pred_continuous_plot = pred_continuous.clone()
            pred_continuous_plot[~t_mask_reg_bool] = 0.0
            samples_output = pred_continuous_plot.data.cpu()[idx].numpy()
            samples_target = target_continuous.data.cpu()[idx].numpy()

            for i in range(n_samples):
                print('epoch {}: plotting prediction {}'.format(epoch, i))
                plot_prediction_det(args.pred_dir, samples_target[i], 
                    samples_output[i], epoch, i, plot_fn=args.plot_fn)

    acc = correct_pixels / total_valid_pixels if total_valid_pixels > 0 else 0.0
    avg_loss = total_loss / len(test_loader)
    avg_loss_cls = total_loss_cls / len(test_loader)
    avg_loss_reg = total_loss_reg / len(test_loader)
    avg_loss_grad = total_loss_grad / len(test_loader)
    print("epoch: {}, test loss: {:.4f} (cls: {:.4f}, reg: {:.4f}, grad: {:.4f}), test accuracy: {:.2f}%".format(
        epoch, avg_loss, avg_loss_cls, avg_loss_reg, avg_loss_grad, acc * 100))

    if epoch % args.log_freq == 0:
        logger['loss_test'].append(avg_loss)
        logger['acc_test'].append(acc)
        logger['loss_cls_test'].append(avg_loss_cls)
        logger['loss_reg_test'].append(avg_loss_reg)
        logger['loss_grad_test'].append(avg_loss_grad)

def train(epoch):
    model.train()
    total_loss = 0.
    total_loss_cls = 0.
    total_loss_reg = 0.
    total_loss_grad = 0.
    correct_pixels = 0
    total_valid_pixels = 0
    for _, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        model.zero_grad()
        input_padded, h_slice, w_slice = maybe_pad_model_input(input)
        output_padded = model(input_padded)
        output = crop_model_output(output_padded, h_slice, w_slice)

        out_class = output[:, :num_classes, ...]
        out_reg = output[:, num_classes:num_classes + 1, ...]
        target_continuous = select_target_channel(target)

        target_discrete = continuous_to_discrete(target_continuous)
        bin_centers = compute_bin_centers(target_continuous)
        target_center = labels_to_centers(target_discrete, bin_centers)
        target_residual = target_continuous - target_center
        target_discrete_ce = target_discrete.clone()
        target_discrete_ce, t_mask_bool = apply_ignore_index_with_t_mask(target_discrete_ce, out_class)

        loss_cls = criterion_ce(out_class, target_discrete_ce)
        t_mask_reg_bool = t_mask_bool.unsqueeze(1).expand_as(out_reg)
        pred_continuous_for_loss = target_center + out_reg
        if t_mask_reg_bool.any():
            loss_reg = F.l1_loss(out_reg[t_mask_reg_bool], target_residual[t_mask_reg_bool])
            loss_grad = compute_gradient_loss(
                pred_continuous_for_loss,
                target_continuous,
                mask=t_mask_reg_bool.float()
            ) / t_mask_reg_bool.sum().float().clamp(min=1.0)
        else:
            loss_reg = torch.tensor(0.0, device=output.device)
            loss_grad = torch.tensor(0.0, device=output.device)
        loss = loss_cls + lambda_reg * loss_reg + lambda_grad * loss_grad

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_loss_cls += loss_cls.item()
        total_loss_reg += loss_reg.item()
        total_loss_grad += loss_grad.item()
        with torch.no_grad():
            # ================== Soft-Argmax smooth assembly ==================
            # 1) Keep argmax for classification accuracy metric
            preds = torch.argmax(out_class, dim=1)

            # 2) Convert logits to probabilities
            probs = F.softmax(out_class, dim=1)

            # 3) Expand bin centers to match feature-map rank
            B_size, C_size = bin_centers.shape
            view_shape = [B_size, C_size] + [1] * (out_class.dim() - 2)
            bin_centers_expanded = bin_centers.view(*view_shape)

            # 4) Probability-weighted center (smooth baseline)
            pred_center_soft = torch.sum(probs * bin_centers_expanded, dim=1, keepdim=True)

            # 5) Add residual branch for final continuous prediction
            pred_continuous = pred_center_soft + out_reg
            # ===============================================================

            valid_mask = (target_discrete_ce != -100)
            correct_pixels += (preds[valid_mask] == target_discrete_ce[valid_mask]).sum().item()
            total_valid_pixels += valid_mask.sum().item()

    avg_loss = total_loss / len(train_loader)
    avg_loss_cls = total_loss_cls / len(train_loader)
    avg_loss_reg = total_loss_reg / len(train_loader)
    avg_loss_grad = total_loss_grad / len(train_loader)
    acc = correct_pixels / total_valid_pixels if total_valid_pixels > 0 else 0.0
    scheduler.step(acc)
    print("epoch: {}, training loss: {:.4f} (cls: {:.4f}, reg: {:.4f}, grad: {:.4f}), accuracy: {:.2f}%".format(
        epoch, avg_loss, avg_loss_cls, avg_loss_reg, avg_loss_grad, acc * 100))

    if epoch % args.log_freq == 0:
        logger['loss_train'].append(avg_loss)
        logger['acc_train'].append(acc)
        logger['loss_cls_train'].append(avg_loss_cls)
        logger['loss_reg_train'].append(avg_loss_reg)
        logger['loss_grad_train'].append(avg_loss_grad)
    # save model
    if epoch % args.ckpt_freq == 0:
        torch.save(model.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch))


print('Start training........................................................')
tic = time()
if args.post:
    print("In post-processing mode, skipping training.")
    with torch.no_grad():
        test(args.ckpt_epoch if args.ckpt_epoch is not None else args.epochs)
    tic2 = time()
    print("Finished testing using pre-trained model in {} seconds".format(tic2 - tic))
else:
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        with torch.no_grad():
            test(epoch)
    tic2 = time()
    print("Finished training {} epochs with {} data using {} seconds (including long... plotting time)"
        .format(args.epochs, args.ntrain, tic2 - tic))

x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq)
logger['time'].append(tic2 - tic)
# plot the rmse, r2-score curve and save them in txt
save_stats(args.train_dir, logger, x_axis)

args.training_time = tic2 - tic
args.n_params, args.n_layers = model._num_parameters_convlayers()
with open(args.run_dir + "/args.txt", 'w') as args_file:
    json.dump(vars(args), args_file, indent=4)
