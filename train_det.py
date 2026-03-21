"""
Train deterministic convolutional encoder-decoder networks (3D-only).
"""

import json
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from args_det_3d import args, device
from models.dense_ed_3d import DenseED
from training.data_3d import infer_io_shapes_from_hdf5, resolve_data_paths
from training.losses import compute_gradient_loss
from training.preprocess import (
    apply_ignore_index_with_t_mask,
    compute_bin_centers,
    continuous_to_discrete,
    crop_model_output,
    labels_to_centers,
    maybe_pad_model_input,
    select_target_channel,
)
from utils.load_data_3d import load_data
from utils.misc import mkdirs
from utils.plot_3d import plot_prediction_det, save_stats


args.train_dir = args.run_dir + "/training"
args.pred_dir = args.train_dir + "/predictions"
mkdirs([args.train_dir, args.pred_dir])

train_data_dir, test_data_dir = resolve_data_paths(args)
infer_io_shapes_from_hdf5(train_data_dir, args)

num_classes = getattr(args, "num_classes", 4)
regression_channels = 1
pad_multiple = getattr(args, "pad_multiple", 16)
cls_thresholds = getattr(args, "cls_thresholds", [0.1, 0.4, 0.7])
target_channel = getattr(args, "target_channel", 0)
lambda_reg = getattr(args, "lambda_reg", 5.0)
lambda_grad = getattr(args, "lambda_grad", 0.0)

model = DenseED(
    in_channels=getattr(args, "nic", 1),
    out_channels=num_classes + regression_channels,
    blocks=args.blocks,
    growth_rate=args.growth_rate,
    init_features=args.init_features,
    drop_rate=args.drop_rate,
    bn_size=args.bn_size,
    bottleneck=args.bottleneck,
    out_activation=None,
    output_size=None,
).to(device)
print(model)

train_loader, _ = load_data(train_data_dir, args.batch_size)
test_loader, _ = load_data(test_data_dir, args.test_batch_size)
print("Loaded data!")

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=10,
    verbose=True,
    threshold=0.0001,
    threshold_mode="rel",
    cooldown=0,
    min_lr=0,
    eps=1e-8,
)

class_weights = torch.tensor(
    getattr(args, "class_weights", [1.0, 2.0, 4.0, 8.0]), dtype=torch.float
).to(device)
criterion_ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

logger = {
    "loss_train": [],
    "loss_test": [],
    "acc_train": [],
    "acc_test": [],
    "loss_cls_train": [],
    "loss_cls_test": [],
    "loss_reg_train": [],
    "loss_reg_test": [],
    "loss_grad_train": [],
    "loss_grad_test": [],
    "time": [],
}


def _compute_soft_center(out_class, bin_centers):
    probs = F.softmax(out_class, dim=1)
    batch_size, n_classes = bin_centers.shape
    view_shape = [batch_size, n_classes] + [1] * (out_class.dim() - 2)
    bin_centers_expanded = bin_centers.view(*view_shape)
    pred_center_soft = torch.sum(probs * bin_centers_expanded, dim=1, keepdim=True)
    return pred_center_soft


def _forward_batch(input_tensor, target_tensor, use_pred_center_for_grad):
    input_padded, h_slice, w_slice = maybe_pad_model_input(input_tensor, pad_multiple=pad_multiple)
    output_padded = model(input_padded)
    output = crop_model_output(output_padded, h_slice, w_slice)

    out_class = output[:, :num_classes, ...]
    out_reg = output[:, num_classes : num_classes + 1, ...]
    target_continuous = select_target_channel(target_tensor, target_channel)

    target_discrete = continuous_to_discrete(target_continuous, cls_thresholds)
    bin_centers = compute_bin_centers(target_continuous, cls_thresholds)
    target_center = labels_to_centers(target_discrete, bin_centers)
    target_residual = target_continuous - target_center

    target_discrete_ce = target_discrete.clone()
    target_discrete_ce, t_mask_bool = apply_ignore_index_with_t_mask(target_discrete_ce, out_class)
    t_mask_reg_bool = t_mask_bool.unsqueeze(1).expand_as(out_reg)

    pred_center_soft = _compute_soft_center(out_class, bin_centers)
    pred_continuous = pred_center_soft + out_reg

    loss_cls = criterion_ce(out_class, target_discrete_ce)
    if t_mask_reg_bool.any():
        loss_reg = F.l1_loss(out_reg[t_mask_reg_bool], target_residual[t_mask_reg_bool])
        if use_pred_center_for_grad:
            grad_input = pred_continuous
        else:
            grad_input = target_center + out_reg

        loss_grad = compute_gradient_loss(
            grad_input,
            target_continuous,
            mask=t_mask_reg_bool.float(),
        ) / t_mask_reg_bool.sum().float().clamp(min=1.0)
    else:
        loss_reg = torch.tensor(0.0, device=output.device)
        loss_grad = torch.tensor(0.0, device=output.device)

    loss = loss_cls + lambda_reg * loss_reg + lambda_grad * loss_grad
    preds = torch.argmax(out_class, dim=1)
    valid_mask = target_discrete_ce != -100
    correct = (preds[valid_mask] == target_discrete_ce[valid_mask]).sum().item()
    total_valid = valid_mask.sum().item()

    return {
        "loss": loss,
        "loss_cls": loss_cls,
        "loss_reg": loss_reg,
        "loss_grad": loss_grad,
        "correct": correct,
        "total_valid": total_valid,
        "pred_continuous": pred_continuous,
        "target_continuous": target_continuous,
        "t_mask_reg_bool": t_mask_reg_bool,
    }


def test(epoch):
    model.eval()
    total_loss = 0.0
    total_loss_cls = 0.0
    total_loss_reg = 0.0
    total_loss_grad = 0.0
    correct_pixels = 0
    total_valid_pixels = 0

    for batch_idx, (input_tensor, target_tensor) in enumerate(test_loader):
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        metrics = _forward_batch(input_tensor, target_tensor, use_pred_center_for_grad=True)
        total_loss += metrics["loss"].item()
        total_loss_cls += metrics["loss_cls"].item()
        total_loss_reg += metrics["loss_reg"].item()
        total_loss_grad += metrics["loss_grad"].item()
        correct_pixels += metrics["correct"]
        total_valid_pixels += metrics["total_valid"]

        if epoch % args.plot_freq == 0 and batch_idx == 0:
            n_samples = 6 if epoch == args.epochs else 2
            idx = torch.randperm(input_tensor.size(0))[:n_samples]

            pred_continuous_plot = metrics["pred_continuous"].clone()
            pred_continuous_plot[~metrics["t_mask_reg_bool"]] = 0.0

            samples_output = pred_continuous_plot.data.cpu()[idx].numpy()
            samples_target = metrics["target_continuous"].data.cpu()[idx].numpy()

            for i in range(n_samples):
                print("epoch {}: plotting prediction {}".format(epoch, i))
                plot_prediction_det(
                    args.pred_dir,
                    samples_target[i],
                    samples_output[i],
                    epoch,
                    i,
                    plot_fn=args.plot_fn,
                )

    acc = correct_pixels / total_valid_pixels if total_valid_pixels > 0 else 0.0
    avg_loss = total_loss / len(test_loader)
    avg_loss_cls = total_loss_cls / len(test_loader)
    avg_loss_reg = total_loss_reg / len(test_loader)
    avg_loss_grad = total_loss_grad / len(test_loader)

    print(
        "epoch: {}, test loss: {:.4f} (cls: {:.4f}, reg: {:.4f}, grad: {:.4f}), test accuracy: {:.2f}%".format(
            epoch, avg_loss, avg_loss_cls, avg_loss_reg, avg_loss_grad, acc * 100
        )
    )

    if epoch % args.log_freq == 0:
        logger["loss_test"].append(avg_loss)
        logger["acc_test"].append(acc)
        logger["loss_cls_test"].append(avg_loss_cls)
        logger["loss_reg_test"].append(avg_loss_reg)
        logger["loss_grad_test"].append(avg_loss_grad)


def train(epoch):
    model.train()
    total_loss = 0.0
    total_loss_cls = 0.0
    total_loss_reg = 0.0
    total_loss_grad = 0.0
    correct_pixels = 0
    total_valid_pixels = 0

    for input_tensor, target_tensor in train_loader:
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        model.zero_grad()
        metrics = _forward_batch(input_tensor, target_tensor, use_pred_center_for_grad=False)
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += metrics["loss"].item()
        total_loss_cls += metrics["loss_cls"].item()
        total_loss_reg += metrics["loss_reg"].item()
        total_loss_grad += metrics["loss_grad"].item()
        correct_pixels += metrics["correct"]
        total_valid_pixels += metrics["total_valid"]

    avg_loss = total_loss / len(train_loader)
    avg_loss_cls = total_loss_cls / len(train_loader)
    avg_loss_reg = total_loss_reg / len(train_loader)
    avg_loss_grad = total_loss_grad / len(train_loader)
    acc = correct_pixels / total_valid_pixels if total_valid_pixels > 0 else 0.0

    scheduler.step(avg_loss)
    print(
        "epoch: {}, training loss: {:.4f} (cls: {:.4f}, reg: {:.4f}, grad: {:.4f}), accuracy: {:.2f}%".format(
            epoch, avg_loss, avg_loss_cls, avg_loss_reg, avg_loss_grad, acc * 100
        )
    )

    if epoch % args.log_freq == 0:
        logger["loss_train"].append(avg_loss)
        logger["acc_train"].append(acc)
        logger["loss_cls_train"].append(avg_loss_cls)
        logger["loss_reg_train"].append(avg_loss_reg)
        logger["loss_grad_train"].append(avg_loss_grad)

    if epoch % args.ckpt_freq == 0:
        torch.save(model.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch))


def _load_checkpoint_for_post_mode():
    if args.ckpt_epoch is not None:
        checkpoint = args.ckpt_dir + "/model_epoch{}.pth".format(args.ckpt_epoch)
    else:
        checkpoint = args.ckpt_dir + "/model_epoch{}.pth".format(args.epochs)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print("Loaded pre-trained model: {}".format(checkpoint))


print("Start training........................................................")
tic = time()

if args.post:
    _load_checkpoint_for_post_mode()
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
    print(
        "Finished training {} epochs with {} data using {} seconds (including long plotting time)".format(
            args.epochs, args.ntrain, tic2 - tic
        )
    )

x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq)
logger["time"].append(tic2 - tic)
save_stats(args.train_dir, logger, x_axis)

args.training_time = tic2 - tic
args.n_params, args.n_layers = model._num_parameters_convlayers()
with open(args.run_dir + "/args.txt", "w") as args_file:
    json.dump(vars(args), args_file, indent=4)
