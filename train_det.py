"""
Train deterministic convolutional encoder-decoder networks
"""

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import h5py
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import os
from time import time
import json

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

if getattr(args, 'ndim', 2) == 3:
    from models.dense_ed_3d import DenseED as DenseED
    from utils.load_data_3d import load_data
    from utils.plot_3d import plot_prediction_det, save_stats
else:
    from models.dense_ed import DenseED
    from utils.load_data import load_data
    from utils.plot import plot_prediction_det, save_stats

args.train_dir = args.run_dir + "/training"
args.pred_dir = args.train_dir + "/predictions"
mkdirs([args.train_dir, args.pred_dir])

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

# --- MODIFIED: 强制将默认训练集指定为你现有的 p_v_256.hdf5 ---
if train_file is not None:
    if os.path.isabs(train_file):
        train_data_dir = train_file
    elif os.path.dirname(train_file):
        train_data_dir = train_file
    else:
        train_data_dir = os.path.join(args.data_dir, train_file)
else:
    # 无论 2D 还是 3D，默认都去 dataset 文件夹下找 p_v_256.hdf5
    train_data_dir = os.path.join(args.data_dir, 'p_v_256.hdf5')

# --- MODIFIED: 强制将默认测试集指定为你现有的 p_v_256.hdf5 ---
if test_file is not None:
    if os.path.isabs(test_file):
        test_data_dir = test_file
    elif os.path.dirname(test_file):
        test_data_dir = test_file
    else:
        test_data_dir = os.path.join(args.data_dir, test_file)
else:
    test_data_dir = os.path.join(args.data_dir, 'p_v_256.hdf5')


# 自动推断通道数和尺寸
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

            if 'sensor_input' in f:
                x_shape = f['sensor_input'].shape
            else:
                x_shape = f['input'].shape
            if len(x_shape) in [4, 5]:
                args.nic = x_shape[1]
            else:
                args.nic = 1
    except Exception as e:
        print('Warning: failed to infer 3D data shapes from {}: {}'.format(train_data_dir, e))
else:
    # 对于2D模式，也自动检测通道数
    try:
        with h5py.File(train_data_dir, 'r') as f:
            # 检测输出通道数
            y_shape = f['output'].shape
            if len(y_shape) == 5 and y_shape[2] == 1:  # (N, C, 1, H, W) 压缩后变成 (N, C, H, W)
                args.noc = y_shape[1]
            elif len(y_shape) == 4:
                args.noc = y_shape[1]
            else:
                args.noc = getattr(args, 'noc', 3)
            
            # 检测输入通道数
            if 'sensor_input' in f:
                x_shape = f['sensor_input'].shape
            else:
                x_shape = f['input'].shape
            if len(x_shape) == 3:  # (N, H, W) 会在load_data转为 (N, 1, H, W)
                args.nic = 1
            elif len(x_shape) >= 4:
                args.nic = x_shape[1]
            else:
                args.nic = 1
    except Exception as e:
        print('Warning: failed to infer 2D data shapes from {}: {}'.format(train_data_dir, e))

model_kwargs = {
    'in_channels': getattr(args, 'nic', 1),
    'out_channels': getattr(args, 'noc', 3),
    'blocks': args.blocks,
    'growth_rate': args.growth_rate,
    'init_features': args.init_features,
    'drop_rate': args.drop_rate,
    'bn_size': args.bn_size,
    'bottleneck': args.bottleneck,
    'out_activation': None
}

if getattr(args, 'ndim', 2) == 3:
    model_kwargs['output_size'] = tuple(getattr(args, 'image_size', [1, 65, 65]))

model = DenseED(**model_kwargs).to(device)
print(model)

# 这里就会直接去加载 ./dataset/p_v_256.hdf5 了
train_loader, _ = load_data(train_data_dir, args.batch_size)
test_loader, _ = load_data(test_data_dir, args.test_batch_size)
print('Loaded data successfully from:', train_data_dir)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

logger = {'rmse_train': [], 'rmse_test': [], 'r2_train': [], 'r2_test': [], 
          'loss_train': [], 'loss_test': [], 'time': []}
best_r2 = -float('inf') 

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

def get_spatial_gradients_1st_2nd(tensor):
    """同时计算一阶和二阶空间梯度，二阶梯度直接对应流体力学的拉普拉斯粘性项"""
    grads_1st, grads_2nd = [], []
    for dim in range(2, tensor.dim()):
        sp, sm = [slice(None)] * tensor.dim(), [slice(None)] * tensor.dim()
        sp[dim], sm[dim] = slice(1, None), slice(None, -1)
        grads_1st.append(tensor[tuple(sp)] - tensor[tuple(sm)])
        
        sc = [slice(None)] * tensor.dim()
        sc[dim] = slice(1, -1)
        sp2, sm2 = [slice(None)] * tensor.dim(), [slice(None)] * tensor.dim()
        sp2[dim], sm2[dim] = slice(2, None), slice(None, -2)
        grads_2nd.append(tensor[tuple(sp2)] - 2.0 * tensor[tuple(sc)] + tensor[tuple(sm2)])
    return grads_1st, grads_2nd

def test(epoch):
    global best_r2
    model.eval()
    mse = 0.
    n_out_pixels_test = 0
    masked_sum_y, masked_sum_y2 = 0., 0.
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            t_mask = build_t_mask(output)
            t_mask = t_mask.expand(output.shape[0], output.shape[1], *output.shape[2:])
            masked_output, masked_target = output * t_mask, target * t_mask
            
            mse += F.mse_loss(masked_output, masked_target, reduction='sum').item()
            n_out_pixels_test += t_mask.sum().item()
            masked_sum_y += masked_target.sum().item()
            masked_sum_y2 += (masked_target ** 2).sum().item()

            if epoch % args.plot_freq == 0 and batch_idx == 0:
                n_samples = 6 if epoch == args.epochs else 2
                idx = torch.randperm(input.size(0))[:n_samples]
                samples_output = masked_output.data.cpu()[idx].numpy()
                samples_target = masked_target.data.cpu()[idx].numpy()

                for i in range(n_samples):
                    plot_prediction_det(args.pred_dir, samples_target[i], 
                        samples_output[i], epoch, i, plot_fn=args.plot_fn)

    rmse_test = np.sqrt(mse / max(n_out_pixels_test, 1.0))
    ss_tot = masked_sum_y2 - (masked_sum_y ** 2) / max(n_out_pixels_test, 1.0)
    r2_score = 1 - mse / max(ss_tot, 1e-12)
    print("epoch: {}, test    r2-score:  {:.6f}, rmse: {:.6f}".format(epoch, r2_score, rmse_test))

    # 保存最佳模型及其检查点
    if r2_score > best_r2:
        best_r2 = r2_score
        save_path = os.path.join(args.ckpt_dir, "best_p_v_256_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"--> [New Best R2: {best_r2:.6f}] Saved model to {save_path}")

    if epoch % args.log_freq == 0:
        logger['r2_test'].append(r2_score)
        logger['rmse_test'].append(rmse_test)
        logger['loss_test'].append(rmse_test)

def train(epoch):
    """
    优化的训练函数，采用多阶段训练策略和动态损失权重：
    - Epoch 1-50: 重点学习基础特征 (lambda_grad = 0.3)
    - Epoch 51-150: 增强物理约束 (lambda_grad = 1.5) 
    - Epoch 151+: 精细调整所有项 (lambda_grad = 2.0)
    """
    model.train()
    mse = 0.
    n_out_pixels_train = 0
    masked_sum_y, masked_sum_y2 = 0., 0.
    total_loss = 0.
    
    # 动态调整梯度损失权重 - 多阶段训练策略
    if epoch <= 50:
        lambda_grad = 0.3
        lambda_l1_base = 0.5
    elif epoch <= 150:
        lambda_grad = 1.5
        lambda_l1_base = 0.7
    else:
        lambda_grad = 2.0
        lambda_l1_base = 0.8
    
    for _, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        model.zero_grad()
        output = model(input)
        t_mask = build_t_mask(output)
        t_mask = t_mask.expand(output.shape[0], output.shape[1], *output.shape[2:])
        
        masked_output, masked_target = output * t_mask, target * t_mask
        pixel_count = max(t_mask.sum().item(), 1.0)
        
        # 1. 基础损失：加权的MSE + L1混合损失，更多偏向于相似性
        loss_mse = F.mse_loss(masked_output, masked_target, reduction='sum') / pixel_count
        loss_l1 = F.l1_loss(masked_output, masked_target, reduction='sum') / pixel_count
        loss_base = 1.0 * loss_mse + lambda_l1_base * loss_l1
        
        # 2. 梯度损失：一阶和二阶梯度约束
        loss_grad = 0.0
        out_g1, out_g2 = get_spatial_gradients_1st_2nd(masked_output)
        tar_g1, tar_g2 = get_spatial_gradients_1st_2nd(masked_target)
        
        # 一阶梯度损失（边界约束）
        for og, tg in zip(out_g1, tar_g1):
            loss_grad += 0.5 * F.l1_loss(og, tg, reduction='sum') / pixel_count
        
        # 二阶梯度损失（流体力学粘性约束 - 最重要!）
        for og, tg in zip(out_g2, tar_g2):
            loss_grad += 1.0 * F.l1_loss(og, tg, reduction='sum') / pixel_count
        
        # 3. 组合总损失
        loss = loss_base + lambda_grad * loss_grad
        
        loss.backward()
        optimizer.step()
        
        mse += F.mse_loss(masked_output, masked_target, reduction='sum').item()
        n_out_pixels_train += t_mask.sum().item()
        masked_sum_y += masked_target.sum().item()
        masked_sum_y2 += (masked_target ** 2).sum().item()
        total_loss += loss.item()

    rmse = np.sqrt(mse / max(n_out_pixels_train, 1.0))
    ss_tot = masked_sum_y2 - (masked_sum_y ** 2) / max(n_out_pixels_train, 1.0)
    r2_score = 1 - mse / max(ss_tot, 1e-12)
    avg_loss = total_loss / max(len(train_loader), 1)
    print("epoch: {}, training r2-score: {:.6f}, loss: {:.6f}, lambda_grad: {:.2f}".format(
        epoch, r2_score, avg_loss, lambda_grad))
    
    if epoch % args.log_freq == 0:
        logger['r2_train'].append(r2_score)
        logger['rmse_train'].append(rmse)
        logger['loss_train'].append(avg_loss)

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
        scheduler.step()
        
    tic2 = time()
    print("Finished training {} epochs with {} data using {} seconds"
        .format(args.epochs, args.ntrain, tic2 - tic))

x_axis = np.arange(args.log_freq, args.epochs + args.log_freq, args.log_freq)
logger['time'].append(tic2 - tic)
save_stats(args.train_dir, logger, x_axis)

args.training_time = tic2 - tic
args.n_params, args.n_layers = model._num_parameters_convlayers()
with open(args.run_dir + "/args.txt", 'w') as args_file:
    json.dump(vars(args), args_file, indent=4)