import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

# initialize model
model = DenseED(in_channels=getattr(args, 'nic', 1), out_channels=getattr(args, 'noc', 3), 
                blocks=args.blocks,
                growth_rate=args.growth_rate, 
                init_features=args.init_features,
                drop_rate=args.drop_rate,
                bn_size=args.bn_size,
                bottleneck=args.bottleneck,
                out_activation=None,
                output_size=tuple(getattr(args, 'image_size', [1, 65, 65])) if getattr(args, 'ndim', 2) == 3 else None).to(device)
print(model)

train_loader, _ = load_data(train_data_dir, args.batch_size)
test_loader, _ = load_data(test_data_dir, args.test_batch_size)
print('Loaded data!')

optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                    #verbose=True, 
                    threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8)

logger = {}
logger['rmse_train'] = []
logger['rmse_test'] = []
logger['r2_train'] = []
logger['r2_test'] = []
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

def test(epoch):#epoch is a number
    model.eval()
    mse = 0.
    n_out_pixels_test = 0
    masked_sum_y = 0.
    masked_sum_y2 = 0.
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        output = model(input)
        t_mask = build_t_mask(output)
        t_mask = t_mask.expand(output.shape[0], output.shape[1], *output.shape[2:])
        masked_output = output * t_mask
        masked_target = target * t_mask
        mse += F.mse_loss(masked_output, masked_target, reduction='sum').item()
        n_out_pixels_test += t_mask.sum().item()
        masked_sum_y += masked_target.sum().item()
        masked_sum_y2 += (masked_target ** 2).sum().item()

        # plot predictions
        if epoch % args.plot_freq == 0 and batch_idx == 0:
            n_samples = 6 if epoch == args.epochs else 2
            idx = torch.randperm(input.size(0))[:n_samples]
            samples_output = masked_output.data.cpu()[idx].numpy()
            samples_target = masked_target.data.cpu()[idx].numpy()

            for i in range(n_samples):
                print('epoch {}: plotting prediction {}'.format(epoch, i))
                plot_prediction_det(args.pred_dir, samples_target[i], 
                    samples_output[i], epoch, i, plot_fn=args.plot_fn)

    rmse_test = np.sqrt(mse / max(n_out_pixels_test, 1.0))
    ss_tot = masked_sum_y2 - (masked_sum_y ** 2) / max(n_out_pixels_test, 1.0)
    r2_score = 1 - mse / max(ss_tot, 1e-12)
    print("epoch: {}, test r2-score:  {:.6f}".format(epoch, r2_score))

    if epoch % args.log_freq == 0:
        logger['r2_test'].append(r2_score)
        logger['rmse_test'].append(rmse_test)

# Import the PINN combined loss from utils.physics_loss
from utils.physics_loss import combined_physics_loss

def train(epoch):
    model.train()
    mse = 0.
    n_out_pixels_train = 0
    masked_sum_y = 0.
    masked_sum_y2 = 0.
    for _, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        model.zero_grad()
        output = model(input)
        t_mask = build_t_mask(output)
        t_mask = t_mask.expand(output.shape[0], output.shape[1], *output.shape[2:])
        
        # Apply combined physics loss instead of pure L1 loss
        loss, loss_dict = combined_physics_loss(
            output, target, 
            mask=t_mask, 
            epoch=epoch, 
            total_epochs=args.epochs,
            div_weight=0.1,
            ns_weight=0.01,
            smooth_weight=0.01
        )
        
        loss.backward()
        optimizer.step()
        
        masked_output = output * t_mask
        masked_target = target * t_mask
        mse += F.mse_loss(masked_output, masked_target, reduction='sum').item()
        n_out_pixels_train += t_mask.sum().item()
        masked_sum_y += masked_target.sum().item()
        masked_sum_y2 += (masked_target ** 2).sum().item()

    rmse = np.sqrt(mse / max(n_out_pixels_train, 1.0))
    scheduler.step(rmse)
    ss_tot = masked_sum_y2 - (masked_sum_y ** 2) / max(n_out_pixels_train, 1.0)
    r2_score = 1 - mse / max(ss_tot, 1e-12)
    print("epoch: {}, training r2: {:.6f}, base_loss: {:.4f}, div_loss: {:.6f}, ns_loss: {:.6f}".format(
        epoch, r2_score, loss_dict['base'], loss_dict['div'], loss_dict['ns']))
    if epoch % args.log_freq == 0:
        logger['r2_train'].append(r2_score)
        logger['rmse_train'].append(rmse)
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