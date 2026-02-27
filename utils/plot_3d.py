import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
import numpy as np
from .misc import to_numpy
plt.switch_backend('agg')


def plot_prediction_det(save_dir, target, prediction, epoch, index, 
                        plot_fn='contourf'):
    """Plot prediction for one input (`index`-th at epoch `epoch`)
    Args:
        save_dir: directory to save predictions
        target (np.ndarray): (3, 65, 65)
        prediction (np.ndarray): (3, 65, 65)
        epoch (int): which epoch
        index (int): i-th prediction
        plot_fn (str): choices=['contourf', 'imshow']
    """
    target, prediction = to_numpy(target), to_numpy(prediction)

    # If data has shape (C, D, H, W), select the middle depth slice.
    # Odd D: middle; Even D: upper of the two middle slices (index D//2).
    def _select_middle_depth(x):
        if x is None:
            return x
        if x.ndim == 4:
            C, D, H, W = x.shape
            if D == 1:
                return x[:, 0, :, :]
            idx = D // 2
            return x[:, idx, :, :]
        return x

    target = _select_middle_depth(target)
    prediction = _select_middle_depth(prediction)

    c = target.shape[0]
    vmin = []
    vmax = []
    for i in range(c):
        channel = np.concatenate((target[i], prediction[i]), axis=None)
        vmin.append(np.amin(channel))
        vmax.append(np.amax(channel))

    fig, axes = plt.subplots(3, c, figsize=(4 * c, 9))
    if c == 1:
        axes = np.array(axes).reshape(3, 1)

    for i in range(c):
        for r in range(3):
            ax = axes[r, i]
            ax.set_aspect('equal')
            ax.set_axis_off()
            if r == 0:
                data = target[i]
            elif r == 1:
                data = prediction[i]
            else:
                data = target[i] - prediction[i]

            if r < 2:
                if plot_fn == 'contourf':
                    cax = ax.contourf(data, 50, cmap='jet',
                                      vmin=vmin[i], vmax=vmax[i])
                elif plot_fn == 'imshow':
                    cax = ax.imshow(data, cmap='jet', origin='lower',
                                    interpolation='bilinear',
                                    vmin=vmin[i], vmax=vmax[i])
            else:
                if plot_fn == 'contourf':
                    cax = ax.contourf(data, 50, cmap='jet')
                elif plot_fn == 'imshow':
                    cax = ax.imshow(data, cmap='jet', origin='lower',
                                    interpolation='bilinear')

            if plot_fn == 'contourf':
                for coll in cax.collections:
                    coll.set_edgecolor("face")
                    coll.set_linewidth(0.000000000001)
            cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                                format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.update_ticks()
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    plt.savefig(save_dir + '/pred_epoch{}_{}.pdf'.format(epoch, index),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_stats(save_dir, logger, x_axis):

    rmse_train = logger.get('rmse_train', [])
    rmse_test = logger.get('rmse_test', [])
    r2_train = logger.get('r2_train', [])
    r2_test = logger.get('r2_test', [])
    loss_train = logger.get('loss_train', [])
    loss_test = logger.get('loss_test', [])
    acc_train = logger.get('acc_train', [])
    acc_test = logger.get('acc_test', [])

    if 'mnlp_test' in logger.keys():
        mnlp_test = logger['mnlp_test']
        if len(mnlp_test) > 0:
            plt.figure()
            plt.plot(x_axis, mnlp_test, label="Test: {:.3f}".format(np.mean(mnlp_test[-5:])))
            plt.xlabel('Epoch')
            plt.ylabel('MNLP')
            plt.legend(loc='upper right')
            plt.savefig(save_dir + "/mnlp_test.pdf", dpi=600)
            plt.close()
            np.savetxt(save_dir + "/mnlp_test.txt", mnlp_test)
    
    if 'log_beta' in logger.keys():
        log_beta = logger['log_beta']
        if len(log_beta) > 0:
            plt.figure()
            plt.plot(x_axis, log_beta, label="Test: {:.3f}".format(np.mean(log_beta[-5:])))
            plt.xlabel('Epoch')
            plt.ylabel('Log-Beta (noise precision)')
            plt.legend(loc='upper right')
            plt.savefig(save_dir + "/log_beta.pdf", dpi=600)
            plt.close()
            np.savetxt(save_dir + "/log_beta.txt", log_beta)

    if len(r2_train) > 0 and len(r2_test) > 0:
        plt.figure()
        plt.plot(x_axis, r2_train, label="Train: {:.3f}".format(np.mean(r2_train[-5:])))
        plt.plot(x_axis, r2_test, label="Test: {:.3f}".format(np.mean(r2_test[-5:])))
        plt.xlabel('Epoch')
        plt.ylabel(r'$R^2$-score')
        plt.legend(loc='lower right')
        plt.savefig(save_dir + "/r2.pdf", dpi=600)
        plt.close()
        np.savetxt(save_dir + "/r2_train.txt", r2_train)
        np.savetxt(save_dir + "/r2_test.txt", r2_test)

    if len(rmse_train) > 0 and len(rmse_test) > 0:
        plt.figure()
        plt.plot(x_axis, rmse_train, label="train: {:.3f}".format(np.mean(rmse_train[-5:])))
        plt.plot(x_axis, rmse_test, label="test: {:.3f}".format(np.mean(rmse_test[-5:])))
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend(loc='upper right')
        plt.savefig(save_dir + "/rmse.pdf", dpi=600)
        plt.close()
        np.savetxt(save_dir + "/rmse_train.txt", rmse_train)
        np.savetxt(save_dir + "/rmse_test.txt", rmse_test)

    if len(acc_train) > 0 and len(acc_test) > 0:
        plt.figure()
        plt.plot(x_axis, np.array(acc_train) * 100, label="Train: {:.2f}%".format(np.mean(acc_train[-5:]) * 100))
        plt.plot(x_axis, np.array(acc_test) * 100, label="Test: {:.2f}%".format(np.mean(acc_test[-5:]) * 100))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='lower right')
        plt.savefig(save_dir + "/accuracy.pdf", dpi=600)
        plt.close()
        np.savetxt(save_dir + "/acc_train.txt", acc_train)
        np.savetxt(save_dir + "/acc_test.txt", acc_test)

    if len(loss_train) > 0 and len(loss_test) > 0:
        plt.figure()
        plt.plot(x_axis, loss_train, label="Train: {:.4f}".format(np.mean(loss_train[-5:])))
        plt.plot(x_axis, loss_test, label="Test: {:.4f}".format(np.mean(loss_test[-5:])))
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.legend(loc='upper right')
        plt.savefig(save_dir + "/loss.pdf", dpi=600)
        plt.close()
        np.savetxt(save_dir + "/loss_train.txt", loss_train)
        np.savetxt(save_dir + "/loss_test.txt", loss_test)



def plot_prediction_bayes(save_dir, target, pred_mean, pred_var, epoch, index, 
        plot_fn='contourf'):
    """Plot predictions at *one* test input

    Args:
        save_dir: directory to save predictions
        target (np.ndarray or torch.Tensor): (3, 65, 65)
        pred_mean (np.ndarray or torch.Tensor): (3, 65, 65)
        pred_var (np.ndarray or torch.Tensor): (3, 65, 65)
        epoch (int): which epoch
        index (int): i-th prediction
        plot_fn (str): choices=['contourf', 'imshow']
    """
    target, pred_mean, pred_var = to_numpy(target), to_numpy(pred_mean), to_numpy(pred_var)

    def _select_middle_depth(x):
        if x is None:
            return x
        if x.ndim == 4:
            C, D, H, W = x.shape
            if D == 1:
                return x[:, 0, :, :]
            idx = D // 2
            return x[:, idx, :, :]
        return x

    target = _select_middle_depth(target)
    pred_mean = _select_middle_depth(pred_mean)
    pred_var = _select_middle_depth(pred_var)

    pred_error = target - pred_mean    
    two_sigma = np.sqrt(pred_var) * 2
    # target: C x H x W
    sfmt = ticker.ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 2))
    cmap = 'jet'
    interpolation = 'bilinear'
    fig = plt.figure(1, (11, 12))
    axes_pad = 0.25
    cbar_pad = 0.1
    label_size = 6

    subplots_position = ['23{}'.format(i) for i in range(1, 7)]

    for i, subplot_i in enumerate(subplots_position):
        if i < 3:
            # share one colorbar
            grid = ImageGrid(fig, subplot_i,          # as in plt.subplot(111)
                             nrows_ncols=(2, 1),
                             axes_pad=axes_pad,
                             share_all=False,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="3%",
                             cbar_pad=cbar_pad,
                             )
            data = (target[i], pred_mean[i])
            channel = np.concatenate(data)
            vmin, vmax = np.amin(channel), np.amax(channel)
            # Add data to image grid
            for j, ax in enumerate(grid):
                if plot_fn == 'contourf':
                    im = ax.contourf(data[j], 50, vmin=vmin, vmax=vmax, cmap=cmap)
                    for c in im.collections:
                        c.set_edgecolor("face")
                        c.set_linewidth(0.000000000001)
                elif plot_fn == 'imshow':
                    im = ax.imshow(data[j], vmin=vmin, vmax=vmax,
                        interpolation=interpolation, cmap=cmap)
                ax.set_axis_off()
            # ticks=np.linspace(vmin, vmax, 10)
            #set_ticks, set_ticklabels
            cbar = grid.cbar_axes[0].colorbar(im, format=sfmt)
            # cbar.ax.set_yticks((vmin, vmax))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.ax.tick_params(labelsize=label_size)
            cbar.ax.toggle_label(True)

        else:
            grid = ImageGrid(fig, subplot_i,  # as in plt.subplot(111)
                             nrows_ncols=(2, 1),
                             axes_pad=axes_pad,
                             share_all=False,
                             cbar_location="right",
                             cbar_mode="each",
                             cbar_size="6%",
                             cbar_pad=cbar_pad,
                             )
            data = (pred_error[i-3], two_sigma[i-3])
            # channel = np.concatenate(data)
            # vmin, vmax = np.amin(channel), np.amax(channel)
            # Add data to image grid
            for j, ax in enumerate(grid):
                if plot_fn == 'contourf':
                    im = ax.contourf(data[j], 50, cmap=cmap)
                    for c in im.collections:
                        c.set_edgecolor("face")
                        c.set_linewidth(0.000000000001)
                elif plot_fn == 'imshow':
                    im = ax.imshow(data[j], interpolation=interpolation, cmap=cmap)
                ax.set_axis_off()
                cbar = grid.cbar_axes[j].colorbar(im, format=sfmt)
                grid.cbar_axes[j].tick_params(labelsize=label_size)
                grid.cbar_axes[j].toggle_label(True)
                # cbar.formatter.set_powerlimits((0, 0))
                cbar.ax.yaxis.set_offset_position('left')
                # print(dir(cbar.ax.yaxis))
                # cbar.update_ticks()

    # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    fig.subplots_adjust(wspace=0.075, hspace=0.075)

    plt.savefig(save_dir + '/pred_at_x_epoch{}_{}.pdf'.format(epoch, index), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_MC(save_dir, monte_carlo, pred_mean, pred_var, mean, n_train):
    """Plot Monte Carlo Output
    
    Args:
        monte_carlo (np.ndarray or torch.Tensor): simulation output
        pred_mean (np.ndarray or torch.Tensor): from surrogate
        pred_var (np.ndarray or torch.Tensor): predictive var using surrogate
        mean (bool): Used in printing. True for plotting mean, False for var
    """
    monte_carlo, pred_mean, pred_var = to_numpy(monte_carlo), \
                                       to_numpy(pred_mean), to_numpy(pred_var)

    def _select_middle_depth(x):
        if x is None:
            return x
        if x.ndim == 4:
            C, D, H, W = x.shape
            if D == 1:
                return x[:, 0, :, :]
            idx = D // 2
            return x[:, idx, :, :]
        return x

    monte_carlo = _select_middle_depth(monte_carlo)
    pred_mean = _select_middle_depth(pred_mean)
    pred_var = _select_middle_depth(pred_var)

    two_sigma = 2 * np.sqrt(pred_var)
    # target: C x H x W
    sfmt = ticker.ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((0, 0))
    cmap = 'jet'
    interpolation = 'bilinear'
    pred_error = monte_carlo - pred_mean
    fig = plt.figure(1, (10, 10))
    axes_pad = 0.25
    cbar_pad = 0.1
    label_size = 6

    subplots_position = ['23{}'.format(i) for i in range(1, 7)]

    for i, subplot_i in enumerate(subplots_position):
        if i < 3:
            # share one colorbar
            grid = ImageGrid(fig, subplot_i,          # as in plt.subplot(111)
                             nrows_ncols=(2, 1),
                             axes_pad=axes_pad,
                             share_all=False,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="3%",
                             cbar_pad=cbar_pad,
                             )
            data = (monte_carlo[i], pred_mean[i])
            channel = np.concatenate(data)
            vmin, vmax = np.amin(channel), np.amax(channel)
            # Add data to image grid
            for j, ax in enumerate(grid):
                # im = ax.imshow(data[j], vmin=vmin, vmax=vmax,
                #                interpolation=interpolation, cmap=cmap)
                im = ax.contourf(data[j], 50, vmin=vmin, vmax=vmax, cmap=cmap)
                for c in im.collections:
                    c.set_edgecolor("face")
                    c.set_linewidth(0.000000000001)
                ax.set_axis_off()
            # ticks=np.linspace(vmin, vmax, 10)
            #set_ticks, set_ticklabels
            cbar = grid.cbar_axes[0].colorbar(im, format=sfmt)
            # cbar.ax.set_yticks((vmin, vmax))
            cbar.ax.tick_params(labelsize=label_size)
            cbar.ax.yaxis.set_offset_position('left')
            cbar.ax.toggle_label(True)

        else:
            grid = ImageGrid(fig, subplot_i,  # as in plt.subplot(111)
                             nrows_ncols=(2, 1),
                             axes_pad=axes_pad,
                             share_all=False,
                             cbar_location="right",
                             cbar_mode="each",
                             cbar_size="6%",
                             cbar_pad=cbar_pad,
                             )
            data = (pred_error[i-3], two_sigma[i-3])
            # channel = np.concatenate(data)
            # vmin, vmax = np.amin(channel), np.amax(channel)
            # Add data to image grid
            for j, ax in enumerate(grid):
                # im = ax.imshow(data[j], interpolation=interpolation, cmap=cmap)
                im = ax.contourf(data[j], 50, cmap=cmap)
                for c in im.collections:
                    c.set_edgecolor("face")
                    c.set_linewidth(0.000000000001)
                ax.set_axis_off()
                cbar = grid.cbar_axes[j].colorbar(im, format=sfmt)
                grid.cbar_axes[j].tick_params(labelsize=label_size)
                grid.cbar_axes[j].toggle_label(True)
                # cbar.formatter.set_powerlimits((0, 0))
                cbar.ax.yaxis.set_offset_position('left')
                # print(dir(cbar.ax.yaxis))
                # cbar.update_ticks()

    # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    fig.subplots_adjust(wspace=0.075, hspace=0.075)

    plt.savefig(save_dir + '/pred_{}_vs_MC.pdf'.format('mean' if mean else 'var'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Done plotting Pred_{}_vs_MC, num of training: {}"
          .format('mean' if mean else 'var', n_train))
