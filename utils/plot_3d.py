import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
import numpy as np
from .misc import to_numpy
plt.switch_backend('agg')


def plot_prediction(save_dir, target, prediction, epoch, index, 
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

            if plot_fn == 'contourf':
                cax = ax.contourf(data, 50, cmap='jet',
                                  vmin=vmin[i], vmax=vmax[i])
            elif plot_fn == 'imshow':
                cax = ax.imshow(data, cmap='jet', origin='lower',
                                interpolation='bilinear',
                                vmin=vmin[i], vmax=vmax[i])

            if plot_fn == 'contourf':
                # 修复 Matplotlib 3.8+ 兼容性问题
                if hasattr(cax, 'collections'):
                    for coll in cax.collections:
                        coll.set_edgecolor("face")
                        coll.set_linewidth(0.000000000001)
                else:
                    cax.set_edgecolor("face")
                    cax.set_linewidth(0.000000000001)

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
    rmse_train = logger['rmse_train']
    rmse_test = logger['rmse_test']
    r2_train = logger['r2_train']
    r2_test = logger['r2_test']

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


