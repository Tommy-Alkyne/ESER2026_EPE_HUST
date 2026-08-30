"""预测结果与训练曲线绘图（2D / 3D 共用实现）。

相对旧版的修正
--------------
1. **matplotlib >= 3.8 兼容**。旧版 `for c in cax.collections` 在新版直接
   `AttributeError`（`QuadContourSet` 已无 `.collections`）。`plot_3d.py` 修过，
   `plot.py` 没同步。
2. **支持任意通道数**。旧版硬编码 3 通道（`samples[[0,3]]` / `subplots(3,3)`），
   实测 C=1 / C=2 直接 IndexError。
3. **色阶真正对齐**。`contourf` 会忽略 vmin/vmax，等值线是自动分档的，所以
   README 说的"对齐真实值极值"在 contourf 下并不成立。现在显式传 `levels`，
   两种模式都严格同色阶；残差行用对称的 diverging 色阶单独标定。
4. 训练曲线支持训练/验证长度不一致（`--post` 模式下旧版必崩）。
"""

import matplotlib
import matplotlib.ticker as ticker
import numpy as np

matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa: E402

from .misc import to_numpy  # noqa: E402

CHANNEL_LABELS = {
    'pvv': ['P', 'Vx', 'Vy'],
    'vvp': ['Vx', 'Vy', 'P'],
}


def _select_middle_depth(x):
    """(C, D, H, W) -> (C, H, W)，取中间层；已是 3D 则原样返回。"""
    if x is None or x.ndim != 4:
        return x
    d = x.shape[1]
    return x[:, 0, :, :] if d == 1 else x[:, d // 2, :, :]


def _channel_names(n_channels, channel_order='pvv'):
    names = CHANNEL_LABELS.get(str(channel_order).lower(), [])
    if len(names) >= n_channels:
        return names[:n_channels]
    return ['ch{}'.format(i) for i in range(n_channels)]


def plot_prediction(save_dir, target, prediction, epoch, index,
                    plot_fn='imshow', channel_order='pvv', mask=None,
                    file_format='pdf'):
    """三行对比图：真值 / 预测 / 残差，每列一个物理通道。

    Args:
        target / prediction: (C, H, W) 或 (C, D, H, W)
        mask: 可选 (H, W) 计算域掩码，管外置 NaN 以便留白
    """
    target = _select_middle_depth(to_numpy(target))
    prediction = _select_middle_depth(to_numpy(prediction))
    if target.ndim == 2:
        target, prediction = target[None], prediction[None]

    n_channels = target.shape[0]
    names = _channel_names(n_channels, channel_order)
    residual = target - prediction

    if mask is not None:
        mask = np.asarray(to_numpy(mask)).astype(bool)
        if mask.shape == target.shape[-2:]:
            blank = ~mask
            target = np.where(blank, np.nan, target)
            prediction = np.where(blank, np.nan, prediction)
            residual = np.where(blank, np.nan, residual)

    fig, axes = plt.subplots(3, n_channels, figsize=(4 * n_channels, 9),
                             squeeze=False)
    row_titles = ['CFD truth', 'Surrogate', 'Residual']

    for c in range(n_channels):
        # 真值与预测共用同一色阶；残差用以 0 为中心的对称色阶
        stack = np.concatenate([target[c].ravel(), prediction[c].ravel()])
        stack = stack[np.isfinite(stack)]
        vmin = float(np.min(stack)) if stack.size else 0.0
        vmax = float(np.max(stack)) if stack.size else 1.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-8
        res = residual[c][np.isfinite(residual[c])]
        rlim = float(np.max(np.abs(res))) if res.size else 1e-8
        rlim = max(rlim, 1e-12)

        for r in range(3):
            ax = axes[r][c]
            ax.set_aspect('equal')
            ax.set_axis_off()
            if r == 0:
                data, lo, hi, cmap = target[c], vmin, vmax, 'jet'
            elif r == 1:
                data, lo, hi, cmap = prediction[c], vmin, vmax, 'jet'
            else:
                data, lo, hi, cmap = residual[c], -rlim, rlim, 'coolwarm'

            if plot_fn == 'contourf':
                # 显式给 levels，否则 contourf 会忽略 vmin/vmax 各自分档
                levels = np.linspace(lo, hi, 51)
                cax = ax.contourf(np.nan_to_num(data, nan=lo), levels=levels,
                                  cmap=cmap, extend='both')
                # matplotlib >= 3.8 的 QuadContourSet 不再有 .collections
                collections = getattr(cax, 'collections', None)
                targets = collections if collections is not None else [cax]
                for coll in targets:
                    if hasattr(coll, 'set_edgecolor'):
                        coll.set_edgecolor('face')
                        coll.set_linewidth(1e-12)
            else:
                cax = ax.imshow(data, cmap=cmap, origin='lower',
                                interpolation='nearest', vmin=lo, vmax=hi)

            if r == 0:
                ax.set_title(names[c], fontsize=11)
            if c == 0:
                ax.text(-0.08, 0.5, row_titles[r], transform=ax.transAxes,
                        rotation=90, va='center', ha='center', fontsize=9)

            cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04,
                                format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.update_ticks()

    plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.3)
    out_path = '{}/pred_epoch{}_{}.{}'.format(save_dir, epoch, index, file_format)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _curve(save_dir, name, ylabel, series, legend_loc='best'):
    """画一条/多条曲线；各序列长度可以不同（--post 模式下就是这种情况）。"""
    drawn = False
    plt.figure()
    for label, values in series:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            continue
        x = np.arange(1, values.size + 1)
        tail = np.mean(values[-5:])
        plt.plot(x, values, label='{}: {:.4f}'.format(label, tail))
        drawn = True
    if not drawn:
        plt.close()
        return
    plt.xlabel('logged step (every log_freq epochs)')
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)
    plt.savefig('{}/{}.pdf'.format(save_dir, name), dpi=300, bbox_inches='tight')
    plt.close()


def save_stats(save_dir, logger, x_axis=None):
    """保存 R² / RMSE 曲线与原始数值。

    `x_axis` 保留只为兼容旧调用，实际横轴按各序列自身长度生成 —— 旧版用
    `np.arange(log_freq, epochs+log_freq, log_freq)` 固定长度，`--post` 模式下
    训练序列为空，`plt.plot` 长度不匹配必崩。
    """
    del x_axis

    _curve(save_dir, 'r2', r'$R^2$-score',
           [('Train', logger.get('r2_train', [])),
            ('Test', logger.get('r2_test', []))], legend_loc='lower right')
    _curve(save_dir, 'rmse', 'RMSE',
           [('Train', logger.get('rmse_train', [])),
            ('Test', logger.get('rmse_test', []))], legend_loc='upper right')

    for key in ('div_rms', 'ns_rms'):
        if logger.get(key):
            _curve(save_dir, key, key.replace('_', ' ').upper(),
                   [(key, logger[key])])

    for key, values in logger.items():
        values = np.asarray(values, dtype=float).ravel()
        if values.size:
            np.savetxt('{}/{}.txt'.format(save_dir, key), values)

    # 分通道 R² 单独存一份，便于看压力/速度哪一个拖后腿
    per_channel = logger.get('r2_test_per_channel')
    if per_channel:
        arr = np.asarray(per_channel, dtype=float)
        np.savetxt('{}/r2_test_per_channel.txt'.format(save_dir), arr)
        plt.figure()
        for c in range(arr.shape[1]):
            plt.plot(np.arange(1, arr.shape[0] + 1), arr[:, c],
                     label='ch{}: {:.4f}'.format(c, arr[-1, c]))
        plt.xlabel('logged step')
        plt.ylabel(r'$R^2$-score (per channel)')
        plt.legend(loc='lower right')
        plt.savefig('{}/r2_per_channel.pdf'.format(save_dir), dpi=300,
                    bbox_inches='tight')
        plt.close()
