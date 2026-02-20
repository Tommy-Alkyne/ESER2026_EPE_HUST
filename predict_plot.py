import h5py
import numpy as np
import matplotlib.pyplot as plt

pred_path = 'dataset/prediction/skiplayer.hdf5'
data_path = '/home/tommy/code/ESER2026_EPE_HUST/dataset/p_c10e6_128.hdf5'

sample_idx = 20
channel_idx = 1
depth_idx = 0

with h5py.File(pred_path, 'r') as pre:
    pred = pre['predictions'][sample_idx, channel_idx, depth_idx]

with h5py.File(data_path, 'r') as f:
    target = f['output'][sample_idx, channel_idx, depth_idx]

err = pred - target
mae = np.mean(np.abs(err))
rmse = np.sqrt(np.mean(err ** 2))

print('pred shape:', pred.shape, 'target shape:', target.shape)
print('mae:', mae, 'rmse:', rmse)

vmin = min(target.min(), pred.min())
vmax = max(target.max(), pred.max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].imshow(target, cmap='jet', vmin=vmin, vmax=vmax)
axes[0].set_title('Target')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(pred, cmap='jet', vmin=vmin, vmax=vmax)
axes[1].set_title('Prediction')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(err, cmap='seismic')
axes[2].set_title('Error (Pred - Target)')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()