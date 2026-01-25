# README 中文更新

## Prediction 预测
用来指定模型.pth文件和数据.hdf5文件来预测的脚本. 使用的是`surrogate_predict.py`
### 参数
--ckpt-file 使用的模型pth文件的路径 例: `--ckpt-file './experiments/2dtest_1024/kle4225/ntrain1024_blocks[3, 6, 3]_growth16_nif48_drop0.0_batch16_lr0.003_wd0.0005_epochs150/checkpoints/model_epoch100.pth'` 注意, 如果路径中有中括号`[`等特殊字符需要用单引号`'`把文件路径括起来. 会有中括号是因为默认的保存命名方式如此

--data-file 用来预测的hd5文件的路径 例: `--data-file './dataset/kle4225_lhs1024.hdf5'` 

--ndim 选择3D还是2D, 如果留空会自动检测 例: `--ndim 2`

--out-file 指定的输出文件的路径 例: `--out-file ./dataset/prediction/2d_epoch100_1024.hdf5` 如果dataset文件夹里没有prediction文件夹需要提前创建

其他的参数可以通过`python3 surrogate_predict.py --help`查看
### 例子

```bash
python3 surrogate_predict.py --ckpt-file './experiments/2dtest_1024/kle4225/ntrain1024_blocks[3, 6, 3]_growth16_nif48_drop0.0_batch16_lr0.003_wd0.0005_epochs150/checkpoints/model_epoch100.pth' --data-file './dataset/kle4225_lhs1024.hdf5' --ndim 2 --out-file ./dataset/prediction/2d_epoch100_1024.hdf5
```
## Train 训练
训练用来生成.pth的模型文件. 使用的是`train_det.py`和`train_svgd.py`
### 参数
--exp-name 训练实验的名字 例: `--exp-name 2dtest_1024` 会在experiments文件夹下生成一个2dtest_1024的文件夹用来保存训练结果

--kle 选择KLE50, KLE500, KLE4225数据集 例: `--kle 4225` 对于作者给出的数据集, 只有4225. 后面可能会改命名规范, 可能会改

--ntrain 选择训练数据的数量 例: `--ntrain 1024` 就是hdf5里4225后面那个数

--ndim 选择3D还是2D 如果留空默认2D. 例: `--ndim 3`

--epochs 训练的轮数 例: `--epochs 150`

--ckpt-freq 多久保存一次模型 例: `--ckpt-freq 50` 就是每50轮保存一次

--plot-freq 多久保存一次画预测图的曲线 例: `--plot-freq 50` 就是每50轮保存一次

--train-file 指定训练数据的hdf5文件路径 例: `--train-file './dataset/10243d.hdf5'` **注意: 如果使用此参数会覆盖--kle 和--ntrain 对训练数据的选择. 如果不使用此参数则默认选择--kle 和--ntrain所指定的数据**

--test-file 指定测试数据的hdf5文件路径 例: `--test-file './dataset/10243d.hdf5'` **注意: 如果使用此参数会覆盖默认的测试数据. 如果不使用此参数则默认选择--kle 和--ntrain所指定的数据**

其他的参数可以通过`python3 train_det.py --help`查看

### 例子
**2D情况**
```bash
 python3 train_det.py --exp-name 2dtest_1024 --epochs 150 --ckpt-freq 50 --plot-freq 50 --ntrain 1024
```
这个例子表示使用1024个训练数据 (选择kle4225_lhs1024.hdf5 作为训练数据), 训练150轮, 每50轮保存一次模型和画图. 结果会保存在`./experiments/2dtest_1024/`文件夹下. 没有指定--kle和--ndim, 默认是4225和2D

**3D例子**
```bash
 python3 train_det.py --exp-name 3dtest_1024 --ndim 3 --epochs 150 --ckpt-freq 50 --plot-freq 50 --train-file './dataset/10243d.hdf5' --test-file './dataset/10243d.hdf5'
```
这个例子表示使用1024个训练数据 (选择`10243d.hdf5` 作为训练数据), 训练150轮, 每50轮保存一次模型和画图. 结果会保存在`./experiments/3dtest_1024/`文件夹下

# 原README
KLE4225 (No dim reduction) | KLE500 | KLE50
:-----:|:------:|:-----:
![](images/kle4225_pred_at_x_312_n512.png?raw=true) | ![](images/kle500_pred_at_x_293_n256.png?raw=true) | ![](images/kle50_pred_at_x_47_n128.png?raw=true)

The three columns in each image correspond to three output fields:
(1) pressure field, (2)(3) velocities field in x and y axies.
The four rows in each image from top to bottom show: (1) simulator output, 
(2) surrogate prediction (predictive mean), (3) error between simulator and prediction, 
(4) two standard devivation of prediction. 

## Dependencies
- python 3 or 2
- PyTorch 0.4
- h5py
- matplotlib
- seaborn


## Installation
- Install PyTorch and other dependencies

- Clone this repo:
```
git clone https://github.com/cics-nd/cnn-surrogate.git
cd cnn-surrogate
```

## Dataset
Download Darcy flow simulation dataset with input Gaussian random field (with 
exponential kernel) and three output fields (pressure and velocities fields).

Download KLE4225 dataset:
```bash
bash ./scripts/download_dataset.sh 4225
```
Change 4225 to 500, 50 to download KLE500, KLE50 datasets. 
Data is saved at `./dataset/`.

Number of training data within each dataset:
* KLE50: [32, 64, 128, 256],
* KLE500: [64, 128, 256, 512],
* KLE4225: [128, 256, 512, 1024].

Number of test data: 500.
Number of Monte Carlo sampling data: 10,000.

KLE4225 | KLE500 | KLE50
:-----:|:------:|:-----:
![](images/kle4225_data_312.png?raw=true) | ![](images/kle500_data_293.png?raw=true) | ![](images/kle50_data_47.png?raw=true)


## Start with a Pre-trained Model

Download a pre-trained model:
```bash
bash ./scripts/download_checkpoints.sh 4225 256
```
where 4225 and 256 refer to the intrinsic input dimensionality of dataset and
number of training data. The pre-trained model is downloaded to the same location (`./experiments/Bayesian/*/`) as if it was trained on local machines. You may proceed to use the pre-trained model for UQ tasks.

## Uncertainty Quantification (UQ)
Perform UQ tasks, including:

- Prediction at one input realization
- Uncertainty propagation
- Distribution estimate
- Uncertainty quality assessment (realiability diagram)
```
python post_proc.py --post --kle 4225 --ntrain 256
```
Do not forget to add the `--post` switch. For KLE4225 with 128 data, `--epochs` should be set to 400 to show convergence.


## Training

### Deterministic Surrogate 

Train a non-Bayesian surroagte (e.g. with 256 data from KLE4225 dataset):
```
python train_det.py --kle 4225 --ntrain 256
```
The runs are saved at `./experiments/deterministic`.

### Bayesian Surrogate
Train a Bayesian surrogate (e.g. with 256 data from KLE4225 dataset
using SVGD with 20 samples):
```
python train_svgd.py --kle 4225 --ntrain 256 --n-samples 20
```
The runs are saved at `./experiments/Bayesian`.
It takes 1-2 hours to train a Bayesian surrogate with 256-512 training data 
(20 samples, 300 epochs, batch-size 16) on a single GPU.

## Citation

If you find this repo useful for your research, please consider to cite:
```latex
@article{ZHU2018415,
title = "Bayesian deep convolutional encoder–decoder networks for surrogate modeling and uncertainty quantification",
journal = "Journal of Computational Physics",
volume = "366",
pages = "415 - 447",
year = "2018",
issn = "0021-9991",
doi = "https://doi.org/10.1016/j.jcp.2018.04.018",
url = "http://www.sciencedirect.com/science/article/pii/S0021999118302341",
author = "Yinhao Zhu and Nicholas Zabaras"
}
```

## Acknowledgments

Thanks Dr. [Steven Atkinson](https://scholar.google.com/citations?user=MVrxtDkAAAAJ&hl=en) for providing Darcy flow simulation dataset.

Code is inspired by [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
