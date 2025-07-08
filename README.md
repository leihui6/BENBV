# BENBV (Boundary Exploration Next Best View)

This is the official repository for the Boundary Exploration of Next Best View Policy in 3D Robotic Scanning.

<p align="center">
<img src="./figures/Animation.gif" width="60%">
<div> </div>
</p>

## Environment

The conda is recommended using to create a virtual environment as:

``` shell
conda create -n nbv python=3.8 ipython
conda activate nbv
```

First, the Pytorch is used in `nbv_explore_net` such that you can install Pytorch as:

``` shell
(depends on the CUDA you installed)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

The cpu version for Pytorch is also available if you just want to run the code in *dataset_generation*.

Second, the other packages are required to install as:

``` shell
pip install -r requirements-no-pytorch.txt
```

The version of `Pytorch 2.3.1` and `Pytorch 1.8` has been tested.

## Usage

### 1. Dataset Generation (also BENBV included)

- Jump to `./dataset_generation` and run `python main.py`
- Stay and run `python show_views.py` where you can select the `next-best-view` visualization.

### 2. BENBV-Net

- Jump to `./nbv_explore_net` and run `python train.py`
