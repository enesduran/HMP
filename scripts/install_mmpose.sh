#!/bin/bash
set -e

export CONDA_ENV_NAME=hmp
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

# export CUDA_HOME=/usr/local/cuda-11.2
echo "Installing necessary packages: mmengine, mmdet, mmcv"
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"

echo "Cloning mmpose into external/mmpose"
cd external
git clone https://github.com/open-mmlab/mmpose.git

echo "Installing mmpose"
cd mmpose
pip install -r requirements.txt
pip install -v -e .
