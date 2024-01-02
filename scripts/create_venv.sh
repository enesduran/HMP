#!/bin/bash
set -e

export CONDA_ENV_NAME="hmp"
echo "Creating virtual environment $CONDA_ENV_NAME"

conda create --name $CONDA_ENV_NAME python=3.10 --force -y 

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME


conda install pytorch-gpu==2.1.0 torchvision==0.15.2 -c conda-forge -y 

pip install -r requirements.txt

# install mmpose 
source scripts/install_mmpose.sh

