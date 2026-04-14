#!/bin/bash
set -e

############ GENERAL ENV SETUP ############

eval "$(conda shell.bash hook)"
conda activate d3il

if [[ "$CONDA_DEFAULT_ENV" != "d3il" ]]; then
    echo "Please activate the d3il environment first."
    exit 1
fi

### Set Channel vars
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels nvidia
conda config --set channel_priority strict


############ PYTHON ############
echo Install mamba
conda install mamba -c conda-forge -y -q


############ REQUIRED DEPENDENCIES (PYBULLET) ############
echo Installing dependencies...

mamba install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y -q

mamba install -c conda-forge pybullet pyyaml scipy opencv pinocchio matplotlib gin-config -y -q

python -m pip install --upgrade "pip<24" setuptools wheel
pip install "gym==0.21.0"


# Open3D for PointClouds and its dependencies. Why does it not install them directly?
mamba install -c conda-forge scikit-learn addict pandas plyfile tqdm -y -q
mamba install -c open3d-admin open3d -y -q

pip install einops
pip install hydra-core==1.1.1
pip install wandb

# Robomimic
pip install termcolor

# ACT
pip install ipython

# BESO
pip install torchsde torchdiffeq

############ MUJOCO BETA SUPPORT INSTALLATION ############
mamba install -c conda-forge imageio -y -q
pip install mujoco==2.3.2

############ INSTALL D3il-Sim & FINALIZE ############
echo
echo Installing D3il-Sim Package
cd environments/d3il && pip install -e .

exit 0
