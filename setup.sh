#!/usr/bin/env bash

# Create environment
conda create -y -n 00_boot python=3.13

# Activate environment
conda activate 00_boot

# Install dependencies
conda install -y matplotlib jupyter tqdm tifffile numpy scikit-image torchvision

# Deactivate environment
conda deactivate