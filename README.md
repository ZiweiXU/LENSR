# Embedding Symbolic Knowledge into Deep Networks

This repository accompanies the paper "Embedding Symbolic Knowledge into Deep Networks" (NeurIPS 2019). 

To replicate the experiments in the paper, clone and run with the NeurIPS2019 tag: 

`git clone -b NeurIPS2019 https://github.com/ZiweiXU/LENSR .`

We have modified the code since then. 

## Required libraries
1. Install Anaconda from [https://www.anaconda.com/]
2. Install all required python packages: `conda env create -f environment.yml`

## For synthetic dataset
1. Change directory to scripts using `cd scripts`
2. Run `train_synthetic.sh` to preprocess the data and train the model.

## For VRD dataset
1. Change directory to scripts using `cd scripts`
2. Run `train_vrd.sh` to download and preprocess the data, and train the model.
