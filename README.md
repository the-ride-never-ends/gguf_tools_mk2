# gguf_tools_mk2

## Overview
A suite of tools to manipulate GGUF, Torch, and other transformers model types. 
This suite is a fork of KerfuffleV2's gguf-tools. You can find it here: https://github.com/KerfuffleV2/gguf-tools

## Key Features
- gguf_tensor_to_image: Generate false-color visualizations of tensor weight values and derivative statistics from transformers models. This can be used to visualize various descriptive statistics about a model's tensor, identify potential issues with training, and identify outliers or patterns. This tool supports both GGUF and Torch model types.

- image_diff_heatmapper_mk2: Generate false-color heatmaps showing differences in weight values in equivalent tensors between two similar transformers models. This can be used to visualize the differences between base models and fine-tunes, models trained with different hyperparameters, or between models trained on different datasets.

## Dependencies
- aiomysql
- multipledispatch
- mysql-connector-python
- pandas
- pymysql
- pyyaml
- gguf>=0.5
- numpy>=1.26
- pillow
- matplotlib
- rasterio
- scipy
- numpy
- pandas
- openpyxl
- torch

## Usage
Click on "start.bat" to run the program. Follow the on-screen instructions to select the tool you want to use and provide the necessary inputs.