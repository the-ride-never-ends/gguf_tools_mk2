# gguf_tools_mk2

## Overview
A suite of tools to manipulate GGUF, Torch, and other transformers model types. 

## Key Features
- gguf_tensor_to_image: Generate false-color visualizations of tensor weight values and derivative statistics from transformers models. This can be used to visualize various descriptive statistics about a model's tensor, identify potential issues with training, and identify outliers or patterns. This tool supports both GGUF and Torch model types.

- image_diff_heatmapper_mk2: Generate false-color heatmaps showing differences in weight values in equivalent tensors between two similar transformers models. This can be used to visualize the differences between base models and fine-tunes, models trained with different hyperparameters, or between models trained on different datasets.


## Dependencies
- pandas
- pymysql
- pyyaml
- mysql-connector-python
- aiomysql
- multipledispatch

## Usage
Click on "start.bat" to run the program. Follow the on-screen instructions to select the tool you want to use and provide the necessary inputs.