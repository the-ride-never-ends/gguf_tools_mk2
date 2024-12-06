# gguf_tools_mk2

## Overview
A suite of tools to manipulate GGUF, Torch, and other transformers model types. 
This suite is a fork of KerfuffleV2's gguf-tools. You can find it here: https://github.com/KerfuffleV2/gguf-tools

## Key Features
### gguf_tensor_to_image
- description: Generate false-color visualizations of tensor weight values and derivative statistics from transformers models. This can be used to visualize various descriptive statistics about a model's tensor, identify potential issues with training, and identify outliers or patterns. This tool supports both GGUF and Torch model types.


### image_diff_heatmapper_mk2
- description: Generate false-color heatmaps showing differences in weight values in equivalent tensors between two similar transformers models. This can be used to visualize the differences between base models and fine-tunes, models trained with different hyperparameters, or between models trained on different datasets.

| Argument Name    | Type | Choices | Default | Required | Help Text |
|------------------|------|---------|---------|----------|-----------|
| model_file1      | str  | - | - | Yes | Filename for the first model, can be GGUF or PyTorch (if PyTorch support available) |
| model_file2      | str  | - | - | No | Filename for the second model, can be GGUF or PyTorch (if PyTorch support available) |
| tensor_name      | str  | - | - | No | Tensor name, must be from models with the same foundation architecture for the differences to be valid. |
| comparison_type  | str  | mean, median, absolute | mean | No | Comparison types |
| color_mode       | str  | grayscale, false color jet, false color vidiris, binned coolwarm | grayscale | No | Color mode |
| output_name      | str  | - | - | No | Output file name for the heatmap. The heatmap will be saved to {OUTPUT_FOLDER} |


## Dependencies
- multipledispatch
- pandas
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
Click on "start.bat" to run the program. Inputs arguments can either be specified in the config.yaml file, or in the terminal after the program has been started. 
- **WARNING: values in config.yaml always override those chosen in terminal. If you want to specify them in the terminal, set the values in config.yaml to NULL beforehand.** 