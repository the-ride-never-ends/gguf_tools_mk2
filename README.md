# gguf_tools_mk2

## Overview
A suite of tools to manipulate GGUF, Torch, Safetensor, and other transformers model types. 
This suite is a fork of KerfuffleV2's gguf-tools. You can find it here: https://github.com/KerfuffleV2/gguf-tools

## Key Features
### tensor_to_image
- description: Generate false-color visualizations of tensor weight values and derivative statistics from transformers models. This can be used to visualize various descriptive statistics about a model's tensor, identify potential issues with training, and identify outliers or patterns. This tool supports GGUF, Torch, and Safetensor model types. **Warning/TODO: Torch models have not been tested yet.**


| Argument Name    | Type | Choices | Default | Required | Help Text |
|------------------|------|----------------|---------|----------|-----------|
| model            | str  | - | - | Yes | Model filename, can be GGUF, PyTorch, or Safetensor (if support available) |
| tensor           | str  | - | - | Yes | Tensor name, may be specified multiple times UNLESS --match-glob or --match-regex is used |
| color_ramp_type  | str  | continuous, discrete | continuous | No | Color ramp type |
| output           | Path | - | - | No* | Output file, will be prefixed with the tensor name if multiple tensor names are specified |
| show-with        | str  | - | - | No* | Show the result with the specified application |
| match-glob       | bool | - | False | No | Interpret tensor name as a glob |
| match-regex      | bool | - | False | No | Interpret tensor name as a regex |
| match-1d         | bool | - | False | No | When using a wildcard, also match 1 dimensional tensors |
| adjust-1d-rows   | int  | - | - | No | Rearrange 1D tensors into multiple rows |
| scale            | float | - | 1.0 | No | Scale the image |
| force            | bool | - | False | No | Force overwriting the output file if it already exists |
| mode             | str  | devs-overall<br> devs-rows<br> devs-cols<br> median-devs-overall<br> median-devs-rows<br> median-devs-cols<br> values-as-is<br> | devs-overall | No | Output modes. devs returns an image or raster of the tensor's normalized standard deviations. Medians returns a tensor's MAD (Median Absolute Deviation). Values-as-is directly maps the tensor's values to a raster.  The options "mean-devs-overall", "mean-devs-rows", "mean-devs-cols" are available as aliases for "devs-overall", "devs-rows", "devs-cols" |
| model-type       | str  | gguf, torch, safetensors | - | No | Specify model type |

*At least one of --output or --show-with must be specified.

Note: The program may offer additional mode options when started using start.bat, including mean-devs-overall, mean-devs-rows, mean-devs-cols, median-devs-overall, median-devs-rows, and median-devs-cols.


### tensor_comparison_to_image
- description: Generate false-color images or rasters showing differences in weight values in equivalent tensors between two similar transformers models. This can be used to visualize the differences between base models and fine-tunes, models trained with different hyperparameters, or between models trained on different datasets.

| Argument Name    | Type | Choices | Default | Required | Help Text |
|------------------|------|---------|---------|----------|-----------|
| model_file1      | str  | - | - | Yes | Filename for the first model, can be GGUF, PyTorch, or Safetensor (if support available) |
| model_file2      | str  | - | - | No | Filename for the second model, can be GGUF, PyTorch, or Safetensor (if support available) |
| tensor_name      | str  | - | - | No | Tensor name, must be from models with the same foundation architecture for the differences to be valid. |
| comparison_type  | str  | mean, median, absolute | mean | No | Comparison types |
| color_mode       | str  | grayscale, false color jet, false color vidiris, binned coolwarm | grayscale | No | Color mode |
| output_name      | str  | - | - | No | Output file name for the heatmap. The heatmap will be saved to {OUTPUT_FOLDER} |


### gguf_checksum
Checksum utility for GGUF files. Not implemented, but the program can still be run via commandline arguments.

### gguf_frankenstein
Model combining utility for GGUF files. Not implemented, but the program can still be run via commandline arguments.

### tensor_to_histogram
TBA

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
- safetensors

## Usage
Click on "start.bat" to run the program. Inputs arguments can either be specified in the config.yaml file, or in the terminal after the program has been started. 
- **WARNING: values in config.yaml always override those chosen in terminal. If you want to specify them in the terminal, set the values in config.yaml to NULL beforehand.** 