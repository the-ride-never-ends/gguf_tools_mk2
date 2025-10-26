#!/usr/bin/env python3
"""Tensor to image converter for transformer models (GGUF and PyTorch)"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Iterable, Protocol


import tqdm
import numpy as np
import numpy.typing as npt


from .utils.write_array_to_geotiff import write_array_to_geotiff
from .tensor_to_graph import tensor_to_graph


from config.config import (
    CFG_SD_CLIP_THRESHOLD,
    CFG_SD_POSITIVE_THRESHOLD,
    CFG_SD_NEGATIVE_THRESHOLD,
    CFG_NEG_SCALE,
    CFG_POS_SCALE,
    CFG_MID_SCALE,
)
from logger import logger

from config.config import OUTPUT_FOLDER, INPUT_FOLDER
from config.file_specific_configs import FileSpecificConfigs
config: Callable = FileSpecificConfigs().config

MODEL: str = config("MODEL")
MODEL_TYPE: str = config("MODEL_TYPE")
TENSOR_NAME: str = config("TENSOR_NAME")
COLOR_RAMP_TYPE: str = config("COLOR_RAMP_TYPE")
OUTPUT_NAME: Path = config("OUTPUT_NAME")
SHOW_WITH: str = config("SHOW_WITH")
MATCH_GLOB: bool = config("MATCH_GLOB")
MATCH_REGEX: bool = config("MATCH_REGEX")
MATCH_1D: bool = config("MATCH_1D")
ADJUST_1D_ROWS: int = config("ADJUST_1D_ROWS")
SCALE: float = config("SCALE")
FORCE: bool = config("FORCE")
OUTPUT_MODE: str = config("OUTPUT_MODE")

try:
    from PIL import Image
except ImportError as e:
    logger.error(f"{__file__} requires Pillow. Example: pip install pillow")
    raise e

try:
    from gguf.constants import GGMLQuantizationType
    from gguf.gguf_reader import GGUFReader, ReaderTensor
except ImportError as e:
    logger.error(f"{__file__} could not load GGMLQuantizationType, GGUFReader, ReaderTensor: {e}")
    raise e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    logger.error(f"{__file__} requires matplotlib installed. Example: pip install matplotlib")
    raise e


from .model_classes.quantized_class import Quantized_Q8_0
from .model_classes.model_abstract_class import Model
from .model_classes.gguf_model import GGUFModel
from .model_classes.torch_model import TorchModel
from .model_classes.safetensor_model import SafetensorModel


def calculate_mad_and_median(tensor: npt.NDArray[np.float32], axis: int = None) -> tuple[float,float]:
    """
    Median and MADs (Median Absolute Deviation). MAD = median(|Yi - median(Yi)|)
    """
    median = np.median(tensor, axis=axis)
    mad = np.median(np.abs(tensor - median), axis=axis) # MAD
    if axis == 1:
        return median[:, None], mad[:, None]
    else:
        return median, mad

def calculate_mean_and_standard_deviation(tensor: npt.NDArray[np.float32], axis: int = None) -> tuple[float,float]:
    mean = np.mean(tensor, dtype=np.float64, axis=axis)
    std_dev = np.std(tensor, dtype=np.float64, axis=axis)
    if axis == 1:
        return mean[:, None], std_dev[:, None]
    else:
        return mean, std_dev


# from random import random
# from easy_nodes import (
#     NumberInput,
#     ComfyNode,
#     MaskTensor,
#     StringInput,
#     ImageTensor,
#     Choice,
# )
# import easy_nodes
# import torch


def comfyui_node():
    pass


#@comfyui_node()
def gguf_tensor_to_image_comfy_ui_node(
                                    adjust_1d_rows: int = 32,
                                    #mode: Choice["mean-devs-overall", "mean-devs-per-row", "mean-devs-overall"],
                                    model: str = None,
                                    model_type: str = None,
                                    match_glob: bool = True,
                                    match_regex: bool = True,
                                    match_1d: bool = True,
                                    output: str = None,
                                    scale: float = 1.0,
                                    show_with: str = "auto",
                                    ) -> None:
    """
    ComfyUI node for generating a heatmap image from a tensor in a GGUF or PyTorch model.

    Args:
        adjust_1d_rows (int): Number of rows to use when reshaping 1D tensors. Default is 32.
        mode (str): Mode for calculating central tendency and deviation. 
            Options include "mean-devs-overall", "mean-devs-rows", "mean-devs-cols", 
            "median-devs-overall", "median-devs-rows", "median-devs-cols". 
            Default is "mean-devs-overall".
        model (str): Path to the model file. Can be a GGUF or PyTorch model.
        model_type (str): Type of the model. Can be "gguf" or "torch". 
            If not specified, it's inferred from the file extension.
        match_glob (bool): If True, use glob pattern matching for tensor names. Default is True.
        match_regex (bool): If True, use regex pattern matching for tensor names. Default is True.
        match_1d (bool): If True, include 1D tensors in matching. Default is True.
        output (str): Path for the output image file.
        scale (float): Scale factor for the output image. Default is 1.0.
        show_with (str): Program to use for displaying the image. Default is "auto".

    Returns:
        dict: A dictionary containing the UI information with the generated heatmap image.
    """
    mode = None
    run = TensorToImage(adjust_1d_rows=adjust_1d_rows,
                            mode=mode,
                            model=model,
                            model_type=model_type,
                            match_glob=match_glob,
                            match_regex=match_regex,
                            match_1d=match_1d,
                            output=output,
                            scale=scale,
                            show_with=show_with)
    run.tensor_to_image()

    return {"ui": {"images": [run.heatmap_image]}} 













["mean", "median", "absolute"]

class TensorToImage:
    """
    A class for converting tensor data from machine learning models into visual representations.
    This class supports extracting tensors from various model formats (GGUF, PyTorch, SafeTensors)
    and converting them into heatmap images or GeoTIFF files. The visualization uses color mapping
    to represent statistical deviations from central tendency measures.
    Attributes:
        SUPPORTED_IMAGE_TYPES (tuple): Supported output image file extensions.
        SUPPORTED_MODEL_TYPES (tuple): Supported input model file extensions.
    The class provides several visualization modes:
        - Mean and standard deviation (overall, by rows, by columns)
        - Median and median absolute deviation (overall, by rows, by columns)
        - Raw values (for GeoTIFF output)
    Color mapping options:
        - Discrete: Uses predefined color scales with distinct color bins
        - Continuous: Uses smooth color gradients based on deviation thresholds
    Supported model formats:
        - GGUF (.gguf): Quantized model format
        - PyTorch (.pth): PyTorch model files
        - SafeTensors (.safetensors): Safe tensor storage format
    Output formats:
        - Standard image formats: PNG, JPG, JPEG, BMP, GIF
        - GeoTIFF formats: TIFF, TIF, GEOTIFF (preserves numerical values)
    Example:
        >>> converter = TensorToImage(
        ...     model="my_model.gguf",
        ...     tensor_name="blk.2.ffn_down.weight",
        ...     output_mode="mean-devs-overall",
        ...     output_name="tensor_visualization.png"
        ... )
        >>> converter.tensor_to_image()
        Configuration parameters can be provided via YAML constants or keyword arguments,
        with YAML constants taking precedence over interactive arguments.
    """
    SUPPORTED_IMAGE_TYPES = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.geotiff', '.tif')
    SUPPORTED_MODEL_TYPES = ('.gguf','.pth','.safetensors')

    def __init__(self, **kwargs) -> None:
        # NOTE YAML constants always take precedent over interactive arguments.

        # Load configuration
        self._load_config(kwargs)

        # Set up model
        self._setup_model()

        # Set up tensor parameters
        self._setup_tensor_parameters(kwargs)

        # Set up output parameters
        self._setup_output_parameters(kwargs)

        # Set up processing parameters
        self._setup_processing_parameters()

    def _load_config(self, kwargs):
        self.adjust_1d_rows = ADJUST_1D_ROWS or kwargs.pop("adjust_1d_rows", 32)
        self.output_mode = OUTPUT_MODE or kwargs.pop("output_mode", "mean-devs-overall")
        self.model = MODEL or kwargs.pop("model")
        self.model_type: str = MODEL_TYPE or kwargs.pop("model_type")
        self.match_glob = MATCH_GLOB or kwargs.pop("match_glob", True)
        self.match_regex = MATCH_REGEX or kwargs.pop("match_regex", True)
        self.match_1d = MATCH_1D or kwargs.pop("match_1d", True)
        self.scale = SCALE or kwargs.pop("scale", 1.0)
        self.show_with = SHOW_WITH or kwargs.pop("show_with", None)
        self.color_ramp_type = COLOR_RAMP_TYPE or kwargs.pop("color_ramp_type", "continuous")


    def _setup_model(self):

        if self.model is None:
            raise ValueError("No model specified in MakeImage class parameters")

        self.model_path: str = os.path.join(INPUT_FOLDER, self.model.lstrip("/\\"))
        logger.debug(f"self.model_path: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Change self.model from the model's file path to an instance of it.
        self.model: Model

        if self.model_type == "gguf" or self.model.lower().endswith(".gguf"):
            self.model = GGUFModel(self.model_path)
        elif self.model_type == "torch" or self.model.lower().endswith(".pth"):
            self.model = TorchModel(self.model_path)
        elif self.model_type == "safetensors" or self.model.lower().endswith(".safetensors"):
            self.model = SafetensorModel(self.model_path)
        else:
            raise ValueError("Unsupported model type.")
    
        logger.info("Model loaded successfully")


    def _setup_tensor_parameters(self, kwargs):
        if self.match_glob and self.match_regex:
            logger.warning("match_glob and match_regex are mutually exclusive options. Defaulting to match_glob...")
            self.match_regex = False

        self.tensor_name: str = TENSOR_NAME or kwargs.pop("tensor_name", "blk.2.ffn_down.weight")
        self.names: str | list[str] = self.get_tensor_names()


    def _setup_output_parameters(self, kwargs):
        _output_file_name = OUTPUT_NAME or kwargs.pop("output_name", "output.png")
        self.output_path = Path(OUTPUT_FOLDER) / _output_file_name
        self.heatmap_image: Image = None


    def _setup_processing_parameters(self):
        # Any additional processing parameters can be set up here
        pass


    def get_tensor_names(self) -> list[str]:
        """
        Retrieves tensor names from the model based on matching criteria.

        This method filters tensor names based on the specified matching method:
        - If match_glob is True, it uses glob pattern matching.
        - If match_regex is True, it uses regular expression matching.
        - Otherwise, it directly uses the provided tensor names.

        The method updates the 'names' attribute of the class with the matched tensor names.

        Returns:
            list[str]: A list of matched tensor names.

        Note:
            - For glob matching, fnmatch.fnmatchcase is used.
            - For regex matching, re.compile and search are used.
            - When neither glob nor regex matching is used, only valid tensor names are included.
        """
        logger.debug("Creating tensor dictionary...")

        tensor_names = (name for name in self.model.tensor_names())

        tensor_dict = {}
        for name in tqdm.tqdm(tensor_names):
            tensor_dict[name] = self.model.get_type_name(name)
            if self.tensor_name in tensor_dict:
                break

        if len(tensor_dict) == 0:
            logger.error(f"No tensors found in loaded model")
            raise ValueError("No tensors found in loaded model")
        elif self.tensor_name not in tensor_dict:
            logger.error(f"Specified tensor '{self.tensor_name}' not found in model")
            raise ValueError(f"Specified tensor '{self.tensor_name}' not found in model")
        else:
            logger.info(f"Found specified tensor: {self.tensor_name}")
            if len(self.tensor_name) == 1:
                return self.tensor_name
            else:
                return tensor_dict[self.tensor_name]


    def reshape_1d_tensor_into_2d(self, tensor: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Reshape a 1D tensor into a 2D tensor if desired.

        This method checks if the input tensor is 1-dimensional and reshapes it based on the 'adjust_1d_rows' attribute.
        If 'adjust_1d_rows' is set, it reshapes the tensor into a 2D array with the specified number of rows.
        If 'adjust_1d_rows' is not set, it adds an extra dimension to make the tensor 2D with 1 row.

        Args:
            tensor (npt.NDArray[np.float32]): The input tensor to be reshaped.

        Returns:
            npt.NDArray[np.float32]: The reshaped tensor. If the input was already 2D or higher, it's returned unchanged.
        """
        # Check if the tensor is 1-dimensional
        if len(tensor.shape) == 1:
            # If adjust_1d_rows attribute is set
            if self.adjust_1d_rows is not None:
                # Reshape the 1D tensor into a 2D array with specified number of rows
                # The number of columns is calculated by dividing the total elements by the number of rows.
                tensor = tensor.reshape((self.adjust_1d_rows, tensor.shape[0] // self.adjust_1d_rows))
            else:
                # If adjust_1d_rows is not set, add an extra dimension to make it 2D
                # This creates a 2D array with 1 row and the original data as columns
                tensor = tensor[None, :]
        return tensor


    def get_central_tendency_and_deviation(self, tensor: npt.NDArray[np.float32]) -> tuple[float,float]:
        """
        Calculate central tendency and deviation metrics for the given tensor.

        This method computes either mean and standard deviation or median and median absolute deviation (MAD)
        based on the specified mode. The calculations can be performed overall, by rows, or by columns.

        Args:
            tensor (npt.NDArray[np.float32]): Input tensor for which to calculate metrics.

        Returns:
            tuple[float, float]: A tuple containing:
                - ct: Mean or median of the tensor.
                - dv: Standard deviation or MAD of the tensor.

        Raises:
            ValueError: If an unknown mode is specified.

        Notes:
            - The mode is determined by the self.output_mode attribute.
            - Available modes:
                - "devs-overall", "mean-devs-overall": Overall mean and standard deviation
                - "devs-rows", "mean-devs-rows": Mean and standard deviation by rows
                - "devs-cols", "mean-devs-cols": Mean and standard deviation by columns
                - "median-devs-overall": Overall median and MAD
                - "median-devs-rows": Median and MAD by rows
                - "median-devs-cols": Median and MAD by columns
        """
            # Calculate central tendency and deviation based on arguments.
        match self.output_mode:
            # Means and Standard Deviations
            case "devs-overall" | "mean-devs-overall":
                logger.info(f"Calculating mean and standard deviation overall for tensor: {self.tensor_name}")
                ct, dv = calculate_mean_and_standard_deviation(tensor)
                type_ = "mean", "standard deviation"
            case "devs-rows" | "mean-devs-rows":
                logger.info(f"Calculating mean and standard deviation by rows for tensor: {self.tensor_name}")
                ct, dv = calculate_mean_and_standard_deviation(tensor,axis=1)
                type_ = "mean", "standard deviation"
            case "devs-cols" | "mean-devs-cols":
                logger.info(f"Calculating mean and standard deviation by columns for tensor: {self.tensor_name}")
                ct, dv = calculate_mean_and_standard_deviation(tensor,axis=0)
                type_ = "mean", "standard deviation"

            # Median and MADs (Median Absolute Deviation). MAD = median(|Yi – median(Yi)|)
            case "median-devs-overall":
                logger.info(f"Calculating median and median absolute deviation overall for tensor: {self.tensor_name}")
                ct, dv = calculate_mad_and_median(tensor)
                type_ = "median", "median absolute deviation"
            case "median-devs-rows":
                logger.info(f"Calculating median and median absolute deviation by rows for tensor: {self.tensor_name}")
                ct, dv = calculate_mad_and_median(tensor, axis=1)
                type_ = "median", "median absolute deviation"
            case "median-devs-cols":
                logger.info(f"Calculating median and median absolute deviation by columns for tensor: {self.tensor_name}")
                ct, dv = calculate_mad_and_median(tensor, axis=0)
                type_ = "median", "median absolute deviation"
            case _:
                raise ValueError("Unknown mode")

        logger.info(f"""
            *** Tensor Stats ***
            name: {self.tensor_name}
            shape: {tensor.shape}
            __len__: {tensor.__len__()}
            {type_[0]}: {ct}
            {type_[1]}: {dv}
            max: {np.max(tensor)}
            min: {np.min(tensor)}
        """)

        return ct, dv, type_

    
    def normalize(self, tensor: npt.NDArray[np.float32], ct: float, dv: float) -> npt.NDArray[np.float32]:
        """Transform a tensor of values into a tensor of normalized standard deviations 
            based on the mean of those values.

        Args:
            tensor (npt.NDArray[np.float32]): Input tensor to be normalized.
            ct (float): A measure of central tendency (mean, median, etc.)
            dv (float): A measure of deviancy based on the central tendency 
                (standard deviation, median absolute deviation, etc.)

        Returns:
            npt.NDArray[np.float32]: Normalized tensor of standard deviations.
        """
        # Avoid division by zero
        return np.zeros_like(tensor) if dv == 0 else (tensor - ct) / dv



    def _scale_tensor_to_0_255(
            self, 
            tensor: npt.NDArray[np.float32],
            ct: float, 
            dv: float,
            ) -> tuple[npt.NDArray[np.uint8], float, float]:
        """
        Scale a tensor to the 0-255 range for image representation.

        Args:
            tensor (npt.NDArray[np.float32]): Input tensor to be scaled.
            ct (float): Central tendency value of the tensor.
            dv (float): Deviation value of the tensor.

        Returns:
            tuple[npt.NDArray[np.uint8], float, float]: 
                Scaled tensor with values in the 0-255 range, positive threshold, and negative threshold.
        """
        # Map the 2D tensor data to the same range as an image 0-255.
        sdp_max = ct + CFG_SD_CLIP_THRESHOLD * dv
            # Set the positive and negative SD thresholds for this specific tensor.
        sdp_thresh = ct + CFG_SD_POSITIVE_THRESHOLD * dv
        sdn_thresh = ct - CFG_SD_NEGATIVE_THRESHOLD * dv
            # Calculate the absolute difference between the tensor data and the mean.
        tda = np.minimum(np.abs(tensor), sdp_max).repeat(3, axis=-1).reshape((*tensor.shape, 3))

        # Scale that range to between 0 and 255.
        tda = 255 * ((tda - np.min(tda)) / np.ptp(tda))

        return tda, sdp_thresh, sdn_thresh

    def _map_discrete(self, 
                      tda: npt.NDArray[np.uint8], 
                      tensor: npt.NDArray[np.float32], 
                      ct: float, 
                      dv: float
                      ) -> npt.NDArray[np.uint8]:
        red_1, red_2, red_3, red_4, red_5, red_6, red_7 = (103,0,13), (179,18,24), (221,42,37), (246,87,62), (252,134,102), (252,179,152), (254,220,205)
        green_1, green_2, green_3, green_4, green_5, green_6, green_7 = (226,244,221), (191,230,185), (148,211,144), (96,186,108), (50,155,81), (13,120,53), (0,68,27) 

        # Negative SD Values. "Reds" color ramp, where darker reds represent more negative SD values.
        tda[tensor <= (ct - 6 * dv), ...] *= red_1  # 67000d
        tda[np.logical_and(tensor > (ct - 6 * dv), tensor <= (ct - 5 * dv)), ...] *= red_2  # b31218
        tda[np.logical_and(tensor > (ct - 5 * dv), tensor <= (ct - 4 * dv)), ...] *= red_3  # dd2a25
        tda[np.logical_and(tensor > (ct - 4 * dv), tensor <= (ct - 3 * dv)), ...] *= red_4  # f6573e
        tda[np.logical_and(tensor > (ct - 3 * dv), tensor <= (ct - 2 * dv)), ...] *= red_5  # fc8666
        tda[np.logical_and(tensor > (ct - 2 * dv), tensor <= (ct - 1 * dv)), ...] *= red_6  # fcb398
        tda[np.logical_and(tensor > (ct - 1 * dv), tensor <= (ct)), ...] *= red_7  # fedccd

        # Positive SD Values. "Greens" color ramp, where darker greens represent more positive SD values.
        tda[np.logical_and(tensor > (ct + 1 * dv), tensor <= (ct)), ...] *= green_1  # e2f4dd
        tda[np.logical_and(tensor > (ct + 2 * dv), tensor <= (ct + 1 * dv)), ...] *= green_2  # bfe6b9
        tda[np.logical_and(tensor > (ct + 3 * dv), tensor <= (ct + 2 * dv)), ...] *= green_3  # 94d390
        tda[np.logical_and(tensor > (ct + 4 * dv), tensor <= (ct + 3 * dv)), ...] *= green_4  # 60ba6c
        tda[np.logical_and(tensor > (ct + 5 * dv), tensor <= (ct + 4 * dv)), ...] *= green_5  # 329b51
        tda[np.logical_and(tensor > (ct + 6 * dv), tensor <= (ct + 5 * dv)), ...] *= green_6 # 0d7835
        tda[tensor >= (ct + 6 * dv), ...] *= green_7  # 00441b
        return tda


    def _map_continuous(self, 
                        tda: npt.NDArray[np.uint8], 
                        tensor: npt.NDArray[np.float32], 
                        sdn_thresh: float, 
                        sdp_thresh: float
                        ) -> npt.NDArray[np.uint8]:
        tda[tensor <= sdn_thresh, ...] *= CFG_NEG_SCALE
        tda[tensor >= sdp_thresh, ...] *= CFG_POS_SCALE
        tda[np.logical_and(tensor > sdn_thresh, tensor < sdp_thresh), ...] *= CFG_MID_SCALE
        return tda


    def make_image_of_(self, 
                       tensor: npt.NDArray[np.float32],
                       ct: float,
                       dv: float,
                       type_: tuple[str, str],
                       use_pyplot: bool = False, 
                       ) -> Image:
        """
        Create an image representation of a given tensor.

        This method processes a tensor and converts it into an RGB image, where the color
        intensity represents the deviation from the central tendency (mean or median).

        Args:
            tensor (npt.NDArray[np.float32]): Input tensor to be converted into an image.

        Returns:
            Image: A PIL Image object representing the tensor data.

        The method performs the following steps:
        1. Reshapes 1D tensors into 2D if necessary.
        2. Calculates central tendency and deviation metrics based on the specified mode.
        3. Maps tensor values to color intensities:
        - For 'discrete' color ramp:
            - Uses discrete color scales for negative and positive deviations.
            - Darker reds represent more negative deviations.
            - Darker greens represent more positive deviations.
        - For 'continuous' color ramp:
            - Applies continuous color scaling based on deviation thresholds.
            - Red for negative deviations, green for positive, and scaled colors in between.
        4. Converts the resulting color data into a PIL Image.

        The color mapping is influenced by several class attributes and constants:
        - self.color_ramp_type: Determines whether to use 'discrete' or 'continuous' color mapping.
        - CFG_SD_CLIP_THRESHOLD: Maximum number of standard deviations for clipping.
        - CFG_SD_POSITIVE_THRESHOLD, CFG_SD_NEGATIVE_THRESHOLD: Thresholds for positive and negative deviations.
        - CFG_NEG_SCALE, CFG_POS_SCALE, CFG_MID_SCALE: Color scaling factors for different ranges.

        Note:
        - The color mapping logic is sensitive to the statistical properties of the input tensor.
        """
        # If it's a 1-dimensional tensor, plot it as bar chart
        if tensor.ndim == 1:
            try:
                image = tensor_to_graph(
                    model_name=self.model_path.split("\\")[-1],
                    tensor_name=self.tensor_name, 
                    tensor=tensor,
                    type_=type_,
                    ct=ct, 
                    dv=dv
                )
            except Exception as e:
                raise RuntimeError(f"Failed to convert 1-d tensor to graph: {e}") from e
            else:
                return image

        tda, sdp_thresh, sdn_thresh = self._scale_tensor_to_0_255(tensor, ct, dv)

        match self.color_ramp_type :
            case "discrete":  # Discrete Colors
                tda = self._map_discrete(tda, tensor, ct, dv)
            case "continuous":  # Continuous Colors
                tda = self._map_continuous(tda, tensor, sdn_thresh, sdp_thresh)
            case _:
                raise ValueError("Unknown color ramp type")

        try:
            image = Image.fromarray(tda.astype(np.uint8), "RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to convert tensor to image: {e}") from e
        else:
            return image


    def _extract_tensor_from_model(self) -> npt.NDArray[np.float32]:
        """
        Extracts a tensor from a given model file based on the tensor name.

        Args:
            model_file (str): Path to the model file. Can be a GGUF or PyTorch model.

        Returns:
            npt.NDArray[np.float32]: The extracted tensor as a NumPy array.

        Raises:
            ValueError: If the model type is unknown or if the tensor extraction fails.
        """
        valid, error_message = self.model.valid(self.tensor_name)
        if valid: 
            return self.model.get_as_f32(self.tensor_name)
        else:
            raise ValueError(f"Error extracting tensor from {self.model}: {error_message}")


    def set_output_path_for_image(self, tk: str) -> None:
        """
        Set the output path for the image of a specific tensor.

        This method determines the output path for the image generated from a given tensor.
        If multiple tensors are being processed, it modifies the output path to include
        the tensor name to avoid overwriting.

        Args:
            tk (str): The name of the tensor being processed.

        Raises:
            ValueError: If the tensor name contains a forward slash ('/').

        Note:
            - If self.output_path is set and multiple tensors are being processed,
            the method prepends the tensor name to the output filename.
            - If self.output_path is not set and multiple tensors are being processed,
            the method appends the tensor name to the output path.
        """
        if not isinstance(tk, str):
            raise TypeError(f"Tensor name must be a string, got {type(tk).__name__}")
        if "/" in tk:
            raise ValueError("Bad tensor name, had '/' in it.")

        if self.output_path is not None:
            self.output_path: Path
            if len(self.names) > 1:
                file_path = self.output_path.parent
                filename = self.output_path.name
                self.output_path = file_path / f"tensor_to_image_{self.output_mode}_{tk}.{filename}"
        else:
            if len(self.names) > 1:
                self.output_path: Path
                file_path = self.output_path.parent
                filename = self.output_path.name
                self.output_path = file_path / f"tensor_to_image_{self.output_mode}{tk}.{filename}"
        return


    def _write_to_tiff(self, tensor: npt.NDArray[np.float32]) -> None:
        """Write tensor data to a TIFF file."""
        match self.output_mode:
            case "values-as-is":
                logger.info("Saving tensor values as-is to tiff file.")
                return write_array_to_geotiff(tensor, self.output_path)
            case _:
                logger.info("Saving central tendency values to tiff file.")
                ct, dv, _ = self.get_central_tendency_and_deviation(tensor)
                tensor = self.normalize(tensor, ct, dv)
                return write_array_to_geotiff(tensor, self.output_path)


    def tensor_to_image(self) -> None:
        """
        Process and convert transformer tensors to images.

        This method iterates through the selected tensors, processes each one,
        and converts them to images. For each tensor, it:
        - 1. Retrieves the tensor data from the model.
        - 2. Skips 1D tensors if not explicitly included.
        - 3. Sets the output path for the image.
        - 4. Creates a heatmap image from the tensor data.
        - 5. Scales the image if specified.
        - 6. Saves the image to the output path if specified.
        - 7. Displays the image using the specified viewer if requested.

        The method handles both single and multiple tensor processing,
        adjusting the output naming convention accordingly.

        Note:
            - The processing is influenced by various class attributes like
            match_1d, scale, output, and show_with.
            - Image scaling uses Lanczos resampling for better quality.
            - For displaying images without saving, a temporary file is used.

        Raises:
            Any exceptions from underlying methods (e.g., file I/O errors).
        """
        logger.info(f"Matching tensors: {self.tensor_name}")
        # logger.info(f"len self.names: {len(self.names)}")

        #assert len(self.names) == 1, "Multiple layers not implemented at this time."

        #for tk in self.names:
        logger.info(f"Processing tensor {self.tensor_name!r}") #(type:{self.model.get_type_name(tk)}, shape:{tensor.shape})",)
        tensor = self._extract_tensor_from_model()

        if not self.match_1d and len(tensor.shape) == 1:
            return #continue

        self.reshape_1d_tensor_into_2d(tensor)

        self.set_output_path_for_image(self.tensor_name)

        if self.output_path.suffix.lower() in ('.tif', '.tiff', '.geotiff'):
            return self._write_to_tiff(tensor)

        ct, dv, type_ = self.get_central_tendency_and_deviation(tensor)

        self.heatmap_image = img = self.make_image_of_(tensor, ct, dv, type_)

        if self.scale != 1.0: # Scale the image so that it fits on the screen (?)
            self.heatmap_image = img = img.resize(
                (
                    max(1, int(img.width * self.scale)),
                    max(1, int(img.height * self.scale)),
                ),
                resample=Image.Resampling.LANCZOS,
            )

        if self.output_path is not None:
            logger.info(f"Saving to '{self.output_path}'...")
            img.save(self.output_path)

        if self.show_with:
            logger.info("Displaying to screen using img...")

            if self.output_path is not None:
                # TODO: Find a default program that works with this.
                # subprocess.call((self.show_with, self.output_path))  # noqa: S603
                img.show()
            else:
                with tempfile.NamedTemporaryFile(suffix=".png") as fp:
                    img.save(fp, format="png")
                    img.show()
                    fp.flush()
                    #subprocess.call((self.show_with, fp.name))  # noqa: S603


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tensor to image converter for LLM models (GGUF and PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """\
            Information on output modes:
              devs-*:
                overall: Calculates the standard deviation from the mean.
                         By default, values below the mean will be red and values above it will be green.
                rows   : Same as above, except the calculation is based on rows.
                cols:  : Same as above, except the calculation is based on columns.
        """,
        ),
    )
    parser.add_argument(
        "model",
        type=str,
        help="model filename, can be GGUF or PyTorch (if PyTorch support available)",
    )
    parser.add_argument(
        "tensor",
        nargs="+",
        type=str,
        help="Tensor name, may be specified multiple times UNLESS --match-glob or --match-regex is used",
    )
    parser.add_argument(
        "--color_ramp_type",
        choices=["continuous", "discrete"],
        default="continuous",
        help="Color ramp type, Default: continuous",
    )
    output_group = parser.add_argument_group(
        "output",
        "At least one of the following must be specified:",
    )
    output_group.add_argument(
        "--output",
        type=Path,
        help="Output file, will be prefixed with the tensor name if multiple tensor names are specified",
    )
    output_group.add_argument(
        "--show-with",
        help="""
            Show the result with the specified application.
            WARNING: If processing multiple tensors and your image application
            does not block then you will end up with a bunch of huge images displayed at the same time""",
    )
    wildcard_group = parser.add_mutually_exclusive_group()
    wildcard_group.add_argument(
        "--match-glob",
        action="store_true",
        help="Interpret tensor name as a glob, so wildcards like blk.0.* will work",
    )
    wildcard_group.add_argument(
        "--match-regex",
        action="store_true",
        help="Interpret tensor name as a regex, so regular expressions like ^blk\\.[012]\\.attn will work",
    )

    parser.add_argument(
        "--match-1d",
        action="store_true",
        help="When using a wildcard, also match 1 dimensional tensors",
    )
    parser.add_argument(
        "--adjust-1d-rows",
        type=int,
        help="""
        Instead of rendering 1D tensors as a wide image with one row, rearrange into multiple rows.
        For example, if we have a 1D tensor 3,200 elements and specify "--adjust-1d-rows 32",
        the output image will have dimensions 100x32. Note: The tensor size must be divisible by
        the specified value.
        """,
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale the image. Default: 1.0 (no scaling)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwriting the output file if it already exists",
    )
    parser.add_argument(
        "--mode",
        choices=["devs-overall", "devs-rows", "devs-cols"],
        default="devs-overall",
        help="""
        Output modes (see below). Default: devs-overall
        NOTE: If the program is started using start.bat
        the options "mean-devs-overall", "mean-devs-rows", "mean-devs-cols" are available as aliases for
        "devs-overall", "devs-rows", "devs-cols", as well as "median-devs-overall", "median-devs-rows", "median-devs-cols"
        overall: Calculate the mean and standard dv over the entire tensor.
        rows   : Same as above, except the calculation is based on rows.
        cols   : Same as above, except the calculation is based on columns.
        """,
    )
    parser.add_argument(
        "--model-type",
        choices=["gguf", "torch"],
        help="Specify model type (gguf or torch)" ,
    )
    return parser


def parse_arguments() -> dict:
    parser = create_parser()
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    if not (args.show_with or args.output):
        logger.error("! At least one of --show or --output must be specified", file=sys.stderr)
        sys.exit(1)

    if (args.match_regex or args.match_glob) and len(args.tensor) != 1:
        logger.warning(
            "! Can only specify one tensor name (pattern) when using --match-glob or --match-regex",
            file=sys.stderr,
        )
    
    # Convert the namespace to a dictionary
    kwargs = vars(args)

    logger.debug("Arguments as kwargs dictionary:")
    for key, value in kwargs.items():
        logger.debug(f"{key}: {value}")

    return kwargs


def main() -> None:
    logger.info("* Starting tensor_to_image program...")

    TensorToImage(parse_arguments()).tensor_to_image()

    logger.info("\n* Done.")

if __name__ == "__main__":
    main()
