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


import numpy as np
import numpy.typing as npt


from .utils.write_array_to_geotiff import write_array_to_geotiff

from logger.logger import Logger
logger = Logger(logger_name=__name__)

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
except ImportError:
    logger.error("This script requires Pillow installed. Example: pip install pillow")
    sys.exit(1)

try:
    from gguf.constants import GGMLQuantizationType
    from gguf.gguf_reader import GGUFReader, ReaderTensor
except ImportError as e:
    logger.error(f"Could not load GGMLQuantizationType, GGUFReader, ReaderTensor: {e}")
    raise e



#### HARD CODED CONSTANTS ###
# Clip values to at max 7 standard deviations from the mean.
CFG_SD_CLIP_THRESHOLD = 7

# Number of standard deviations above the mean to be positively scaled.
CFG_SD_POSITIVE_THRESHOLD = 1

# Number of standard deviations below the mean to be negatively scaled.
CFG_SD_NEGATIVE_THRESHOLD = 1

# RGB scaling for pixels that meet the negative threshold.
CFG_NEG_SCALE = (1.2, 0.2, 1.2)

# RGB scaling for pixels that meet the positive threshold.
CFG_POS_SCALE = (0.2, 1.2, 1.2)

# RGB scaling for pixels between those ranges.
CFG_MID_SCALE = (0.1, 0.1, 0.1)
# CFG_MID_SCALE = (0.6, 0.6, 0.9) Original Values


from .model_classes.quantized_class import Quantized_Q8_0
from .model_classes.model_abstract_class import Model
from .model_classes.gguf_model import GGUFModel
from .model_classes.torch_model import TorchModel


def calculate_mad_and_median(tensor: npt.NDArray[np.float32], axis: int = None) -> tuple[float,float]:
    """
    Median and MADs (Median Absolute Deviation). MAD = median(|Yi - median(Yi)|)
    """
    median = np.median(tensor, dtype=np.float64, axis=axis)
    mad = np.median(np.abs(tensor - median), dtype=np.float64, axis=axis)
    return median[:, None], mad[:, None] if axis == 1 else median, mad

def calculate_mean_and_standard_deviation(tensor: npt.NDArray[np.float32], axis: int = None) -> tuple[float,float]:
    mean = np.mean(tensor, dtype=np.float64, axis=axis)
    std_dev = np.std(tensor, dtype=np.float64, axis=axis)
    return mean[:, None], std_dev[:, None] if axis == 1 else mean, std_dev


def comfyui_node():
    pass

#@comfyui_node()
def gguf_tensor_to_image_comfy_ui_node(
                                    adjust_1d_rows: int = 32,
                                    mode: str = "mean-devs-overall",
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


["mean", "median", "absolute"],

class TensorToImage:

    def __init__(self, **kwargs) -> None:
        # NOTE YAML constants always take precedent over interactive arguments.
        self.adjust_1d_rows: int = ADJUST_1D_ROWS or kwargs.pop("adjust_1d_rows",  32)
        self.mode: str = OUTPUT_MODE or kwargs.pop("mode", "mean-devs-overall")

        self.model: str = MODEL or kwargs.pop("model")
        if self.model is None:
            raise ValueError("No model specified in MakeImage class parameters")
        self.model_path: str = os.path.join(INPUT_FOLDER, self.model.lstrip("/\\"))
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model_type: str = MODEL_TYPE or kwargs.pop("model_type")
        self.match_glob: bool = MATCH_GLOB or kwargs.pop("match_glob", True)
        self.match_regex: bool = MATCH_REGEX or kwargs.pop("match_regex", True)

        if self.match_glob and self.match_regex:
            logger.warning("match_glob and match_regex are mutually exclusive options. Defaulting to match_glob...")
            self.match_regex = False

        self.match_1d: bool = MATCH_1D or kwargs.pop("match_1d", True)

        _output_file_name = OUTPUT_NAME or kwargs.pop("output", "output.png")
        self.output_path: Path = os.path.join(OUTPUT_FOLDER, _output_file_name)

        self.scale: float = SCALE or kwargs.pop("scale", 1.0)
        self.show_with: str = SHOW_WITH or kwargs.pop("show_with", None)
        self.heatmap_image = None

        # Change self.model from the model's file path to an instance of it.
        self.model: Model
        if self.model_type == "gguf" or self.model.lower().endswith(".gguf"):
            self.model = GGUFModel(self.model_path)
        elif self.model_type == "torch" or self.model.lower().endswith(".pth"):
            self.model = TorchModel(self.model_path)
        else:
            raise ValueError("Unsupported model type.")
        logger.info("Model loaded successfully")

        self.tensor_name: str = TENSOR_NAME or kwargs.pop("tensor", "blk.2.ffn_down.weight")
        self.names: list[str] = self.get_tensor_names()


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
        if self.match_glob:
            names = [ # Use fnmatch to find tensor names that match the given glob patterns
                name for name in self.model.tensor_names()
                if any(fnmatch.fnmatchcase(name, pat) for pat in self.tensor_name)
            ]
        elif self.match_regex:
            res = [re.compile(r) for r in self.tensor_name] # Compile the regex patterns
            names =  [  # Find tensor names that match any of the compiled regex patterns
                name for name in self.model.tensor_names() if any(r.search(name) for r in res)
            ]
        else:
            # Use the tensor names provided directly, but only if they are valid
            names = [name for name in self.tensor_name if self.model.valid(name)[0]]

        if len(names) == 0:
            logger.error(f"No names found in loaded model\nnames: {names}")
            raise ValueError("No names found in loaded model")
        else:
            logger.info(f"Got tensors: {names}")
            return names


    def reshape_1d_tensor_into_2d_if_desired(self, tensor: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
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


    def return_central_tendency_and_deviation_metrics(self, tensor: npt.NDArray[np.float32]) -> tuple[float,float]:
        """
        Calculate central tendency and deviation metrics for the given tensor.

        This method computes either mean and standard deviation or median and median absolute deviation (MAD)
        based on the specified mode. The calculations can be performed overall, by rows, or by columns.

        Args:
            tensor (npt.NDArray[np.float32]): Input tensor for which to calculate metrics.

        Returns:
            tuple[float, float]: A tuple containing:
                - central_tendency: Mean or median of the tensor.
                - deviation: Standard deviation or MAD of the tensor.

        Raises:
            ValueError: If an unknown mode is specified.

        Notes:
            - The mode is determined by the self.mode attribute.
            - Available modes:
                - "devs-overall", "mean-devs-overall": Overall mean and standard deviation
                - "devs-rows", "mean-devs-rows": Mean and standard deviation by rows
                - "devs-cols", "mean-devs-cols": Mean and standard deviation by columns
                - "median-devs-overall": Overall median and MAD
                - "median-devs-rows": Median and MAD by rows
                - "median-devs-cols": Median and MAD by columns
        """
            # Calculate mean and SD based on arguments.
        match self.mode:
            # Means and Standard Deviations
            case "devs-overall" | "mean-devs-overall":
                central_tendency, deviation = calculate_mean_and_standard_deviation(tensor)
            case "devs-rows" | "mean-devs-rows":
                central_tendency, deviation = calculate_mean_and_standard_deviation(tensor,axis=1)
            case "devs-cols" | "mean-devs-cols":
                central_tendency, deviation = calculate_mean_and_standard_deviation(tensor,axis=0)

            # Median and MADs (Median Absolute Deviation). MAD = median(|Yi – median(Yi)|)
            case "median-devs-overall":
                central_tendency, deviation = calculate_mad_and_median(tensor)
            case "median-devs-rows":
                central_tendency, deviation = calculate_mad_and_median(tensor, axis=1)
            case "median-devs-cols":
                central_tendency, deviation = calculate_mad_and_median(tensor, axis=0)
            case _:
                raise ValueError("Unknown mode")

        return central_tendency, deviation


    def make_image_of_(self, 
                       tensor: npt.NDArray[np.float32],
                       central_tendency: float,
                       deviation: float,
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
        # Map the 2D tensor data to the same range as an image 0-255.
        sdp_max = central_tendency + CFG_SD_CLIP_THRESHOLD * deviation
            # Set the positive and negative SD thresholds for this specific tensor.
        sdp_thresh = central_tendency + CFG_SD_POSITIVE_THRESHOLD * deviation
        sdn_thresh = central_tendency - CFG_SD_NEGATIVE_THRESHOLD * deviation
            # Calculate the absolute difference between the tensor data and the mean.
        tda = np.minimum(np.abs(tensor), sdp_max).repeat(3, axis=-1).reshape((*tensor.shape, 3))
            # Scale that range to between 0 and 255.
        tda = 255 * ((tda - np.min(tda)) / np.ptp(tda))

        match self.color_ramp_type :
            case "discrete":  # Discrete Colors
                # Negative SD Values. This uses a discrete "Reds" color ramp, where darker reds represent more negative SD values.
                tda[tensor <= (central_tendency - 6 * deviation), ...] *= (103,0,13)  # 67000d
                tda[np.logical_and(tensor > (central_tendency - 6 * deviation), tensor <= (central_tendency - 5 * deviation)), ...] *= (179,18,24)  # b31218
                tda[np.logical_and(tensor > (central_tendency - 5 * deviation), tensor <= (central_tendency - 4 * deviation)), ...] *= (221,42,37)  # dd2a25
                tda[np.logical_and(tensor > (central_tendency - 4 * deviation), tensor <= (central_tendency - 3 * deviation)), ...] *= (246,87,62)  # f6573e
                tda[np.logical_and(tensor > (central_tendency - 3 * deviation), tensor <= (central_tendency - 2 * deviation)), ...] *= (252,134,102)  # fc8666
                tda[np.logical_and(tensor > (central_tendency - 2 * deviation), tensor <= (central_tendency - 1 * deviation)), ...] *= (252,179,152)  # fcb398
                tda[np.logical_and(tensor > (central_tendency - 1 * deviation), tensor <= (central_tendency)), ...] *= (254,220,205)  # fedccd

                # Positive SD Values. This uses a discrete "Greens" color ramp, where darker greens represent more positive SD values.
                tda[np.logical_and(tensor > (central_tendency + 1 * deviation), tensor <= (central_tendency)), ...] *= (226,244,221)  # e2f4dd
                tda[np.logical_and(tensor > (central_tendency + 2 * deviation), tensor <= (central_tendency + 1 * deviation)), ...] *= (191,230,185)  # bfe6b9
                tda[np.logical_and(tensor > (central_tendency + 3 * deviation), tensor <= (central_tendency + 2 * deviation)), ...] *= (148,211,144)  # 94d390
                tda[np.logical_and(tensor > (central_tendency + 4 * deviation), tensor <= (central_tendency + 3 * deviation)), ...] *= (96,186,108)  # 60ba6c
                tda[np.logical_and(tensor > (central_tendency + 5 * deviation), tensor <= (central_tendency + 4 * deviation)), ...] *= (50,155,81)  # 329b51
                tda[np.logical_and(tensor > (central_tendency + 6 * deviation), tensor <= (central_tendency + 5 * deviation)), ...] *= (13,120,53)  # 0d7835
                tda[tensor >= (central_tendency + 6 * deviation), ...] *= (0,68,27)  # 00441b

            case "continuous":  # Continuous Colors
                tda[tensor <= sdn_thresh, ...] *= CFG_NEG_SCALE
                tda[tensor >= sdp_thresh, ...] *= CFG_POS_SCALE
                tda[np.logical_and(tensor > sdn_thresh, tensor < sdp_thresh), ...] *= CFG_MID_SCALE

            case _:
                raise ValueError("Unknown color ramp type")

        return Image.fromarray(tda.astype(np.uint8), "RGB")


    def _extract_tensor_from_model(self) -> npt.NDArray[np.float32]:
        """
        Extracts a tensor from a given model file based on the tensor name.

        Args:
            model_file (str): Path to the model file. Can be a GGUF or PyTorch model.

        Returns:
            npt.NDArray[np.float32]: The extracted tensor as a NumPy array.

        Raises:
            ValueError: If the model type is unknown or if the tensor extraction fails.
            NotImplementedError: If a Stable Diffusion model is provided (currently unsupported).
        """
        # # Initialize the model
        # match model_file.split('.')[-1].lower():
        #     case "gguf":
        #         model = GGUFModel(model_file)
        #     case "pth":
        #         model = TorchModel(model_file)
        #     case "stable_diffusion":
        #         raise NotImplementedError("Stable Diffusion models are not yet supported.")
        #     case _:
        #         raise ValueError("Unknown Model Type")

        # Validate and retrieve the tensor
        is_valid, error_message = self.model.valid(self.tensor_name)
        if not is_valid:
            raise ValueError(f"Error extracting tensor from {self.model}: {error_message}")
        else: # If it's a valid tensor, cast it as an NDArray[Float32]
            return self.model.get_as_f32(self.tensor_name)


    def set_output_path_for_this_image_of_(self, tk: str) -> None:
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
        if "/" in tk:
            raise ValueError("Bad tensor name")

        if self.output_path is not None:
            if len(self.names) > 1:
                filepath = self.output_path.parent
                filename = self.output_path.name
                self.output_path = filepath / f"{tk}.{filename}"
        else:
            if len(self.names) > 1:
                filepath = self.output_path.parent
                filename = self.output_path.name
                self.output_path = filepath / f"{tk}.{filename}"
        return


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
        logger.info(f"Matching tensors: {', '.join(repr(n) for n in self.names)}")

        for tk in self.names:
            tensor = self._extract_tensor_from_model()

            if not self.match_1d and len(tensor.shape) == 1:
                continue

            self.reshape_1d_tensor_into_2d_if_desired(tensor)

            if self.mode == "values-as-is":
                return write_array_to_geotiff(tensor, self.output_path)

            central_tendency, deviation = self.return_central_tendency_and_deviation_metrics(tensor)

            self.set_output_path_for_this_image_of_(tk)

            logger.info(f"Processing tensor {tk!r} (type:{self.model.get_type_name(tk)}, shape:{tensor.shape})",)

            self.heatmap_image = img = self.make_image_of_(tensor, central_tendency, deviation)

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
                logger.info("Displaying to screen...")

                if self.output_path is not None:
                    subprocess.call((self.show_with, self.output_path))  # noqa: S603
                else:
                    with tempfile.NamedTemporaryFile(suffix=".png") as fp:
                        img.save(fp, format="png")
                        fp.flush()
                        subprocess.call((self.show_with, fp.name))  # noqa: S603


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tensor to image converter for LLM models (GGUF and PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """\
            Information on output modes:
              devs-*:
                overall: Calculates the standard deviation deviation from the mean.
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
        overall: Calculate the mean and standard deviation over the entire tensor.
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
