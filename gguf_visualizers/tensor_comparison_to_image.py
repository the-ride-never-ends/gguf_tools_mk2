#!/usr/bin/env python3
"""Produce heatmaps of differences in tensor values for AI models (GGUF and PyTorch)"""
from __future__ import annotations

import argparse
import os
import sys
from textwrap import dedent
from typing import Callable, Never


import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import numpy.typing as npt
from PIL import Image


from gguf_visualizers.tensor_to_image import GGUFModel, TorchModel, Model
from .utils._have_the_same_file_extension import _have_the_same_file_extension
from .utils._check_if_array_was_normalized_correctly import _check_if_array_was_normalized_correctly
from .utils._right_now import _right_now
from .utils.find_this_file_under_this_directory_and_return_the_files_path import (
    find_this_file_under_this_directory_and_return_the_files_path
)

from .utils.write_array_to_geotiff import write_array_to_geotiff


from config.config import OUTPUT_FOLDER, INPUT_FOLDER
from logger.logger import Logger
logger = Logger(logger_name=__name__)

from config.file_specific_configs import FileSpecificConfigs
config: Callable = FileSpecificConfigs().config

MODEL_FILE_PATH1: str = config("MODEL_FILE_PATH1")
MODEL_FILE_PATH2: str = config("MODEL_FILE_PATH2")
TENSOR_NAME: str = config("TENSOR_NAME")
COMPARISON_TYPE: str = config("COMPARISON_TYPE")
COLOR_MODE: str = config("COLOR_MODE")
OUTPUT_NAME: str = config("OUTPUT_NAME")


def comfyui_node():
    pass


#@comfyui_node
def tensor_comparison_to_image_comfy_ui_node(
                                            model_file1: str, 
                                            model_file2: str, 
                                            tensor_name: str, 
                                            comparison_type: str, 
                                            color_mode: str, 
                                            output_name: str, 
                                            ) -> None:
        """
        ComfyUI node for generating a heatmap of the difference between two tensors in two models.

        Args:
            model_file1 (str): Path to the first model file. Can be a GGUF or PyTorch model.
        """
        run = TensorComparisonToImage(
            model_file1=model_file1, 
            model_file2=model_file2, 
            tensor_name=tensor_name,
            comparison_type=comparison_type,
            color_mode=color_mode,
            output_name=output_name
            )
        run.tensor_comparison_to_image()

        return {"ui": {"images": [run.heatmap_image]}}




class TensorComparisonToImage:

    SUPPORTED_IMAGE_TYPES = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.geotiff', '.tif')

    def __init__(self, **kwargs):

        model_path1 = MODEL_FILE_PATH1 or kwargs.pop("model_file1")
        model_path2 = MODEL_FILE_PATH2 or kwargs.pop("model_file2")

        self.model_file1: str = os.path.join(INPUT_FOLDER, model_path1.lstrip("/\\"))
        self.model_file2: str = os.path.join(INPUT_FOLDER, model_path2.lstrip("/\\"))
        self.type_check_model_files()

        self.tensor_name: str = TENSOR_NAME or kwargs.pop("tensor_name")
        if self.tensor_name is None or self.tensor_name == "":
            raise ValueError("Tensor name cannot be empty.")

        self.comparison_type: str = COMPARISON_TYPE or kwargs.pop("comparison_type")
        self.color_mode: str = COLOR_MODE or kwargs.pop("color_mode")

        self.output_name = OUTPUT_NAME or kwargs.pop(
            "output_name", 
            f"diff_heatmap_{os.path.basename(self.model_file1)}_and_{os.path.basename(self.model_file1)}_{_right_now()}.png"
        )
        # find_this_file_under_this_directory_and_return_the_files_path
        self.output_path: str = os.path.join(OUTPUT_FOLDER, self.output_name)

        # If the image path does not end with a file-type, default to png
        if not self.output_path.lower().endswith(self.SUPPORTED_IMAGE_TYPES):
            self.output_path = f"{self.output_path}.png"

        self.tensor1: npt.NDArray[np.float32] = None
        self.tensor2: npt.NDArray[np.float32] = None

        # Load tensors from the models.
        self.tensor1 = self._extract_tensor_from_model(self.model_file1)
        self.tensor2 = self._extract_tensor_from_model(self.model_file2)

        # Check if the tensors have the same dimensions.
        if self.tensor1.shape != self.tensor1.shape:
            raise ValueError("Tensors must be of the same dimensions.")

        self.heatmap_image: Image = None

    def type_check_model_files(self) -> Never: 
        if self.model_file1 is None or self.model_file2 is None:
            msg = f"self.model_file1: {self.model_file1}\nself.model_file2: {self.model_file2}"
            logger.error(msg)
            raise ValueError(f"Both model_file1 and model_file2 must be provided.\n{msg}")
        if not os.path.exists(self.model_file1) or not os.path.exists(self.model_file2):
            msg = f"self.model_file1: {self.model_file1}\nself.model_file2: {self.model_file2}"
            logger.error(msg)
            raise FileNotFoundError(f"One or both model files not found under the given paths.\n{msg}")
        
        # Check if the models have the same ending prefix e.g. gguf, pth, etc.
        if not _have_the_same_file_extension(self.model_file1, self.model_file2):
            raise ValueError(f"Model prefixes do not match\n{os.path.basename(self.model_file1)}\n{os.path.basename(self.model_file2)}")


    def _extract_tensor_from_model(
        self,
        model_file: str, 
        ) -> npt.NDArray[np.float32]:
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
        # Initialize the model
        match model_file.split('.')[-1].lower():
            case "gguf":
                model = GGUFModel(model_file)
            case "pth":
                model = TorchModel(model_file)
            case "stable_diffusion":
                raise NotImplementedError("Stable Diffusion models are not yet supported.")
            case _:
                raise ValueError("Unknown Model Type")

        # Validate and retrieve the tensor
        is_valid, error_message = model.valid(self.tensor_name)
        if not is_valid:
            raise ValueError(f"Error extracting tensor from {model_file}: {error_message}")
        else: # If it's a valid tensor, cast it as an NDArray[Float32]
            return model.get_as_f32(self.tensor_name)


    def _directly_compare_tensor_values(self,
        tensor1: npt.NDArray[np.float32], 
        tensor2: npt.NDArray[np.float32]
        ) -> npt.NDArray[np.float32]:
        """
        Check and normalize the difference array, increasing precision and scaling if necessary.

        This function checks if the input difference array is properly normalized. If not, it
        attempts to normalize it by increasing precision and, if needed, scaling the input tensors.

        Args:
            diff_array (npt.NDArray[np.float32]): The initial difference array to check and normalize.
            tensor1 (npt.NDArray[np.float32]): The first input tensor used for normalization if needed.
            tensor2 (npt.NDArray[np.float32]): The second input tensor used for normalization if needed.

        Returns:
            npt.NDArray[np.float32]: The normalized difference array.

        Raises:
            ValueError: If the array cannot be normalized even after increasing precision and scaling.

        Notes:
            - The function first checks if the input diff_array is uniform (min == max).
            - If uniform, it increases precision to float64 and recalculates the difference.
            - If still uniform, it applies a scaling factor of 100.0 to the input tensors and recalculates.
            - The function logs various information and warnings during the process.
        """

        # Type check the tensors.
        if not isinstance(self.tensor1, (npt.NDArray)):
            raise TypeError(f"tensor1 is not a Numpy Array, but a {type(self.tensor1)}")
        if not isinstance(self.tensor2, (npt.NDArray)):
            raise TypeError(f"tensor2 is not a Numpy Array, but a {type(self.tensor2)}")

        # Check if the tensors have the same dimensions.
        if tensor1.shape != tensor2.shape:
            raise ValueError("Input tensors must be the same size")

        # Compare element-wise differences into a single array for visualization
        absolute_diff_array = self.tensor1 - self.tensor1

        # Figure out whether element-wise differences are additive or subtractive, relative to the tensor in model1
        #sign_of_diff = np.sign(tensor1 - tensor2)

        logger.info(f"""
            * Direct comparison:
                max diff = {absolute_diff_array.max()}, 
                min diff = {absolute_diff_array.min()}
        """)

        return absolute_diff_array


    def _compare_normalized_tensor_means(self) -> npt.NDArray[np.float32]:
        """
        Compare two tensors using mean-based normalization.

        This function normalizes the input tensors using their respective means and standard deviations,
        then computes the difference between the normalized tensors.

        Returns:
            npt.NDArray[np.float32]: The difference array between the normalized tensors.

        Raises:
            ValueError: If the input tensors are not of the same size.

        Notes:
            - Normalization is performed as: (tensor - mean) / standard_deviation
            - It uses the _check_if_array_was_normalized_correctly function to ensure proper normalization of the difference array.
        """
        # Check is tensors are the same dimensions.
        if self.tensor1.shape != self.tensor2.shape:
            raise ValueError("Tensors must be of the same size")
        
        tensor_list = []
        for i, tensor in enumerate([self.tensor1, self.tensor2], start=1):
            model_name = getattr(self, f"model_file{i}")

            # Calculate numerically stable means and standard deviations
            mean, std_dev = np.mean(tensor, dtype=np.float64), np.std(tensor, dtype=np.float64)

            # Log tensor values for diagnostics
            logger.info(f"* Tensor ({self.tensor_name} from {model_name}) stats\nmean = {mean}\nsd = {std_dev}\n\nmax = {tensor.max()}\nmin = {tensor.min()}",f=True)

            # Normalize arrays to make their means directly comparable.
            normalized = (tensor - mean) / std_dev
            tensor_list.append(normalized)

        # Calculate the difference between normalized arrays.
        mean_diff_array = tensor_list[0] - tensor_list[1]

        mean_diff_array = _check_if_array_was_normalized_correctly(mean_diff_array, self.tensor1, self.tensor2)

        return mean_diff_array


    def _compared_normalized_tensor_medians(self) -> npt.NDArray[np.float32]:
        """
        Compare two tensors using median-based normalization.

        This function normalizes the input tensors using their respective medians and Median Absolute Deviations (MAD),
        then computes the difference between the normalized tensors.

        Returns:
            npt.NDArray[np.float32]: The difference array between the normalized tensors.

        Notes:
            - The function uses median and MAD for normalization, which can be more robust to outliers compared to mean-based methods.
            - Normalization is performed as: (tensor - median) / MAD
            - The function logs various statistics about the input tensors for diagnostic purposes.
        """

        normalized_arrays = []
        for i, tensor in enumerate([self.tensor1, self.tensor2], start=1):
            tensor_name = f"tensor{i}"
            model_file = getattr(self, f"model_file{i}")

            # Calculate numerically stable medians and MADs (Median Absolute Deviation). MAD = median(|Yi – median(Yi)|)
            logger.info(f"Calculating stable medians and MADs (Median Absolute Deviation) for {tensor_name}.")
            median, mad = np.median(tensor, dtype=np.float64), np.median(np.abs(tensor - np.median(tensor)))

            logger.info(f"""
                * Tensor{i} ({tensor_name} from {model_file}) stats:
                    median = {median},
                    MAD = {mad}
                    max = {np.max(tensor)},
                    min = {np.min(tensor)}
            """,f=True)
 
            logger.info("Normalizing array by median...")
            normalized = (tensor - median) / mad
            normalized_arrays.append(normalized)

        logger.info("Calculating the difference between normalized arrays...")
        median_diff_array = normalized_arrays[0] - normalized_arrays[1]
        logger.info(f"""
            * Median comparison:
                median_diff = {median_diff_array}
        """,f=True)

        return median_diff_array


    def _covert_comparison_result_to_a_heatmap_image(
            self,
            color_mode: str, 
            diff_array: npt.NDArray[np.float32], # Comparison result
            ) -> Image:

        logger.info("Checking if array is 2D and numerical...")
        if len(diff_array.shape) != 2 or not np.issubdtype(diff_array.dtype, np.number):
            raise ValueError("Input diff_array must be a 2D numerical array")

        logger.info(f"""
            Input diff_array stats (in SD): 
                max = {diff_array.max()}, 
                min = {diff_array.min()}, 
                mean = {np.mean(diff_array, dtype=np.float64)}, 
                sd = {np.std(diff_array, dtype=np.float64)}
            """,f=True)

        logger.info(f"Applying colormap '{color_mode}'...")
        match color_mode:
            case "grayscale":
                colormap_output = plt.cm.gray(diff_array)

            case "false color jet":
                colormap_output = plt.cm.jet(diff_array)

            case "false color vidiris":
                colormap_output = plt.cm.viridis(diff_array)

            case "binned coolwarm":
                # Create a custom colormap with discrete bins
                n_bins = 10  # Number of bins
                cmap = plt.cm.coolwarm
                bounds = np.linspace(diff_array.min(), diff_array.max(), n_bins + 1)
                norm = BoundaryNorm(bounds, cmap.N)
                colormap_output = cmap(norm(diff_array))
            case _:
                logger.warning("Unknown color mode. Defaulting to grayscale...")
                colormap_output = plt.cm.gray(diff_array)

        logger.debug(f"Colormap output shape: {colormap_output.shape}")

        logger.info("Ensuring correct shape for colormap output...")
        if colormap_output.ndim == 3 and colormap_output.shape[2] in [3, 4]:
            logger.info(f"Alpha channel present. Discarding...")
            heatmap_array = colormap_output[..., :3]  # Discarding alpha channel if present
        else:
            raise ValueError("Unexpected shape for color map output")

        logger.info(f"Converting to 8-bit format...")
        heatmap_array = (heatmap_array * 255).astype(np.uint8)

        logger.info(f"Converting to PIL Image...")
        if heatmap_array.ndim != 3 or heatmap_array.shape[2] != 3:
            raise ValueError("Heatmap array must be 3-dimensional with 3 channels for RGB")

        # Apply colormap
        match color_mode:
            case "grayscale":
                heatmap_image = Image.fromarray(heatmap_array, 'L')
            case "false color jet" | "false color vidiris" | "binned coolwarm":
                heatmap_image = Image.fromarray(heatmap_array, 'RGB')
            case _:
                heatmap_image = Image.fromarray(heatmap_array, 'L')

        return heatmap_image


    def tensor_comparison_to_image(self) -> None:

        # Log basic stats for both tensors
        for i in [1, 2]:
            logger.info(f"Tensor{i} ({self.tensor_name} from {getattr(self, f'model_file{i}')}) stats\nmax = {np.max(getattr(self, f'tensor{i}')):.4f}\nmin = {np.min(getattr(self, f'tensor{i}')):.4f}",f=True)

        logger.info("Checking for identical tensors...")
        if np.array_equal(self.tensor1, self.tensor2):
            logger.error("* The input tensors are identical, differences will be zero.")
            raise ValueError("Tensors must be different in order to perform meaningful comparisons.")
        else:
            logger.info("Tensors verified to not be identical.")

        # Perform tensor comparison based on specified type. Default is mean-based comparison.
        match self.comparison_type:
            case 'absolute':
                logger.info("Performing absolute comparison between tensors...")
                difference_comparison_result = self._directly_compare_tensor_values()
            case 'mean':
                logger.info("Performing comparison of means between tensors...")
                difference_comparison_result = self._compare_normalized_tensor_means()
            case 'median':
                logger.info("Performing comparison of medians between tensors...")
                difference_comparison_result = self._compared_normalized_tensor_medians()
            case _:
                logger.warning("Unknown comparison type. Defaulting to mean-based comparison...")
                difference_comparison_result = self._compare_normalized_tensor_means()

        if self.output_name.endswith((".geotiff", ".tiff", ".tif",)):
            logger.info("Saving difference_comparison_result as a geotiff file...")
            write_array_to_geotiff(difference_comparison_result, self.output_path)
        else:
            self.heatmap_image: Image = self._covert_comparison_result_to_a_heatmap_image(
                                                self.color_mode, 
                                                difference_comparison_result)

        if self.heatmap_image is not None:
            try:
                logger.info(f"Saving to '{self.output_path}'...")
                self.heatmap_image.save(self.output_path)
            except Exception as e:
                logger.error(f"Error saving the image: {e}")
        else:
            logger.error("Failed to create the image.")


def create_parser() -> argparse.ArgumentParser:
    # TODO Implement color_ramp_type argument
    parser = argparse.ArgumentParser(
        description="Produce heatmaps of differences in tensor values for LLM models (GGUF and PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """\
            Information on output modes:
              devs-*:
                overall: Calculates differences in tensor values between two models with the same foundation architecture.
                         By default, output will be a grayscale raster that has the same dimensions as the tensors.
                rows   : Same as above, except the calculation is based on rows.
                cols:  : Same as above, except the calculation is based on columns.
        """,
        ),
    )
    parser.add_argument(
        "model_file1",
        type=str,
        required=True,
        help="Filename for the first model, can be GGUF or PyTorch (if PyTorch support available)",
    )
    parser.add_argument(
        "model_file2",
        type=str,
        help="Filename for the second model, can be GGUF or PyTorch (if PyTorch support available)",
    )
    parser.add_argument(
        "tensor_name",
        type=str,
        help="Tensor name, must be from models with the same foundation architecture for the differences to be valid.",
    )
    parser.add_argument(
        "--comparison_type",
        choices=["mean", "median", "absolute"],
        default="mean",
        help="Comparison types, Default: mean",
    )
    parser.add_argument(
        "--color_mode",
        choices=["grayscale", "false color jet", "false color vidiris", "binned coolwarm"],
        default="grayscale",
        help="Color mode, Default: grayscale",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        help=f"Output file name for the heatmap. The heatmap will be saved to {OUTPUT_FOLDER}",
    )
    return parser


def parse_arguments():
    parser = create_parser()

    if len(sys.argv) != 7:
        logger.error("Usage: python tensor_comparison_to_image.py <model_file1> <model_file2> <tensor_name> --comparison_type=<comparison_type> --color_mode=<color_mode> --output_path=<output_path>")
        sys.exit(1)

    return parser.parse_args()


def main() -> None:

    logger.info("* Starting tensor_comparison_to_image program...")
    TensorComparisonToImage(parse_arguments()).tensor_comparison_to_image()
    logger.info("*\nDone.")


if __name__ == "__main__":
    main()