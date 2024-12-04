import argparse
import os
from pathlib import Path
import re
import sys
from textwrap import dedent


import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import numpy.typing as npt
from PIL import Image


from gguf_visualizers.gguf_tensor_to_image import GGUFModel, TorchModel, Model


from config.config import OUTPUT_FOLDER
from logger.logger import Logger
logger = Logger(logger_name=__name__)


def _scale_tensor(
        tensor: npt.NDArray[np.float32],
        t_min: int|float = 0,
        t_max: int|float = 1000,
        ) -> tuple[npt.NDArray[np.float32], float]:
    """
    Scale the input tensor to a range between 0 and 1000.

    This function applies min-max normalization to the input tensor and then
    scales it to the range [t_max, t_min]. It also returns the scale factor used.

    Args:
        tensor (npt.NDArray[np.float32]): The input tensor to be scaled.
        t_min: The minimum of the range of the desired target scaling.

    Returns:
        tuple[npt.NDArray[np.float32], float]: A tuple containing:
            - The scaled tensor with values between 0 and 1000.
            - The scale factor used (always 1000 in this implementation).

    Note:
        The formula used is based on the min-max normalization technique:
        scaled_value = ((value - r_min) / (r_max - r_min)) * (new_max - new_min) + new_min
        where new_min = 0 and new_max = 1000.

    Reference:
        Formula adapted from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    """
    r_min, r_max = tensor.min(), tensor.max()
    scaled_tensor = ((tensor - r_min) / (r_max - r_min)) * (t_max - t_min) + t_min
    return scaled_tensor


def _have_then_same_file_extension(*file_names: str | argparse.Namespace) -> bool:
    """
    Check if all given file names have the same file extension.

    This function compares the file extensions of all provided file names.
    It supports both string file names and argparse.Namespace objects.

    Args:
        *file_names (str | argparse.Namespace): Variable number of file names or argparse.Namespace objects.

    Returns:
        bool: True if all file names have the same extension, False otherwise.

    Raises:
        ValueError: If fewer than two file names are provided.

    Example:
        >>> _have_then_same_file_extension('file1.txt', 'file2.txt', 'file3.txt')
        True
        >>> _have_then_same_file_extension('file1.txt', 'file2.jpg', 'file3.txt')
        False
    """
    if len(file_names) < 2:
        raise ValueError("At least two files are required for comparison.")

    # Convert all inputs to strings
    file_str_list = [str(file) for file in file_names]

    # Regex pattern to extract file extension
    pattern: str = r'\.([^.]+)$'

    # Extracting extensions
    extensions_list = [re.search(pattern, file_str) for file_str in file_str_list]

    # Check if all extensions are found
    if not all(extensions_list):
        return False

    # Extract the actual extension strings
    ext_strs_list = [ext.group(1) for ext in extensions_list if ext]

    # Compare all extensions to the first one
    return all(ext == ext_strs_list[0] for ext in ext_strs_list)


def _extract_tensor_from_model(
        model_file: str | Path, 
        tensor_name: str
        ) -> tuple[bool, None | str, None | npt.NDArray[np.float32]]:
    """Extracts a tensor from a given model file based on the tensor name.

    Args:
        model_file: Path to the model file. Can be a GGUF or PyTorch model.
        tensor_name: Name of the tensor to extract.

    Returns:
        A tuple containing:
            - A boolean indicating success
            - An error message if unsuccessful (None if successful)
            - The extracted tensor as a NumPy array if successful (None if unsuccessful)

    Raises:
        ValueError: If the model type is unknown.
    """
    # Initialize the model
    if model_file == "gguf" or model_file.lower().endswith(".gguf"):
        model = GGUFModel(model_file)
    elif model_file == "torch" or model_file.lower().endswith(".pth"):
        model = TorchModel(model_file)
    #elif model_file1 == "stable_diffusion"
        #model1 = StableDiffusionModel(model_file1)
    else:
        raise ValueError("Unknown Model Type")

    # Validate and retrieve the tensor
    is_valid, error_message = model.valid(tensor_name)
    if not is_valid:
        return False, error_message, None
    else: # If it's a valid tensor, cast it as an NDArray[Float32]
        return True, None, model.get_as_f32(tensor_name)


def _norm_array_check(
        diff_array: npt.NDArray[np.float32], 
        tensor1: npt.NDArray[np.float32], 
        tensor2: npt.NDArray[np.float32]
        ) -> npt.NDArray[np.float32]:
    # Define scale factor
    scale_factor = 100.0

    # Get min and max values of the tensor
    min_val, max_val = diff_array.min(), diff_array.max()
    logger.info(f"* Min/Max Values in diff_array: max = {max_val}, min = {min_val}")

    if min_val == max_val:
        logger.warning("* Uniform array detected, increasing precision from float32 to float64.")

        # Calculate numerically stable means and standard deviations.
        mean1, std_dev1 = np.mean(tensor1, dtype=np.float64), np.std(tensor1, dtype=np.float64)
        mean2, std_dev2 = np.mean(tensor2, dtype=np.float64), np.std(tensor2, dtype=np.float64)

        # Normalize arrays
        normalized1 = (tensor1 - mean1) / std_dev1
        normalized2 = (tensor2 - mean2) / std_dev2

        # Calculate the difference between normalized arrays
        diff_array = normalized1 - normalized2

        # Check that the difference array is normalized correctly.
        min_val, max_val = diff_array.min(), diff_array.max()

        # If array value is not normalized correctly by increasing precision, try to scale it as well.
        if min_val == max_val:
            logger.warning(f"* Uniform array still detected, max = {max_val}, min = {min_val}")
            logger.warning(f"* Increasing precision failed to normalize arrays. Increasing precision and scaling by {scale_factor}.")

            # Calculate numerically stable means and standard deviations from the scaled tensors
            mean1, std_dev1 = np.mean(tensor1 * scale_factor, dtype=np.float64), np.std(tensor1 * scale_factor, dtype=np.float64)
            mean2, std_dev2 = np.mean(tensor2 * scale_factor, dtype=np.float64), np.std(tensor2 * scale_factor, dtype=np.float64)

            # Normalize scaled arrays
            normalized1 = (tensor1 - mean1) / std_dev1
            normalized2 = (tensor2 - mean2) / std_dev2

            # Calculate the difference between normalized arrays
            diff_array = normalized1 - normalized2

            # Check that the scaled difference array is normalized correctly.
            min_val, max_val = diff_array.min(), diff_array.max()

            if min_val == max_val:
                logger.warning(f"* Uniform array still detected, max = {max_val}, min = {min_val}")
                raise ValueError(f"Array could not be normalized by precision increase and scaling by {scale_factor}.")
            else:
                logger.info("* Array normalization successful by precision increase and scaling.")
                return diff_array
        else:
            logger.info("* Array normalization successful by precision increase.")
            return diff_array
    else:
        logger.info("* Array normalization successful.")
        return diff_array


def _direct_compare_tensors(
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
    if not isinstance(tensor1, (npt.NDArray)):
        raise TypeError(f"tensor1 is not a Numpy Array, but a {type(tensor1)}")
    if not isinstance(tensor2, (npt.NDArray)):
        raise TypeError(f"tensor2 is not a Numpy Array, but a {type(tensor2)}")

    # Check if the tensors have the same dimensions.
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensors must be the same size")

    # Compare element-wise differences into a single array for visualization
    absolute_diff_array = tensor1 - tensor2

    # Figure out whether element-wise differences are additive or subtractive, relative to the tensor in model1
    sign_of_diff = np.sign(tensor1 - tensor2)

    logger.info(f"""
        * Direct comparison:
             max diff = {absolute_diff_array.max()}, 
             min diff = {absolute_diff_array.min()}
    """)

    return absolute_diff_array


def _mean_compare_tensors(tensor1: npt.NDArray[np.float32], 
                         tensor2: npt.NDArray[np.float32], 
                         tensor_name: str, 
                         model_file1: str, 
                         model_file2: str) -> npt.NDArray[np.float32]:
    """
    Compare two tensors using mean-based normalization.

    This function normalizes the input tensors using their respective means and standard deviations,
    then computes the difference between the normalized tensors.

    Args:
        tensor1 (npt.NDArray[np.float32]): The first tensor to compare.
        tensor2 (npt.NDArray[np.float32]): The second tensor to compare.
        tensor_name (str): Name of the tensor being compared.
        model_file1 (str): Name or path of the first model file.
        model_file2 (str): Name or path of the second model file.

    Returns:
        npt.NDArray[np.float32]: The difference array between the normalized tensors.

    Raises:
        ValueError: If the input tensors are not of the same size.

    Notes:
        - The function uses mean and standard deviation for normalization.
        - Normalization is performed as: (tensor - mean) / standard_deviation
        - The function logs various statistics about the input tensors for diagnostic purposes.
        - It uses the _norm_array_check function to ensure proper normalization of the difference array.
    """
    # Check is tensors are the same dimensions.
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must be of the same size")

    # Calculate numerically stable means and standard deviations
    mean1, std_dev1 = np.mean(tensor1, dtype=np.float64), np.std(tensor1, dtype=np.float64)
    mean2, std_dev2 = np.mean(tensor2, dtype=np.float64), np.std(tensor2, dtype=np.float64)
    # Log tensor values for diagnostics
    logger.info(f"""
        * Tensor1 ({tensor_name} from {model_file1}) stats:
            mean = {mean1},
            sd = {std_dev1},
            max = {tensor1.max()},
            min = {tensor1.min()}
    """)
    logger.info(f"""
        * Tensor2 ({tensor_name} from {model_file2}) stats:
            mean = {mean2},
            sd = {std_dev2},
            max = {tensor2.max()},
            min = {tensor2.min()}
    """)

    # Normalize arrays
    normalized1 = (tensor1 - mean1) / std_dev1
    normalized2 = (tensor2 - mean2) / std_dev2

    # Calculate the difference between normalized arrays
    mean_diff_array = normalized1 - normalized2

    # Check if the array was normalized correctly.
    mean_diff_array = _norm_array_check(mean_diff_array, tensor1, tensor2)

    return mean_diff_array


def _median_compare_tensors(tensor1, tensor2, tensor_name, model_file1, model_file2) -> npt.NDArray[np.float32]:
    """
    Compare two tensors using median-based normalization.

    This function normalizes the input tensors using their respective medians and Median Absolute Deviations (MAD),
    then computes the difference between the normalized tensors.

    Args:
        tensor1 (npt.NDArray[np.float32]): The first tensor to compare.
        tensor2 (npt.NDArray[np.float32]): The second tensor to compare.
        tensor_name (str): Name of the tensor being compared.
        model_file1 (str): Name or path of the first model file.
        model_file2 (str): Name or path of the second model file.

    Returns:
        npt.NDArray[np.float32]: The difference array between the normalized tensors.

    Raises:
        ValueError: If the input tensors are not of the same size.

    Notes:
        - The function uses median and MAD for normalization, which can be more robust to outliers compared to mean-based methods.
        - Normalization is performed as: (tensor - median) / MAD
        - The function logs various statistics about the input tensors for diagnostic purposes.
    """
    # Check is tensors are the same dimensions.
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must be of the same size")

    # Calculate numerically stable medians and MADs (Median Absolute Deviation). MAD = median(|Yi – median(Yi)|)
    logger.info("Calculating stable medians and MADs (Median Absolute Deviation) for tensor1.")
    median1, mad_1 = np.median(tensor1, dtype=np.float64), np.median(np.abs(tensor1 - np.median(tensor1)))
    logger.info("Calculating stable medians and MADs (Median Absolute Deviation) for tensor2.")
    median2, mad_2 = np.median(tensor2, dtype=np.float64), np.median(np.abs(tensor2 - np.median(tensor2)))

    logger.info(f"""
        * Tensor1 ({tensor_name} from {model_file1}) stats:
            median = {median1},
            MAD = {mad_1}
            max = {np.max(tensor1)},
            min = {np.min(tensor1)}
    """,f=True)
    logger.info(f"""
        * Tensor2 ({tensor_name} from {model_file2}) stats:
            median = {median2},
            MAD = {mad_2}
            max = {np.max(tensor2)},
            min = {np.min(tensor2)}
    """,f=True)

    logger.info("Normalizing arrays by median...")
    normalized1 = (tensor1 - median1) / mad_1
    normalized2 = (tensor2 - median2) / mad_2

    logger.info("Calculating the difference between normalized arrays...")
    median_diff_array = normalized1 - normalized2
    logger.info(f"""
        * Median comparison:
            median_diff = {median_diff_array}
    """,f=True) # TODO Figure out what the hell mad_diff meant. # mad_diff = {mad_diff}

    return median_diff_array


def _covert_comparison_result_to_a_heatmap_image(
        color_mode: str, 
        diff_array: npt.NDArray[np.float32],
        ) -> Image:

    # Logging initial array info
    logger.info(f"* Input diff_array shape: {diff_array.shape}, dtype: {diff_array.dtype}")

    logger.info("Checking if array is 2D and numerical...")
    if len(diff_array.shape) != 2 or not np.issubdtype(diff_array.dtype, np.number):
        raise ValueError("Input diff_array must be a 2D numerical array")

    logger.info(f"""
        * Input diff_array stats (in SD): 
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
            bounds = np.linspace(
                diff_array.min(), 
                diff_array.max(), 
                n_bins + 1
            )
            norm = plt.Normalize(
                vmin=diff_array.min(), 
                vmax=diff_array.max()
            )
            colormap_output = cmap(norm(diff_array))

        case _:
            logger.warning("Unknown color mode. Defaulting to grayscale...")
    logger.debug(f"* Colormap output shape: {colormap_output.shape}")


    logger.info("Ensuring correct shape for colormap output...")
    if colormap_output.ndim == 3 and colormap_output.shape[2] in [3, 4]:
        logger.info(f"* Alpha channel present. Discarding...")
        heatmap_array = colormap_output[..., :3]
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
            logger.warning("Unknown color mode. Defaulting to grayscale...")

    return heatmap_image


from datetime import datetime
def _right_now() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def comfyui_node():
    pass

@comfyui_node
def image_diff_heatmapper_mk2_comfy_ui_node(
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
        run = ImageDiffHeatMapperMk2(
            model_file1=model_file1, 
            model_file2=model_file2, 
            tensor_name=tensor_name,
            comparison_type=comparison_type,
            color_mode=color_mode,
            output_name=output_name
            )
        run.image_diff_heatmapper_mk2()

        return {"ui": {"images": [run.heatmap_image]}}


class ImageDiffHeatMapperMk2:

    SUPPORTED_IMAGE_TYPES = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    def __init__(self, args: argparse.Namespace = None, **kwargs):

        self.model_file1: str = args.model_file1 or kwargs.pop("model_file1")
        self.model_file2: str = args.model_file2 or kwargs.pop("model_file2")

        if self.model_file1 is None or self.model_file2 is None:
            raise ValueError("Both model_file1 and model_file2 must be provided.")
        if not os.path.exists(self.model_file1) or not os.path.exists(self.model_file2):
            raise FileNotFoundError("One or both model files not found under the given paths.")
        
        # Check if the models have the same ending prefix e.g. gguf, pth, etc.
        if not _have_then_same_file_extension(self.model_file1, self.model_file2):
            raise ValueError(f"Model prefixes do not match\n{os.path.basename(self.model_file1)}\n{os.path.basename(self.model_file2)}")


        self.tensor_name: str = args.tensor_name or kwargs.pop("tensor_name", "blk.2.ffn_down.weight")
        if self.tensor_name is None or self.tensor_name == "":
            raise ValueError("Tensor name cannot be empty.")


        self.comparison_type: str = args.comparison_type or kwargs.pop("comparison_type", "mean")
        self.color_mode: str = args.color_mode or kwargs.pop("color_mode", "grayscale")

        self.output_name = args.output_name or kwargs.pop(
            "output_name", 
            f"diff_heatmap_{os.path.basename(self.model_file1)}_and_{os.path.basename(self.model_file1)}_{_right_now()}.png"
        )
        self.output_name: str = os.path.join(OUTPUT_FOLDER, self.output_name)
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
            - It uses the _norm_array_check function to ensure proper normalization of the difference array.
        """
        # Check is tensors are the same dimensions.
        if self.tensor1.shape != self.tensor2.shape:
            raise ValueError("Tensors must be of the same size")
        
        tensor_list = []
        for i, tensor in enumerate([self.tensor1, self.tensor1], start=1):
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

        # Check if the array was normalized correctly.
        mean_diff_array = _norm_array_check(mean_diff_array, self.tensor1, self.tensor1)

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


    def image_diff_heatmapper_mk2(self) -> None:

        # Log basic stats for both tensors
        for i in [1, 2]:
            tensor = getattr(self, f"tensor{i}")
            logger.info(f"Tensor{i} ({self.tensor_name} from {getattr(self, f'model_file{i}')}) stats: max = {np.max(tensor):.4f}, min = {np.min(tensor):.4f}",f=True)

        logger.info("Checking for identical tensors...")
        if np.array_equal(self.tensor1, self.tensor2):
            logger.warning("* The input tensors are identical, differences will be zero.")

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


def image_diff_heatmapper_mk2(args: argparse.Namespace):
    # TODO add Stable Diffusion model support
    # Set argument variables
    #torch.svd(weights)
    color_mode = args.color_mode
    model_file1 = args.model_file1
    model_file2 = args.model_file2
    comparison_type = args.comparison_type
    tensor_name = args.tensor_name
    output_path = args.output_path
    model_file1: Model
    model_file2: Model

    # Regex pattern to check if the models have the same ending prefix e.g. gguf, pth, etc.
    if not _have_then_same_file_extension(model_file1, model_file1):
        raise ValueError("Model Prefixes Do Not Match")


    logger.info(f"Extracting tensor {tensor_name} from model {model_file1}...")
    success1, error1, tensor1 = _extract_tensor_from_model(model_file1, tensor_name)
    if not success1:
        raise ValueError(f"Error extracting tensor from {model_file1}: {error1}")

    logger.info(f"Extracting tensor {tensor_name} from model {model_file2}...")
    success2, error2, tensor2 = _extract_tensor_from_model(model_file2, tensor_name)
    if not success2:
        raise ValueError(f"Error extracting tensor from {model_file2}: {error2}")


    logger.info(f"""
        * Tensor1 ({tensor_name} from {model_file1}) stats: 
            max = {np.max(tensor1)}, 
            min = {np.min(tensor1)}
    """,f=True)
    logger.info(f"""
        * Tensor2 ({tensor_name} from {model_file2}) stats: 
            max = {np.max(tensor2)}, 
            min = {np.min(tensor2)}
    """,f=True)


    logger.info("Checking for identical tensors...")
    if np.array_equal(tensor1, tensor2):
        logger.warning("* The input tensors are identical, differences will be zero.")


    # Perform comparison based on specified type
    match comparison_type:
        case 'absolute':
            logger.info("Performing absolute comparison between tensors...")
            difference_comparison_result = _direct_compare_tensors(tensor1, tensor2)
        case 'mean':
            logger.info("Performing comparison of means between tensors...")
            difference_comparison_result = _mean_compare_tensors(tensor1, tensor2, tensor_name, model_file1, model_file2)
        case 'median':
            logger.info("Performing comparison of medians between tensors...")
            difference_comparison_result = _median_compare_tensors(tensor1, tensor2, tensor_name, model_file1, model_file2)
        case _:
            raise ValueError("Unknown Comparison Type")


    # Convert comparison result to image
    image = _covert_comparison_result_to_a_heatmap_image(color_mode, difference_comparison_result)


    if image is not None:
        output_file_path = str(output_path)  # Convert Path to string

        if not output_file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            raise ValueError(f"Invalid file extension for output image: {output_file_path}")

        try:
            logger.info(f"* Saving to: {output_file_path}")
            image.save(output_file_path)
        except Exception as e:
            logger.error(f"* Error saving the image: {e}")


def main() -> None:
    # TODO Implement color_ramp_type argument
    parser = argparse.ArgumentParser(
        description="Produces heatmaps of differences in tensor values for LLM models (GGUF and PyTorch)",
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
    if len(sys.argv) != 7:
        logger.error("Usage: python image_diff_heatmapper_mk2.py <model_file1> <model_file2> <tensor_name> --comparison_type=<comparison_type> --color_mode=<color_mode> --output_path=<output_path>")
        sys.exit(1)

    args = parser.parse_args()
    image_diff_heatmapper_mk2(args)

if __name__ == "__main__":
    logger.info("* Starting image_diff_heatmapper_mk2 program...")
    main()