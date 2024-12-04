#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable, Protocol

import numpy as np
import numpy.typing as npt

from logger.logger import Logger
logger = Logger(logger_name=__name__)


try:
    from PIL import Image
except ImportError:
    logger.error("This script requires Pillow installed. Example: pip install pillow")
    sys.exit(1)

try:
    from gguf.constants import GGMLQuantizationType
    from gguf.gguf_reader import GGUFReader, ReaderTensor
except ImportError:
    pass

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


# TODO This class appears to be deprecated.
class Quantized:
    def __init__(self, dtype: np.dtype[Any], block_size: int) -> None:
        self.dtype = dtype
        self.block_size = block_size

    def quantize(self, arr: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        raise NotImplementedError("Ohno")

    def dequantize(self, arr: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        raise NotImplementedError("Ohno")


class Quantized_Q8_0(Quantized):  # noqa: N801
    block_size = 32
    dtype = np.dtype([("d", "f2"), ("qs", "i1", (block_size,))])

    # Mini Q8_0 quantization in Python!
    @classmethod
    def quantize(cls, arr: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        n_blocks = arr.size // cls.block_size
        blocks = arr.reshape((n_blocks, cls.block_size))

        # Much faster implementation of block quantization contributed by @Cebtenzzre
        def quantize_blocks(blocks: npt.NDArray[Any]) -> Iterable[tuple[Any, Any]]:
            d = abs(blocks).max(axis=1) / np.float32(127)

            with np.errstate(divide="ignore"):
                qs = (blocks / d[:, None]).round()

            qs[d == 0] = 0
            yield from zip(d, qs)

        return np.fromiter(
            quantize_blocks(blocks),
            count=n_blocks,
            dtype=cls.dtype,
        )

    @classmethod
    def dequantize(
        cls,
        arr: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.float32]:
        blocks = arr.view(dtype=cls.dtype)
        return (blocks["d"][:, None] * np.float32(blocks["qs"])).flatten()


class Model(Protocol):
    def __init__(self, filename: Path | str) -> None:
        pass

    def tensor_names(self) -> Iterable[str]:
        pass

    def valid(self, key: str) -> tuple[bool, None | str]:
        pass

    def get_as_f32(self, key: str) -> npt.NDArray[np.float32]:
        pass

    def get_type_name(self, key: str) -> str:
        pass


class GGUFModel(Model):
    def __init__(self, filename: Path | str) -> None:
        try:
            import gguf
        except ImportError:
            logger.error(
                "! Loading GGUF models requires the gguf Python model",
                file=sys.stderr,
            )
            sys.exit(1)
        logger.info(f"* Loading GGUF model: {filename}")
        self.gguf = gguf
        self.reader = gguf.GGUFReader(filename, "r")
        self.tensors: OrderedDict[str, ReaderTensor] = OrderedDict(
            (tensor.name, tensor) for tensor in self.reader.tensors
        )

    def tensor_names(self) -> Iterable[str]:
        return self.tensors.keys()

    def valid(self, key: str) -> tuple[bool, None | str]:
        tensor = self.tensors.get(key)
        if tensor is None:
            return (False, "Tensor not found")
        if tensor.tensor_type not in (
            self.gguf.GGMLQuantizationType.F16,
            self.gguf.GGMLQuantizationType.F32,
            self.gguf.GGMLQuantizationType.Q8_0,
        ):
            return (False, "Unhandled type")
        if len(tensor.shape) > 2:
            return (False, "Unhandled dimensions")
        return (True, "OK")

    def get_as_f32(self, key: str) -> npt.NDArray[np.float32]:
        tensor = self.tensors[key]
        if tensor.tensor_type == self.gguf.GGMLQuantizationType.F16:
            return tensor.data.view(dtype=np.float32)
        if tensor.tensor_type == self.gguf.GGMLQuantizationType.F32:
            return tensor.data
        if tensor.tensor_type == self.gguf.GGMLQuantizationType.Q8_0:
            return Quantized_Q8_0.dequantize(tensor.data).reshape(tensor.shape)
        raise ValueError("Unhandled tensor type")

    def get_type_name(self, key: str) -> str:
        return self.tensors[key].tensor_type.name


class TorchModel(Model):
    def __init__(self, filename: Path | str) -> None:
        try:
            import torch
        except ImportError:
            logger.error(
                "! Loading PyTorch models requires the Torch Python model",
                file=sys.stderr,
            )
            sys.exit(1)

        logger.info(f"* Loading PyTorch model: {filename}")
        self.torch = torch
        self.model = torch.load(filename, map_location="cpu", mmap=True)
        self.tensors: OrderedDict[str, None] = OrderedDict(
            (tensor_name, tensor.squeeze())
            for tensor_name, tensor in self.model.items()
        )

    def tensor_names(self) -> Iterable[str]:
        return self.tensors.keys()

    def valid(self, key: str) -> tuple[bool, None | str]:
        tensor = self.tensors.get(key)
        if tensor is None:
            return (False, "Tensor not found")
        if tensor.dtype not in (
            self.torch.float32,
            self.torch.float16,
            self.torch.bfloat16,
        ):
            return (False, "Unhandled type")
        if len(tensor.shape) > 2:
            return (False, "Unhandled dimensions")
        return (True, "OK")

    def get_as_f32(self, key: str) -> npt.NDArray[np.float32]:
        return self.tensors[key].to(dtype=self.torch.float32).numpy()

    def get_type_name(self, key: str) -> str:
        return str(self.tensors[key].dtype)


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

@comfyui_node
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
    
    run = GgufTensorToImage(adjust_1d_rows=adjust_1d_rows,
                            mode=mode,
                            model=model,
                            model_type=model_type,
                            match_glob=match_glob,
                            match_regex=match_regex,
                            match_1d=match_1d,
                            output=output,
                            scale=scale,
                            show_with=show_with)
    run.gguf_tensor_to_image()

    return {"ui": {"images": [run.heatmap_image]}} 


class GgufTensorToImage:

    def __init__(self, args=None, **kwargs) -> None:

        self.adjust_1d_rows: int = args.adjust_1d_rows or kwargs.pop("adjust_1d_rows",  32)
        self.mode: str = args.mode or kwargs.pop("mode", "mean-devs-overall")
        self.model: str = args.model or kwargs.pop("model")
        if self.model is None:
            raise ValueError("No model specified in MakeImage class parameters")

        self.model_type: str = args.model_type or kwargs.pop("model_type")
        self.match_glob: bool = args.match_glob or kwargs.pop("match_glob", True)
        self.match_regex: bool = args.match_regex or kwargs.pop("match_regex", True)
        self.match_1d: bool = args.match_1d or kwargs.pop("match_1d", True)
        self.output: Path = args.output or kwargs.pop("output")
        self.scale: float = args.scale or kwargs.pop("scale", 1.0)
        self.show_with: str = args.show_with or kwargs.pop("show_with", None)
        self.heatmap_image = None


        # Cast the model as the appropriate type.
        self.model: Model
        if self.model_type == "gguf" or self.model.lower().endswith(".gguf"):
            self.model = GGUFModel(self.model)
        elif self.model_type == "torch" or self.model.lower().endswith(".pth"):
            self.model = TorchModel(self.model)
        else:
            raise ValueError("Can't handle this type of model, sorry")

        self.tensor_name: str = args.tensor or kwargs.pop("tensor", "blk.2.ffn_down.weight")
        self.names: list[str] = self.get_tensor_names()


    def get_tensor_names(self) -> list[str]:
        if self.match_glob:
            self.names = [ # Use fnmatch to find tensor names that match the given glob patterns
                name for name in self.model.tensor_names()
                if any(fnmatch.fnmatchcase(name, pat) for pat in self.tensor_name)
            ]
        elif self.match_regex:
            res = [re.compile(r) for r in self.tensor_name] # Compile the regex patterns
            self.names = [  # Find tensor names that match any of the compiled regex patterns
                name for name in self.model.tensor_names() if any(r.search(name) for r in res)
            ]
        else:
            # Use the tensor names provided directly, but only if they are valid
            self.names = [name for name in self.tensor_name if self.model.valid(name)[0]]
        return


    def reshape_1d_tensor_into_2d_if_desired(self, tensor: npt.NDArray[np.float32]) -> tuple[float,float]:
        # Check if the tensor is 1-dimensional TODO Check to see if this comment is hallucinated.
        if len(tensor.shape) == 1:
            # If adjust_1d_rows argument is provided
            if self.adjust_1d_rows is not None:
                # Reshape the 1D tensor into a 2D array with specified number of rows
                # The number of columns is calculated by dividing the total elements by the number of rows.
                tensor = tensor.reshape((self.adjust_1d_rows, tensor.shape[0] // self.adjust_1d_rows))
            else:
                # If adjust_1d_rows is not provided, add an extra dimension to make it 2D
                # This creates a 2D array with 1 row and the original data as columns
                tensor = tensor[None, :]
        return tensor


    def return_central_tendency_and_deviation_metrics(self, tensor: npt.NDArray[np.float32]) -> tuple[float,float]:
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


    def make_image_of_(self, tensor: npt.NDArray[np.float32]) -> Image:

        self.reshape_1d_tensor_into_2d_if_desired(tensor)

        central_tendency, deviation = self.return_central_tendency_and_deviation_metrics(tensor)

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


    def extract_tensor_from_model(self, model_file: str) -> npt.NDArray[np.float32]:
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


    def set_output_path_for_this_image_of_(self, tk) -> None:

        if "/" in tk:
            raise ValueError("Bad tensor name")

        if self.output is not None:
            if len(self.names) > 1:
                filepath = self.output.parent
                filename = self.output.name
                self.output = filepath / f"{tk}.{filename}"
        else:
            if len(self.names) > 1:
                self.output = self.output / f"{tk}.{filename}"
        return


    def gguf_tensor_to_image(self) -> None:
        logger.info(f"Matching tensors: {', '.join(repr(n) for n in self.names)}")

        for tk in self.names:
            tensor = self.model.get_as_f32(tk)

            if not self.match_1d and len(tensor.shape) == 1:
                continue

            self.set_output_path_for_this_image_of_(tk)

            logger.info(f"Processing tensor {tk!r} (type:{self.model.get_type_name(tk)}, shape:{tensor.shape})",)

            self.heatmap_image = img = self.make_image_of_(tensor)

            if self.scale != 1.0:
                self.heatmap_image = img = img.resize(
                    (
                        max(1, int(img.width * self.scale)),
                        max(1, int(img.height * self.scale)),
                    ),
                    resample=Image.Resampling.LANCZOS,
                )

            if self.output is not None:
                logger.info(f"Saving to '{self.output}'...")
                img.save(self.output)

            if self.show_with:
                logger.info("Displaying to screen...")

                if self.output is not None:
                    subprocess.call((self.show_with, self.output))  # noqa: S603
                else:
                    with tempfile.NamedTemporaryFile(suffix=".png") as fp:
                        img.save(fp, format="png")
                        fp.flush()
                        subprocess.call((self.show_with, fp.name))  # noqa: S603


def make_image(args: argparse.Namespace, td: npt.NDArray[np.float32]) -> Image.Image:

    # Check if the tensor is 1-dimensional TODO Check to see if this comment is hallucinated.
    if len(td.shape) == 1:
        # If adjust_1d_rows argument is provided
        if args.adjust_1d_rows is not None:
            # Reshape the 1D tensor into a 2D array with specified number of rows
            # The number of columns is calculated by dividing the total elements by the number of rows.
            td = td.reshape((args.adjust_1d_rows, td.shape[0] // args.adjust_1d_rows))
        else:
            # If adjust_1d_rows is not provided, add an extra dimension to make it 2D
            # This creates a 2D array with 1 row and the original data as columns
            td = td[None, :]

    # Calculate mean and SD based on arguments.
    match args.mode:
        case "devs-overall":
            sd = np.std(td)
            mean = np.mean(td)
        case "devs-rows":
            sd = np.std(td, axis=1)[:, None]
            mean = np.mean(td, axis=1)[:, None]
        case "devs-cols":
            sd = np.std(td, axis=0)
            mean = np.mean(td, axis=0)
        case _:
            raise ValueError("Unknown mode")

    # Map the 2D tensor data to the same range as an image 0-255.
    sdp_max = mean + CFG_SD_CLIP_THRESHOLD * sd # Set the maximum value for the positive SD threshold.
        # Set the positive and negative SD thresholds for this specific tensor.
    sdp_thresh = mean + CFG_SD_POSITIVE_THRESHOLD * sd 
    sdn_thresh = mean - CFG_SD_NEGATIVE_THRESHOLD * sd
        # Calculate the absolute difference between the tensor data and the mean.
    tda = np.minimum(np.abs(td), sdp_max).repeat(3, axis=-1).reshape((*td.shape, 3))
        # Scale that range to between 0 and 255.
    tda = 255 * ((tda - np.min(tda)) / np.ptp(tda))


    match args.color_ramp_type :
        case "discrete":  # Discrete Colors
            # Negative SD Values. This uses a discrete "Reds" color ramp, where darker reds represent more negative SD values.
            tda[td <= (mean - 6 * sd), ...] *= (103,0,13)  # 67000d
            tda[np.logical_and(td > (mean - 6 * sd), td <= (mean - 5 * sd)), ...] *= (179,18,24)  # b31218
            tda[np.logical_and(td > (mean - 5 * sd), td <= (mean - 4 * sd)), ...] *= (221,42,37)  # dd2a25
            tda[np.logical_and(td > (mean - 4 * sd), td <= (mean - 3 * sd)), ...] *= (246,87,62)  # f6573e
            tda[np.logical_and(td > (mean - 3 * sd), td <= (mean - 2 * sd)), ...] *= (252,134,102)  # fc8666
            tda[np.logical_and(td > (mean - 2 * sd), td <= (mean - 1 * sd)), ...] *= (252,179,152)  # fcb398
            tda[np.logical_and(td > (mean - 1 * sd), td <= (mean)), ...] *= (254,220,205)  # fedccd

            # Positive SD Values. This uses a discrete "Greens" color ramp, where darker greens represent more positive SD values.
            tda[np.logical_and(td > (mean + 1 * sd), td <= (mean)), ...] *= (226,244,221)  # e2f4dd
            tda[np.logical_and(td > (mean + 2 * sd), td <= (mean + 1 * sd)), ...] *= (191,230,185)  # bfe6b9
            tda[np.logical_and(td > (mean + 3 * sd), td <= (mean + 2 * sd)), ...] *= (148,211,144)  # 94d390
            tda[np.logical_and(td > (mean + 4 * sd), td <= (mean + 3 * sd)), ...] *= (96,186,108)  # 60ba6c
            tda[np.logical_and(td > (mean + 5 * sd), td <= (mean + 4 * sd)), ...] *= (50,155,81)  # 329b51
            tda[np.logical_and(td > (mean + 6 * sd), td <= (mean + 5 * sd)), ...] *= (13,120,53)  # 0d7835
            tda[td >= (mean + 6 * sd), ...] *= (0,68,27)  # 00441b

        case "continuous":  # Continuous Colors
            tda[td <= sdn_thresh, ...] *= CFG_NEG_SCALE
            tda[td >= sdp_thresh, ...] *= CFG_POS_SCALE
            tda[np.logical_and(td > sdn_thresh, td < sdp_thresh), ...] *= CFG_MID_SCALE

        case _:
            raise ValueError("Unknown color ramp type")

    return Image.fromarray(tda.astype(np.uint8), "RGB")


def gguf_tensor_to_image(args: argparse.Namespace) -> None:

    # Cast the model as the appropriate type.
    model: Model
    if args.model_type == "gguf" or args.model.lower().endswith(".gguf"):
        model = GGUFModel(args.model)
    elif args.model_type == "torch" or args.model.lower().endswith(".pth"):
        model = TorchModel(args.model)
    else:
        raise ValueError("Can't handle this type of model, sorry")
    
    if args.match_glob:
        names = [ # Use fnmatch to find tensor names that match the given glob patterns
            name for name in model.tensor_names()
            if any(fnmatch.fnmatchcase(name, pat) for pat in args.tensor)
        ]
    elif args.match_regex:
        res = [re.compile(r) for r in args.tensor] # Compile the regex patterns
        names = [  # Find tensor names that match any of the compiled regex patterns
            name for name in model.tensor_names() if any(r.search(name) for r in res)
        ]
    else:
        # Use the tensor names provided directly, but only if they are valid
        names = [name for name in args.tensor if model.valid(name)[0]]

    logger.info(f"* Matching tensors: {', '.join(repr(n) for n in names)}")
    for tk in names:
        tensor = model.get_as_f32(tk)

        if not args.match_1d and len(tensor.shape) == 1:
            continue

        type_name = model.get_type_name(tk)
        output: None | Path = None

        if "/" in tk:
            raise ValueError("Bad tensor name")

        if args.output is not None:
            if len(names) == 1:
                output = args.output
            else:
                filepath = args.output.parent
                filename = args.output.name
                output = filepath / f"{tk}.{filename}"

        logger.info(f"* Processing tensor {tk!r} (type:{type_name}, shape:{tensor.shape})",)

        img = make_image(args, tensor)
        if args.scale != 1.0:
            img = img.resize(
                (
                    max(1, int(img.width * args.scale)),
                    max(1, int(img.height * args.scale)),
                ),
                resample=Image.Resampling.LANCZOS,
            )

        if output is not None:
            logger.info(f"-  Saving to: {output}")
            img.save(output)

        if args.show_with:
            logger.info("-  Displaying to screen")

            if output is not None:
                subprocess.call((args.show_with, output))  # noqa: S603
            else:
                with tempfile.NamedTemporaryFile(suffix=".png") as fp:
                    img.save(fp, format="png")
                    fp.flush()
                    subprocess.call((args.show_with, fp.name))  # noqa: S603


def main() -> None:
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

    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    if not (args.show_with or args.output):
        logger.error("! At least one of --show or --output must be specified", file=sys.stderr)
        sys.exit(1)

    if (args.match_regex or args.match_glob) and len(args.tensor) != 1:
        logger.warning(
            "! Can only specify one tensor name (pattern) when using --match-glob or --match-regex",
            file=sys.stderr,
        )

    gguf_tensor_to_image(args)
    logger.info("\n* Done.")


if __name__ == "__main__":
    logger.info("* Starting gguf_tensor_to_image program...")
    main()
