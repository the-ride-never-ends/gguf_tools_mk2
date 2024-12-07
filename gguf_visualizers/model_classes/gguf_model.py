#!/usr/bin/env python3
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
from typing import Iterable


import numpy as np
import numpy.typing as npt


from .model_abstract_class import Model
from .quantized_class import Quantized_Q8_0


from logger.logger import Logger
logger = Logger(logger_name=__name__)


try:
    from gguf.constants import GGMLQuantizationType
    from gguf.gguf_reader import GGUFReader, ReaderTensor
except ImportError as e:
    logger.error(f"Could not load GGMLQuantizationType, GGUFReader, ReaderTensor: {e}")
    raise e


class GGUFModel(Model):
    def __init__(self, filename: Path | str) -> None:
        try:
            import gguf
        except ImportError as e:
            logger.error(f"Loading GGUF models requires the gguf Python model: {e}")
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
        logger.debug(f"tensor: {tensor}")
        logger.debug(f"tensor.data: {tensor.data}")
        if tensor.tensor_type == self.gguf.GGMLQuantizationType.F16:
            return tensor.data.view(dtype=np.float32)
        if tensor.tensor_type == self.gguf.GGMLQuantizationType.F32:
            return tensor.data
        if tensor.tensor_type == self.gguf.GGMLQuantizationType.Q8_0:
            return Quantized_Q8_0.dequantize(tensor.data).reshape(tensor.shape)
        raise ValueError("Unhandled tensor type")

    def get_type_name(self, key: str) -> str:
        return self.tensors[key].tensor_type.name
