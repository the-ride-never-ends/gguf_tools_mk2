#!/usr/bin/env python3
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
from typing import Iterable


import numpy as np
import numpy.typing as npt


from .model_abstract_class import Model


from logger.logger import Logger
logger = Logger(logger_name=__name__)


class TorchModel(Model):

    def __init__(self, filename: Path | str) -> None:

        try:
            import torch
        except ImportError:
            logger.error("! Loading PyTorch models requires the Torch Python model")
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