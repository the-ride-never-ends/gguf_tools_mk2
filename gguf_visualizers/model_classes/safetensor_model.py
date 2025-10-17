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


class SafetensorModel(Model):

    def __init__(self, filename: Path | str) -> None:

        try:
            from safetensors import safe_open
        except ImportError:
            logger.error("! Loading Safetensor models requires the safetensors Python module")
            sys.exit(1)

        logger.info(f"* Loading Safetensor model: {filename}")
        self.filename = filename
        self.safe_open = safe_open
        
        # Load tensor names and metadata
        with safe_open(filename, framework="numpy", device="cpu") as f:
            self.tensor_keys = list(f.keys())
            self.metadata = f.metadata()

    def tensor_names(self) -> Iterable[str]:
        return self.tensor_keys

    def valid(self, key: str) -> tuple[bool, None | str]:
        if key not in self.tensor_keys:
            return (False, "Tensor not found")
        
        # Open file to check tensor properties
        with self.safe_open(self.filename, framework="numpy", device="cpu") as f:
            tensor = f.get_tensor(key)
            
            # Check dtype
            if tensor.dtype not in (np.float32, np.float16, np.int8, np.int16, np.int32):
                return (False, "Unhandled type")
            
            # Check dimensions
            if len(tensor.shape) > 2:
                return (False, "Unhandled dimensions")
                
        return (True, "OK")

    def get_as_f32(self, key: str) -> npt.NDArray[np.float32]:
        with self.safe_open(self.filename, framework="numpy", device="cpu") as f:
            tensor = f.get_tensor(key)
            
            # Convert to float32 if needed
            if tensor.dtype != np.float32:
                return tensor.astype(np.float32)
            return tensor

    def get_type_name(self, key: str) -> str:
        with self.safe_open(self.filename, framework="numpy", device="cpu") as f:
            tensor = f.get_tensor(key)
            return str(tensor.dtype)
