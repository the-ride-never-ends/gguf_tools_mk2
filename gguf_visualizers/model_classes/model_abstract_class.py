from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import numpy.typing as npt


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