from typing import Any, Iterable


import numpy as np
import numpy.typing as npt


class _Quantized:
    """
    Abstract class for model quantizations.
    NOTE: Since model classes are implemented using classmethods, this class is technically not abstract.
    However, we treat it as such for typing/standardization purposes.
    """
    def __init__(self, dtype: np.dtype[Any], block_size: int) -> None:
        self.dtype = dtype
        self.block_size = block_size

    def quantize(self, arr: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """
        Quantize a float32 array to uint8.

        This method should be implemented by subclasses to provide specific quantization logic.

        Parameters:
            arr (npt.NDArray[np.float32]): The input array to be quantized.

        Returns:
            npt.NDArray[np.uint8]: The quantized array.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Ohno")

    def dequantize(self, arr: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """
        Dequantize a uint8 array back to float32.

        This method should be implemented by subclasses to provide specific dequantization logic.

        Parameters:
            arr (npt.NDArray[np.uint8]): The input array to be dequantized.

        Returns:
            npt.NDArray[np.float32]: The dequantized array.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Ohno")


class Quantized_Q8_0(_Quantized):  # noqa: N801
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
