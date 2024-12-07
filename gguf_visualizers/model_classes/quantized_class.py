from typing import Any, Iterable


import numpy as np
import numpy.typing as npt

from logger.logger import Logger
logger = Logger(logger_name=__name__)

from gguf.quants import Q8_0

class _Quantized:
    """
    Abstract class for model quantizations.
    NOTE: Since model classes are implemented using classmethods, this class is technically not abstract.
    However, we treat it as such for typing/standardization purposes.
    """
    dtype: np.dtype[Any]
    block_size: int

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

            # Find the maximum absolute value in each block, divided by 127
            d = abs(blocks).max(axis=1) / np.float32(127)

            # Divide each block by its scaling factor and round
            with np.errstate(divide="ignore"):
                qs = (blocks / d[:, None]).round()

            # Handle blocks that are all zeros
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
        #return (blocks["d"][:, None] * np.float32(blocks["qs"])).flatten()

        # Unoptimized hack-around for broadcasting errors.
        # TODO This is trashy and gross. Learn numpy and fix it!
        # It also needs to be verified.
        results = []
        for block in blocks:

            block_results = []
            for scalar, weights_array in block:
                # Multiply the scaling factor with each element in the weights array
                scaled_arr = np.array(np.float32(weights_array)) * scalar
                block_results.append(scaled_arr)
            results.append(block_results)
        results = np.array(results)

        #logger.debug(f"results: {results}")
        logger.debug(f"results shape: {results.shape}",f=True)

        results = results.flatten()
        #logger.debug(f"results: {results}")
        logger.debug(f"results shape: {results.shape}",f=True)
        return results


        # # # Reshape d to match the broadcasting requirements
        # # d_expanded = blocks["d"][:, None]  # This should give (4096, 1, 344) MAXIMUM ABSOLUTE DEVIATION
        # # qs_data = np.float32(blocks["qs"])    # This is (4096, 344, 32) # Multiply them all together and you get the total values in the tensor.



        # # # Print shapes for debugging
        # # logger.debug(f"d_expanded shape: {d_expanded.shape}")
        # # logger.debug(f"qs_data shape: {qs_data.shape}")

        # # logger.debug(f"d_expanded:\n{d_expanded}",f=True)
        # # logger.debug(f"qs_data:\n{qs_data}",f=True)
        
        # # # Perform the multiplication with correct broadcasting
        # # results = d_expanded * qs_data
        # # logger.debug(f"result: {result}",f=True,t=30)

        # return results.flatten()
        # # 
