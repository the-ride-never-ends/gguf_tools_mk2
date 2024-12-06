from datetime import datetime
from typing import Callable


import numpy as np
import numpy.typing as npt


from logger.logger import Logger
logger = Logger(logger_name=__name__)


def _check_if_array_was_normalized_correctly(
        diff_array: npt.NDArray[np.float32],
        tensor1: npt.NDArray[np.float32],
        tensor2: npt.NDArray[np.float32]
        ) -> npt.NDArray[np.float32]:
    """
    Check if the difference array is normalized correctly and attempt to normalize it if not.

    This function checks if the input difference array is properly normalized. If not, it attempts
    to normalize it by increasing precision and, if necessary, scaling the input tensors.

    Args:
        diff_array (npt.NDArray[np.float32]): The initial difference array to check.
        tensor1 (npt.NDArray[np.float32]): The first input tensor.
        tensor2 (npt.NDArray[np.float32]): The second input tensor.

    Returns:
        npt.NDArray[np.float32]: The normalized difference array.

    Raises:
        ValueError: If the array cannot be normalized even after increasing precision and scaling.

    Notes:
        - The function first checks if the difference array has uniform values.
        - If uniform, it increases precision to float64 and attempts normalization.
        - If still uniform, it scales the input tensors by a factor of 100 and tries again.
    """
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