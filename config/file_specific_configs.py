import inspect
import os
from typing import Any

from config.utils.config._get_config import get_config

class FileSpecificConfigs:

    def __init__(self, filename=None):
        # Get the name of the file that the method is called from
        # Remove the .py from the end, then make it UPPER_CASE
        self.filename: str = filename or os.path.splitext(os.path.basename(inspect.stack()[1].filename))[0].upper()

    def config(self, constant: str) -> Any:
        return get_config(self.filename, constant)


    # def get_config(self, path: str, constant: str) -> Any | None:
    #     """
    #     Get a key from a yaml file.

    #     Args:
    #         path (str): The path to the desired key, using dot notation for nested structures.
    #         constant (str): The specific key to retrieve.

    #     Returns:
    #         Union[Any, bool]: The value of the key if found, False otherwise.

    #     Examples:
    #         >>> config("SYSTEM", "CONCURRENCY_LIMIT")
    #         2
    #         >>> config("SYSTEM", "NONEXISTENT_KEY") or 3
    #         3
    #     """
    #     keys = path + "." + constant

    #     # Split the path into individual keys
    #     keys = path.split('.') + [constant]

    #     # Traverse the nested dictionary
    #     current_data = self.data
    #     for i, key in enumerate(keys):
    #         if isinstance(current_data, dict) and key in current_data:
    #             if i == len(keys) - 1:
    #                 print("***")
    #                 print(f"{'.'.join(keys[:i+1])} | {current_data[key]}")
    #                 return current_data[key]
    #             else:
    #                 current_data = current_data[key]
    #         else:
    #             print(f"Could not load config {constant} from {'.'.join(keys[:i+1])}. Using default instead.")
    #             return None
