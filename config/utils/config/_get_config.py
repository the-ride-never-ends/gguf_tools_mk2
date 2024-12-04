from typing import Any

from ._get_config_files import get_config_files

# Load private and public config.yaml files.
data: dict = get_config_files()

def get_config(path:str, constant:str) -> Any | bool:
    """
    Get a key from a yaml file.

    Args:
        path (str): The path to the desired key, using dot notation for nested structures.
        constant (str): The specific key to retrieve.

    Returns:
        Union[Any, bool]: The value of the key if found, False otherwise.

    Examples:
        >>> config("SYSTEM", "CONCURRENCY_LIMIT")
        2
        >>> config("SYSTEM", "NONEXISTENT_KEY") or 3
        3
    """
    keys = path + "." + constant

    # Split the path into individual keys
    keys = path.split('.') + [constant]

    # Traverse the nested dictionary
    current_data = data
    for i, key in enumerate(keys):
        if isinstance(current_data, dict) and key in current_data:
            if i == len(keys) - 1:
                print(f"Config {constant} from {'.'.join(keys[:i+1])} set to {current_data[key]}")
                return current_data[key]
            else:
                current_data = current_data[key]
        else:
            print(f"Could not load config {constant} from {'.'.join(keys[:i+1])}. Using default instead.")
            return False