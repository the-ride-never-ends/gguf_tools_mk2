from typing import Any


from config.utils.config.__init__ import CONFIG_DATA


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
    current_data = CONFIG_DATA
    for i, key in enumerate(keys):
        if isinstance(current_data, dict) and key in current_data:
            if i == len(keys) - 1:
                full_key = '.'.join(keys[:i+1])
                num_dashes_needed = len(full_key) * len(str(current_data[key]))
                dashes = "-" * num_dashes_needed
                current_data_key = current_data[key] if "PRIVATE" not in full_key else "XXXXXXXXXX"
                print(f"{dashes}\n{full_key}: | {current_data_key} |")
                return current_data[key]
            else:
                current_data = current_data[key]
        else:
            print(f"Could not load config {constant} from {'.'.join(keys[:i+1])}. Using default instead.")
            return False