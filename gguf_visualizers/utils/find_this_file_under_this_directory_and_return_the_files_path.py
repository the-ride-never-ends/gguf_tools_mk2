import os
from pathlib import Path


def find_this_file_under_this_directory_and_return_the_files_path(file_name: str, directory: str) -> Path:
    """
    Search for a file in the current directory and its sub-directories,
    and return its absolute path.

    Args:
        file_name (str): The name of the file to search for.
    Returns:
        the path of the file
    """
    for root, _, files in os.walk(directory):
        if file_name in files:
            return Path(root) / file_name

    raise FileNotFoundError(f"File '{file_name}' not found in the current directory or its subdirectories.")