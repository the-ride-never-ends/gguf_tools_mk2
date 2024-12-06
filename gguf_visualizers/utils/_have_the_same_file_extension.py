import argparse
import re

def _have_the_same_file_extension(*file_names: str | argparse.Namespace) -> bool:
    """
    Check if all given file names have the same file extension.

    This function compares the file extensions of all provided file names.
    It supports both string file names and argparse.Namespace objects.

    Args:
        *file_names (str | argparse.Namespace): Variable number of file names or argparse.Namespace objects.

    Returns:
        bool: True if all file names have the same extension, False otherwise.

    Raises:
        ValueError: If fewer than two file names are provided.

    Example:
        >>> _have_the_same_file_extension('file1.txt', 'file2.txt', 'file3.txt')
        True
        >>> _have_the_same_file_extension('file1.txt', 'file2.jpg', 'file3.txt')
        False
    """
    if len(file_names) < 2:
        raise ValueError("At least two files are required for comparison.")

    # Convert all inputs to strings
    file_str_list = [str(file) for file in file_names]

    # Regex pattern to extract file extension
    pattern: str = r'\.([^.]+)$'

    # Extracting extensions
    extensions_list = [re.search(pattern, file_str) for file_str in file_str_list]

    # Check if all extensions are found
    if not all(extensions_list):
        return False

    # Extract the actual extension strings
    ext_strs_list = [ext.group(1) for ext in extensions_list if ext]

    # Compare all extensions to the first one
    return all(ext == ext_strs_list[0] for ext in ext_strs_list)
