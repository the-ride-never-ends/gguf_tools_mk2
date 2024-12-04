
import os
import time

MAX_FILES_TO_DELETE_AT_ONCE = 100

def _delete_files_until_50_percent_of_max_allowed_size(
    file_path_list: list[tuple[str, float, float]],
    total_size_in_bytes: float,
    max_size_in_bytes: float
    ) -> None:
    """
    Delete the oldest, largest log files until the total folder size is under 50% of the maximum allowed size 
    or until MAX_FILES_TO_DELETE_AT_ONCE logs have been deleted (currently 100).

    Args:
        file_path_list (list[tuple[str, float, float]: List of tuples containing file information (path, size, date).
        total_size (float): Current total size of all log files in bytes.
        max_size_in_bytes (float): Maximum allowed size of the folder in bytes.
    """
    # Return if the case file_path_list is empty for some reason.
    if not file_path_list:
        print("No logs found in debug folder.")
        return

    # Sort the file list by date (oldest first), then by size (largest first)
    file_path_list.sort(key=lambda x: (x[2], -x[1]))

    # Delete files until the total size is under 50% of the max size
    deleted_files = 0
    acceptable_size = max_size_in_bytes / 2
    current_size = total_size_in_bytes

    for file_path, file_size, _ in file_path_list:

        if current_size <= acceptable_size:
            break

        try:
            os.remove(file_path)
            current_size -= file_size
            deleted_files += 1
        except Exception as e:
            print(f"WARNING: Error deleting file '{file_path}': {e}")
            continue

        # Stop deleting files if we've deleted 100 of them already.
        if deleted_files >= MAX_FILES_TO_DELETE_AT_ONCE:
            break

    print(f"Deleted {deleted_files} logs from debug_logs folder.")
    return


def _get_log_files_info_and_total_size(debug_folder: str) -> tuple[list[tuple[str, float, float]], float]:
    """
    Get a list of all log files in the debug folder, as well as their sizes and creation dates,
    and calculate the total size of all log files.

    Args:
        debug_folder (str): The path to the debug folder containing log files.

    Returns:
        tuple: A tuple containing two elements:
            - list[tuple[str, float, float]: A list of tuples, each containing (file_path, file_size, file_date) for each log file.
            - float: The total size of all log files in bytes.
    """
    total_size_in_bytes = 0
    file_path_list = []

    for root, _, filenames in os.walk(debug_folder):
        for filename in filenames:
            if filename.endswith('.log'):
                file_path = os.path.join(root, filename)
                try:
                    file_size = os.path.getsize(file_path)
                    file_date = os.path.getmtime(file_path)

                    total_size_in_bytes += file_size
                    file_info = (file_path, file_size, file_date)
                    file_path_list.append(file_info)

                except OSError as e:
                    print(f"WARNING: Error accessing file '{file_path}': {e}")
                    continue

    return file_path_list, total_size_in_bytes


def _handle_user_input(timeout_seconds: int = 30) -> bool:
    """Handle user input with timeout."""
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout_seconds:
            print(f"Timeout: No response received within {timeout_seconds} seconds. Returning...")
            return False

        enter = input("Would you like to delete some of them? The oldest, largest files will be deleted first (y/n): ")
        if enter.lower() in ['y', 'n']:
            return True if enter.lower() == 'y' else False
        print("Please enter 'y' or 'n': ")


def delete_logs_if_they_get_too_big_on_disk(debug_folder: str, max_size_in_megabytes: float | int) -> None:
    """
    Delete old log files if the total size exceeds the specified maximum.
    NOTE: This function has a lot of checks behind it to prevent it from accidentally deleting something important.

    Args:
        debug_folder: Path to the debug folder containing log files.
        max_size_in_megabytes: Maximum allowed size of the folder in megabytes.

    """
    # Check if input variables have valid values.
    if debug_folder is None or not debug_folder.strip():
        print("WARNING: debug_folder must be a non-empty string. Returning...")
        return

    if max_size_in_megabytes is None or max_size_in_megabytes <= 0:
        print("WARNING: max_size_in_megabytes must be a positive float or integer. Returning...")
        return

    if not os.path.isdir(debug_folder):
        print(f"WARNING: debug_folder directory '{debug_folder}' does not exist. Returning...")
        return

    file_path_list, total_size_in_bytes = _get_log_files_info_and_total_size(debug_folder)

    # Convert to bytes for easier comparison.
    max_size_in_bytes = float(max_size_in_megabytes * (1024 ** 2))

    if total_size_in_bytes > max_size_in_bytes:
        print(f"WARNING: Total size of logs in debug folder is more than the maximum size of {max_size_in_megabytes} megabytes.")
        if not _handle_user_input():
            return
        else:
            _delete_files_until_50_percent_of_max_allowed_size(file_path_list, total_size_in_bytes, max_size_in_bytes)
    return
