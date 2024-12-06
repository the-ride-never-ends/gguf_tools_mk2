from pathlib import Path


def move_logs_folders_into_this_folder_if_there_are_too_many_of_them(
        overflow_debug_folder: str,
        debug_log_folder: str,
        too_many: int = 25
    ) -> None:
    """
    Move log folders from debug_log_folder to overflow_debug_folder if there are too many.

    This function checks the number of folders in the debug_log_folder. If the number
    exceeds the specified threshold (too_many), it moves all folders from debug_log_folder
    to overflow_debug_folder.

    Args:
        overflow_debug_folder (str): The path to the folder where excess log folders will be moved.
        debug_log_folder (str): The path to the folder containing the log folders to be checked.
        too_many (int, optional): The threshold number of folders. Defaults to 25.

    Returns:
        None

    Side Effects:
        - Prints a message if folders are being moved.
        - Moves folders from debug_log_folder to overflow_debug_folder if the threshold is exceeded.
    """
    debug_path = Path(debug_log_folder)
    overflow_path = Path(overflow_debug_folder)

    # Count folders, excluding the overflow_debug_folder itself and the _debug_logs_go_here.txt file
    log_folders = [f for f in debug_path.iterdir() if f.is_dir() and f.name != "overflow_debug_folder"]
    if len(log_folders) > too_many:
        print(f"Moving {len(log_folders)} log folders to {overflow_path} because there are too many folders in {debug_path}.")
        for folder in log_folders:
            try:
                destination = overflow_path / folder.name
                folder.rename(destination)
            except Exception as e:
                print(f"ERROR moving folder {folder.name}: {e}")
