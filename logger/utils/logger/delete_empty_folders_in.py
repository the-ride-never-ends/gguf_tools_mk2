import os

def delete_empty_folders_in(directory: str) -> None:
    """
    Delete empty folders in the given directory.
    It walks through the directory tree bottom-up to ensure that nested empty folders
    are deleted properly.

    Args:
        directory (str): The path to the directory to search for empty folders.
    """
    for root, dirs, _ in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                # Check if the directory is empty
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    print(f"Deleted empty folder: '{dir_path}'")
            except OSError as e:
                print(f"OSError deleting folder '{dir_path}': {e}")
            except Exception as e:
                print(f"Error deleting folder '{dir_path}': {e}")

