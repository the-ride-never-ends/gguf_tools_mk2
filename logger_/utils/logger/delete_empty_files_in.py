import os

# EXAMPLE OF os.walk
# os.walk returns a generator, that creates a tuple of values (current_path, directories in current_path, files in current_path).
#
# Every time the generator is called it will follow each directory recursively until no further sub-directories are available from the initial directory that walk was called upon.
#
# As such,
#
# os.walk('C:\dir1\dir2\startdir').next()[0] # returns 'C:\dir1\dir2\startdir'
# os.walk('C:\dir1\dir2\startdir').next()[1] # returns all the dirs in 'C:\dir1\dir2\startdir'
# os.walk('C:\dir1\dir2\startdir').next()[2] # returns all the files in 'C:\dir1\dir2\startdir'

# Auto-clean the specified directory of empty files.
def delete_empty_files_in(root_folder, with_ending: str = ".log"):
    """
    Delete empty files (i.e. file size == 0) with the specified ending
    from the every folder under the specified directory.
    """
    file_ending = with_ending # Syntactic sugar.
    count = 0
    for root, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(file_ending):
                file_path = os.path.join(root, filename)
                try:
                    if os.path.getsize(file_path) == 0: # 0kb
                        os.remove(file_path)
                        count += 1
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
                    continue
    print(f"Deleted {count} files with '{file_ending}' ending.")
    return
