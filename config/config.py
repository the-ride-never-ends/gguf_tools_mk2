import os


from .utils.config._get_config import get_config as config
from logger.logger import Logger
logger = Logger(logger_name=__name__)

# Define hard-coded constants
script_dir = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = script_dir = os.path.dirname(script_dir)
MAIN_FOLDER = os.path.dirname(script_dir)
YEAR_IN_DAYS: int = 365
DEBUG_FILEPATH: str = os.path.join(MAIN_FOLDER, "debug_logs")
RANDOM_SEED: int = 420

# Get YAML config variables
try:
    # SYSTEM
    path = "SYSTEM"
    SKIP_STEPS: bool = config(path, 'SKIP_STEPS') or True
    FILENAME_PREFIX: str = config(path, 'FILENAME_PREFIX') or f"{MAIN_FOLDER}"

    # FILENAMES
    path = "FILENAMES"
    INPUT_FILENAME: str = config(path, 'INPUT_FILENAME') or "input.csv"

    # PRIVATE PATH FOLDERS
    path = "PRIVATE_FOLDER_PATHS"
    INPUT_FOLDER: str = config(path, 'INPUT_FOLDER') or os.path.join(MAIN_FOLDER, "input")
    OUTPUT_FOLDER: str = config(path, 'OUTPUT_FOLDER') or os.path.join(MAIN_FOLDER, "output")

    print("YAML configs loaded.")

except KeyError as e:
    logger.exception(f"Missing configuration item: {e}")
    raise KeyError(f"Missing configuration item: {e}")

except Exception as e:
    logger.exception(f"Could not load configs: {e}")
    raise e

