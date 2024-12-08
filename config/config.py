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

# Define program-specific hard-coded constants
# Clip values to at max 7 standard deviations from the mean.
CFG_SD_CLIP_THRESHOLD = 7
# Number of standard deviations above the mean to be positively scaled.
CFG_SD_POSITIVE_THRESHOLD = 1
# Number of standard deviations below the mean to be negatively scaled.
CFG_SD_NEGATIVE_THRESHOLD = 1
# RGB scaling for pixels that meet the negative threshold.
CFG_NEG_SCALE = (1.2, 0.2, 1.2)
# RGB scaling for pixels that meet the positive threshold.
CFG_POS_SCALE = (0.2, 1.2, 1.2)
# RGB scaling for pixels between those ranges.
CFG_MID_SCALE = (0.1, 0.1, 0.1)
# CFG_MID_SCALE = (0.6, 0.6, 0.9) Original Values


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

