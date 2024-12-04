import csv

from logger.Logger import Logger

logger = Logger(logger_name=__name__)

def save_to_csv(data: list[dict], filepath: str) -> None:
    """
    Save a list of dictionaries to a CSV file.
    """
    if not data:
        logger.warning("No data to save.")
        return
    
    keys = data[0].keys()
    
    with open(filepath, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)
    
    logger.info(f"Data saved to {filepath}")
