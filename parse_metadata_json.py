import csv
import json
from pathlib import Path


def parse_metadata_json(json_path: str | Path):
    
    if not isinstance(json_path, (str, Path)):
        raise TypeError(f"json_path must be a string or a Path object, got {type(json_path).__name__}")
    
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"The file '{json_path}' does not exist.")

    try:
        with open(json_path, "r") as file:
            data = json.load(file)
    except Exception as e:
        raise IOError(f"An error occurred while reading the JSON file: {e}") from e

