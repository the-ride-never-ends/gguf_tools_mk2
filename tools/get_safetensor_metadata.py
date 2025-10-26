import json
from pathlib import Path
from pprint import pprint
import sys

# metadata = get_safetensors_metadata("bigscience/bloomz-560m")
try:
    from huggingface_hub import get_safetensors_metadata, parse_safetensors_file_metadata
except ImportError as e:
    print("Required module 'huggingface_hub' is not installed.")
    sys.exit(1)

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent

def main():
    repo_id = "RunDiffusion/Juggernaut-XL-v9"
    filename = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
    try:
        # "bigscience/bloomz-560m"
        metadata = parse_safetensors_file_metadata(repo_id, filename)
        tensors = metadata.tensors
        # pprint(f"metadata.metadata: {metadata.metadata}")
        # pprint(f"metadata.tensors: {tensors}")
        print("Got metadata")
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    else:
        json_path = THIS_DIR / f"{filename}.tensors.json"

        _tensors_dict = {}
        for key, value in metadata.tensors.items():
            value_dict = value.__dict__
            _tensors_dict[key] = value_dict
        try:
            with open(json_path, "w") as file:
                json.dump(_tensors_dict, file, indent=4)
            print(f"Metadata saved to {json_path}")
        except Exception as e:
            print(f"Error saving file to json: {e}")
            return 1
        else:
            return 0

if __name__ == "__main__":
    sys.exit(main())