
import os

import yaml

def get_config_files() -> dict:
    """
    Load YAML configuration files and return their contents.
    
    ### Returns
    - Dictionary with the loaded configuration data.
    
    ### Raises
    - FileNotFoundError: If either config file is not found.
    - yaml.YAMLError: If there's an error parsing the YAML files.
    """
    # Get the filepaths of all the config files.
    config_module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    top_level_dir = os.path.dirname(config_module_dir)
   
    _configs = {
        "SYSTEM": {
            "SKIP_STEPS": True,
            "FILENAME_PREFIX": top_level_dir
        },
        "FILENAMES": {
            "INPUT_FILENAME": "input.csv"
        }
    }
    _priv_config = {
        "PRIVATE_FOLDER_PATHS": {
            "INPUT_FOLDER": os.path.join(top_level_dir, "input"),
            "OUTPUT_FOLDER": os.path.join(top_level_dir, "output"),
        }
    }

    print(f"top_level_dir: {top_level_dir}")
    print(f"config_module_dir: {config_module_dir}")
    config_filenames = ["config.yaml", "private_config.yaml", "_private_config.yaml"]
    config_dict = {}

    for filename in config_filenames:
        top_level_path = os.path.join(top_level_dir, filename)
        config_module_path = os.path.join(config_module_dir, filename)

        if os.path.exists(top_level_path):
            config_path = top_level_path
        elif os.path.exists(config_module_path):
            config_path = config_module_path
            print(f"{filename} not found in top level directory. Using the one from config module directory.")
        else:
            config_path = top_level_path
            print(f"Warning: Could not find {filename}. Creating a new one from get_config_files' hardcoded presets...")
        
            # Determine which preset to use based on the filename
            if filename == "config.yaml":
                preset = _configs
            elif filename in "private_config.yaml":
                preset = _priv_config
            else:
                preset = {}
        
            # Create the new YAML file
            with open(config_path, 'w') as f:
                yaml.dump(preset, f)
    
        # Load the config file
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config is not None:
                    config_dict.update(loaded_config)
                else:
                    print(f"Warning: {filename} is empty or invalid.")
        except yaml.YAMLError as e:
            print(f"Error parsing {filename}: {e}")

    return config_dict