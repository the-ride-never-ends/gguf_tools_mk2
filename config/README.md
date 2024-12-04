# Config Module

## Overview
This module provides functionality for managing configuration settings in Python applications. 
It offers a flexible and easy-to-use approach to handling configuration data, supporting both local files and environment variables.

## Key Features
- Load configuration from YAML files
- Support for environment variable overrides
- Nested configuration structure
- Easy access to configuration values
- Default value support

## Dependencies
This module requires the following external libraries:
- PyYAML

## Usage
To use this module, import it as follows:
```python
from config import DEBUG_FILEPATH

print(DEBUG_FILEPATH)
"path/to/debug_files"
