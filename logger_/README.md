# Logger Module

## Overview
The module sets up logging based on a configuration file (config.yaml) or falls back to default settings.
It supports various log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) and provides methods to log
messages with different severities. The Logger class can be instantiated with custom names, batch IDs,
and log levels, making it flexible for different parts of an application.

## Key Features
- Dynamic log file routing and folder creation
- A specialized prompt logger for use within LLM engines.
- Configurable log levels at the local and global level
- Options to format log messages with timestamps, log levels, and 'pretty' formatting.
- Optional program pausing at the individual message level.
- Support for both file and console logging

## Dependencies
This module requires the following external libraries:
- PyYAML

## Usage
To use this module, import it as follows:
```python
from logger import Logger
logger = Logger(logger_name="my_app")

logger.info("Application started")
logger.debug("Debug information")
logger.error("An error occurred")
