"""
Module for the custom logging class.

This module provides a comprehensive logging solution with dynamic log folder generation,
routing capabilities, and specialized prompt logging for LLM engines.

Key Components:
1. Logger Class: Creates loggers with customizable settings and methods for various log levels.
2. Prompt Logger: Specialized logger for use within LLM engines.
3. Utility Functions: Manage log files and generate unique IDs.
4. Configuration Handling: Manages log levels and global settings via config.yaml or defaults.

Features:
- Dynamic creation of log files and folders
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Flexible instantiation with custom names, batch IDs, and log levels
- Automatic message formatting and log routing
- Support for both file and console logging
- Signal handling for graceful shutdown and log file completion
- Automatic management of log file sizes and cleanup of empty files and folders

Configuration:
- Uses config.yaml for settings, with fallback to default values
- Allows forcing a global log level for the entire program

Usage Example:
    from logger.logger import Logger
    logger = Logger(logger_name="my_app")
    logger.info("Application started")
    logger.debug("Debug information")
    logger.error("An error occurred")

Advanced Usage:
    logger.info("Important message", f=True)  # Formats message with asterisks
    logger.debug("Pausing after this", t=2)   # Pauses execution for 2 seconds after logging
    logger.warning("Silent warning", off=True)  # Logs message but doesn't print to console

Note: 
- Requires the 'yaml' package for parsing the configuration file.
- The 'q' parameter in logging methods is deprecated due to Python's limitations 
  in differentiating between regular and formatted strings at runtime.
"""
from datetime import datetime
import logging
import os
import signal
import sys
import time
from typing import Callable
import uuid


import yaml


from .utils.logger.delete_logs_if_they_get_too_big_on_disk import (
    delete_logs_if_they_get_too_big_on_disk
)
from .utils.logger.delete_empty_folders_in import delete_empty_folders_in
from .utils.logger.delete_empty_files_in import delete_empty_files_in


def make_id():
    return str(uuid.uuid4())


# Define general folder for log files
script_dir = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
PROGRAM_NAME = os.path.basename(PROJECT_ROOT)
debug_log_folder = os.path.join(PROJECT_ROOT, "debug_logs")

# Clean up debug folders.
delete_empty_files_in(debug_log_folder, with_ending=".log")
delete_empty_files_in(script_dir, with_ending='.Identifier')
max_size_in_megabytes = 200
delete_logs_if_they_get_too_big_on_disk(debug_log_folder, max_size_in_megabytes)
delete_empty_folders_in(debug_log_folder)

# Create a folder for current instantiation of the class.
# This should be created a-new every time the program is run or tested.
_RIGHT_NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
debug_log_folder = os.path.join(debug_log_folder, _RIGHT_NOW)
if not os.path.exists(debug_log_folder):
    os.mkdir(debug_log_folder)


# Import DEBUG config
# NOTE We do a separate yaml import to avoid circular imports with the config file.
config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    DEFAULT_LOG_LEVEL = config['SYSTEM']['DEFAULT_LOG_LEVEL']
    FORCE_DEFAULT_LOG_LEVEL_FOR_WHOLE_PROGRAM: bool = config['SYSTEM']['FORCE_DEFAULT_LOG_LEVEL_FOR_WHOLE_PROGRAM']
    print(f"DEFAULT_LOG_LEVEL set to {DEFAULT_LOG_LEVEL}\nFORCE_DEFAULT_LOG_LEVEL_FOR_WHOLE_PROGRAM set to {FORCE_DEFAULT_LOG_LEVEL_FOR_WHOLE_PROGRAM}")
except ModuleNotFoundError as e:
    print("ModuleNotFoundError when opening config.yaml")
except Exception as e:
    # Automatically run the entire program in debug mode if we lack configs.
    DEFAULT_LOG_LEVEL = logging.DEBUG
    FORCE_DEFAULT_LOG_LEVEL_FOR_WHOLE_PROGRAM = True
    print(f"Could not get debug level from config.yaml due to '{e}'.\nDefault LOG_LEVEL set to '{DEFAULT_LOG_LEVEL}'\nDefault FORCE_DEFAULT_LOG_LEVEL set to '{FORCE_DEFAULT_LOG_LEVEL_FOR_WHOLE_PROGRAM}'")

# NOTE
# CRITICAL = 50
# FATAL = CRITICAL
# ERROR = 40
# WARNING = 30
# WARN = WARNING
# INFO = 20
# DEBUG = 10
# NOTSET = 0

class Logger:
    """
    Create a logger with dynamic log folder generation and routing capabilities.
    Includes a specialized prompt logger for use within LLM engines.

    Parameters:
        logger_name (str): Name for the logger. Defaults to the program's name (top-level directory's name).
        prompt_name (str): Name of a prompt. Used by the prompt logger. Defaults to "prompt_log".
        batch_id (str): The logger's batch id. Used by the prompt logger. Defaults to random UUID4 string.
        current_time (datetime): The time a logger is initialized. Defaults to now() in "%Y-%m-%d_%H-%M-%S" format.
        log_level (int): The logging level. Defaults to DEFAULT_LOG_LEVEL from config or logging.DEBUG if config is unavailable.
        stacklevel (int): The depth of function calls for determining log origin. Defaults to None.

    Attributes:
        logger_name (str): The name of the logger.
        prompt_name (str): The name of the prompt (for prompt logging).
        batch_id (str): The batch ID for the logger.
        current_time (str): The initialization time of the logger.
        logger_folder (str): The folder where log files are stored.
        log_level (int): The current logging level.
        stacklevel (int): The depth of function calls for determining log origin.
        logger (logging.Logger): The underlying Python logger object.
        filename (str): The name of the log file.
        filepath (str): The full path to the log file.
        file_handler (logging.FileHandler): Handler for writing logs to a file.
        asterisk (str): Formatting string with asterisks.
        line (str): Formatting string with dashes.
        exception_symbol (str): Symbol used for exception logging.
        shutting_down (bool): Flag to indicate if the logger is shutting down.

    Methods:
        info(message, f=False, q=True, t=None, off=False): Log a message with severity 'INFO'.
        debug(message, f=False, q=True, t=None, off=False): Log a message with severity 'DEBUG'.
        warning(message, f=False, q=True, t=None, off=False): Log a message with severity 'WARNING'.
        error(message, f=False, q=True, t=None, off=False): Log a message with severity 'ERROR'.
        critical(message, f=False, q=True, t=None, off=False): Log a message with severity 'CRITICAL'.
        exception(message, f=False, q=True, t=None, off=False): Log a message with severity 'ERROR', including exception information.
        _setup_signal_handlers(): Register signal handlers for graceful shutdown.
        _handle_shutdown_signal(signum, frame): Handle shutdown signals.
        _cleanup(): Clean up logging resources on exit.
        _f(message): Format the message with asterisks.
        _message_template(message, method, f, q, t, off): Template for formatting and logging messages.

    Example:
        >>> from logger.logger import Logger
        >>> filename = __name__ # example.py
        >>> logger = Logger(logger_name=filename)
        >>> logger.info("Hello world!") # line 4
        '2024-09-18 18:38:44,185 - example_logger - INFO - example.py: 4 - Hello world!'

    Note:
        - The class uses the following log levels:
        CRITICAL = 50, ERROR = 40, WARNING = 30, INFO = 20, DEBUG = 10, NOTSET = 0
        - FATAL is an alias for CRITICAL, and WARN is an alias for WARNING.
        - The 'q' parameter in logging methods is deprecated due to Python's inability to differentiate 
          between regular and formatted strings at runtime.
        - The class includes signal handling for graceful shutdown on SIGINT and SIGTERM signals.
    """

    def __init__(self,
                 logger_name: str=PROGRAM_NAME,
                 prompt_name: str="prompt_log",
                 batch_id: str=make_id(),
                 current_time: datetime=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                 log_level: int=DEFAULT_LOG_LEVEL,
                 stacklevel: int=None
                ):
        self.logger_name = logger_name
        self.prompt_name = prompt_name
        self.batch_id = batch_id
        self.current_time = current_time
        self.logger_folder = debug_log_folder
        self.log_level = log_level if not FORCE_DEFAULT_LOG_LEVEL_FOR_WHOLE_PROGRAM else DEFAULT_LOG_LEVEL
        self.stacklevel = stacklevel
        self.logger: logging.Logger = None
        self.filename = None
        self.filepath = None
        self.file_handler = None
        # Formatting variables.
        self.asterisk = "\n********************\n"
        self.line = "\n--------------------\n"
        self.exception_symbol = None
        # Register signal handlers
        # This make it so log files are written even when the program errors or is stopped.
        self._setup_signal_handlers()
        self.shutting_down = False # Keep track of shutdown status

        # Create the specified log folder if it doesn't exist.
        # This assures that we always have a valid path for the log file.
        self.logger_folder = os.path.join(self.logger_folder, self.logger_name)
        if not os.path.exists(self.logger_folder):
            os.makedirs(self.logger_folder)

        # Determine properties of the logger based on its name.
        match logger_name:
            case logger_name if logger_name == PROGRAM_NAME: # If logger_name is the program's name
                self.logger = logging.getLogger(f"{PROGRAM_NAME}_logger")
                filename = f"{PROGRAM_NAME}_debug_log_{self.current_time}.log"
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s: %(lineno)d - %(message)s')
                self.stacklevel = self.stacklevel or 2 # We make stacklevel=2 as otherwise it'll give the filename and line numbers from the logger class itself.

            case "prompt":
                self.logger =  logging.getLogger(f"prompt_logger_for_{self.prompt_name}_batch_id_{self.batch_id}")
                filename =  f"{self.prompt_name}_{self.batch_id}_{self.current_time}.log"
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(lineno)d - %(message)s')
                self.stacklevel = self.stacklevel or 1 # Force the stack to log the LLM engine's name

            case _: # All other specialized loggers.
                self.logger = logging.getLogger(f"{self.logger_name}_logger")
                filename = f"{self.logger_name}_debug_log_{self.current_time}.log"
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s: %(lineno)d - %(message)s')
                self.stacklevel = self.stacklevel or 2

        # Create the logger itself.
        self.logger.setLevel(self.log_level) # Set the default log level.
        self.logger.propagate = False # Prevent logs from being handled by parent loggers

        if not self.logger.handlers:
            # Create handlers (file and console)
            self.filepath = os.path.join(self.logger_folder, filename)
            file_handler = logging.FileHandler(self.filepath)
            console_handler = logging.StreamHandler()

            # Set level for handlers
            file_handler.setLevel(logging.DEBUG)
            console_handler.setLevel(logging.DEBUG)

            # Create formatters and add it to handlers
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _setup_signal_handlers(self):
        """
        Register the signal handlers.
        These will be called in case of forced shutdowns and keyboard interrupts.
        """
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self, signum: int, frame) -> None:
        """
        Handle shutdown signals gracefully
        """
        if self.shutting_down:
            self.logger.warning("Received second shutdown signal! Forcing exit...")
            sys.exit(1)
            
        self.shutting_down = True
        signal_name = signal.Signals(signum).name
        
        self.logger.info(f"Received shutdown signal: {signal_name}")
        self.logger.info("Starting graceful shutdown...")
        
        try:
            self._cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            sys.exit(1)
            
        self.logger.info("Graceful shutdown completed")
        sys.exit(0)

    def _cleanup(self) -> None:
        """
        Cleanup logging resources on exit.
        """
        for handler in self.logger.handlers:
            handler.flush()
        logging.shutdown()

    def _f(self, message: str) -> str:
        """
        Format the message with a line of asterisks above and below it.
        The number of asterisks will have the same length as the input message, 
        with a maximum character length of 100.
        """
        self.asterisk = '*' * len(message)
        # Cut off the asterisk string at 50 characters to prevent wasted log space.
        self.asterisk = self.asterisk[:100] if len(message) > 100 else self.asterisk 
        return f"\n{self.asterisk}\n{message}\n{self.asterisk}\n"

    def _message_template(self, message: str, method: Callable, f: bool, q: bool, t: float, off: bool) -> None:
        """
        f is for formatting with self.asterisk.\n
        q is for automatically putting single quotes around f-string curly brackets
        t is for pausing the program by a specified number of seconds after the message has been printed to console.\n
        off turns off the logger for this message.
        NOTE q is deprecated due to Python's inability to tell the difference between a regular and formatted string at runtime.
        """
        if not off:
            if not f: # We move up the stack by 1 because it's a nested method.
                method(message, stacklevel=self.stacklevel+1)
            else:
                method(self._f(message), stacklevel=self.stacklevel+1)
            if t:
                time.sleep(t)

    def info(self, message, f: bool=False, q: bool=True, t: float=None, off: bool=False) -> None:
        """
        f is for formatting with self.asterisk.\n
        q is for automatically putting single quotes around f-string curly brackets
        t is for pausing the program by a specified number of seconds after the message has been printed to console.\n
        off turns off the logger for this message.
        NOTE q is deprecated due to Python's inability to tell the difference between a regular and formatted string at runtime.
        """
        self._message_template(message, self.logger.info, f, q, t, off)

    def debug(self, message, f: bool=False, q: bool=True, t: float=None, off: bool=False) -> None:
        """
        f is for formatting with self.asterisk.\n
        q is for automatically putting single quotes around f-string curly brackets
        t is for pausing the program by a specified number of seconds after the message has been printed to console.\n
        off turns off the logger for this message.\n
        NOTE q is deprecated due to Python's inability to tell the difference between a regular and formatted string at runtime.
        """
        self._message_template(message, self.logger.debug, f, q, t, off)

    def warning(self, message, f: bool=False, q: bool=True, t: float=None, off: bool=False) -> None:
        """
        f is for formatting with self.asterisk.\n
        q is for automatically putting single quotes around f-string curly brackets
        t is for pausing the program by a specified number of seconds after the message has been printed to console.\n
        off turns off the logger for this message.\n
        NOTE q is deprecated due to Python's inability to tell the difference between a regular and formatted string at runtime.
        """
        self._message_template(message, self.logger.warning, f, q, t, off)

    def error(self, message, f: bool=False, q: bool=True, t: float=None, off: bool=False) -> None:
        """
        f is for formatting with self.asterisk.\n
        q is for automatically putting single quotes around f-string curly brackets
        t is for pausing the program by a specified number of seconds after the message has been printed to console.\n
        off turns off the logger for this message.\n
        NOTE q is deprecated due to Python's inability to tell the difference between a regular and formatted string at runtime.
        """
        self._message_template(message, self.logger.error, f, q, t, off)

    def critical(self, message, f: bool=False, q: bool=True, t: float=None, off: bool=False) -> None:
        """
        f is for formatting with self.asterisk.\n
        q is for automatically putting single quotes around f-string curly brackets
        t is for pausing the program by a specified number of seconds after the message has been printed to console.\n
        off turns off the logger for this message.\n
        NOTE q is deprecated due to Python's inability to tell the difference between a regular and formatted string at runtime.
        """
        self._message_template(message, self.logger.critical, f, q, t, off)

    def exception(self, message, f: bool=False, q: bool=True, t: float=None, off: bool=False) -> None:
        """
        f is for formatting with self.asterisk.\n
        q is for automatically putting single quotes around f-string curly brackets
        t is for pausing the program by a specified number of seconds after the message has been printed to console.\n
        off turns off the logger for this message.\n
        NOTE q is deprecated due to Python's inability to tell the difference between a regular and formatted string at runtime.
        """
        self._message_template(message, self.logger.exception, f, q, t, off)

