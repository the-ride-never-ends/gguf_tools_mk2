import os
import sys
import traceback


from .delete_empty_files_in import delete_empty_files_in
from .delete_empty_folders_in import delete_empty_folders_in

from logger.logger import Logger
logger = Logger(logger_name="UNCAUGHT_EXCEPTION")


# Define general folder for log files
script_dir = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
PROGRAM_NAME = os.path.basename(PROJECT_ROOT)
debug_log_folder = os.path.join(PROJECT_ROOT, "debug_logs")
overflow_debug_folder = os.path.join(debug_log_folder, "overflow_debug_logs")


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """
    Handle uncaught exceptions in the application.

    This function is designed to be used as a custom exception handler for sys.excepthook.
    It logs critical errors with detailed information about the exception, excluding
    KeyboardInterrupt exceptions.

    Args:
        exc_type (Type[BaseException]): The type of the exception.
        exc_value (BaseException): The exception instance.
        exc_traceback (TracebackType): A traceback object encapsulating the call stack at the point where the exception occurred.

    Returns:
        None

    NOTE:
        THIS FILE NEEDS TO BE IMPORTED INTO ANOTHER MODULE (.e.g main.py, __init__.py) FOR THIS TO WORK!
        See: https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
        See: https://stackoverflow.com/questions/48642111/can-i-make-python-output-exceptions-in-one-line-via-logging
        See: https://stackoverflow.com/questions/48170682/can-structured-logging-be-done-with-pythons-standard-library/48202500#48202500
    """
    exc_info = (exc_type, exc_value, exc_traceback)

    # Ignore keyboard interrupts.
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(*exc_info)
    else:
        exc_traceback = str("\n".join(traceback.format_tb(exc_traceback)))

        write_val = f"""
        exc_type: {str(exc_type)}
        exc_value: {str(exc_value)}
        exc_traceback: {exc_traceback}
        """
        logger.critical(f"!!! Uncaught Exception !!!\n{write_val}", f=True, t=5)

    # Delete empty folders and files to make finding the errors easier.
    delete_empty_files_in(debug_log_folder, with_ending=".log")
    delete_empty_folders_in(debug_log_folder)
    return


sys.excepthook = handle_uncaught_exception
