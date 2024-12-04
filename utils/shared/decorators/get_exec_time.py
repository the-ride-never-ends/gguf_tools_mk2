from functools import wraps
import time
from typing import Any, Callable

from logger import Logger

def get_exec_time(func: Callable) -> Any:
    """
    Decorator to calculate how long a function takes to execute.

    Examples:
    >>> @get_time
    >>> def factorial(num):
    >>>     time.sleep(2)
    >>>     print(math.factorial(num))
    >>> factorial(10)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        # Define the logger.
        logger = Logger(logger_name=func.__module__)

        # Log start time.
        begin = time.time()
        
        # Execute the function.
        result = func(*args, **kwargs)

        # Log end time.
        end = time.time()
        logger.info(f"Total execution time for '{func.__name__}':  {end - begin}")
        return result
    return wrapper


