from functools import wraps
import time
from typing import Any, Callable

from logger import Logger

def adjust_wait_time_for_execution(wait_in_seconds: float=5) -> Callable[..., Any]:
    """
    Adjust a sleep waiting period to account for the clock time taken to execute a synchronous function.
    Useful for optimizing waiting periods based on a reference value e.g. a robots.txt delay.
    """
    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args,**kwargs) -> Any|None:

            # Initialize nonlocal and logger
            nonlocal wait_in_seconds
            logger = Logger(logger_name=func.__module__)

            # Wait for function to run, clock it's runtime, then subtract that from wait_in_seconds.
            start = time.time()
            result = func(*args,**kwargs)
            end = time.time() - start
            wait_in_seconds -= end

            # Pause for the duration of the adjusted wait_in_seconds, then return the function's result.
            logger.info(f"Execution time for function '{func.__name__}' took {end} seconds to execute.\nWait time is now '{wait_in_seconds}' seconds.")
            if wait_in_seconds > 0:
                time.sleep(wait_in_seconds)
            return result

        return wrapper
    return decorator