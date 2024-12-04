# """
# A decorator that wraps a function in a try-except block with optional retries and exception raising.

# This decorator allows you to automatically handle exceptions for a function,
# with the ability to specify the number of retry attempts and whether to
# ultimately raise the exception or not.

# NOTE: 'Exception' is automatically added to the exception argument list if not specified.
# NOTE: The decorator can also be used for asynchronous functions by using the flag "async_" for asynchronous functions.

# Args:
#     exception (list): A tuple of exception types to catch. Defaults to [Exception].
#     raise_exception (bool): If True, raises the caught exception after all retries
#                             have been exhausted. If False, suppresses the exception.
#                             Defaults to False.
#     retries (int): The number of times to retry the function if an exception occurs.
#                     If None, the function will only be attempted once. Defaults to None.
#     logger (logging.Logger): A logger instance. Defaults to None.
#     async_ (bool): Flag for if the decorated function is a coroutine
#                     If not present, the function is treated as synchronous.

# Returns:
#     function: A decorated function or coroutine that implements the try-except logic.

# Example:
# >>> # Synchronous example
# >>> @try_except(exception=[ValueError, TypeError], raise_exception=True, retries=3)
# >>> def test_func(x):
# >>>     return x / 0 
# >>> test_func(-1)
# ERROR:__main__:ValueError exception in 'test_func': x cannot be negative
# Retrying (0/3)...
# >>> # Asynchronous example
# >>> @try_except(exception=[ValueError, TypeError], raise_exception=True, retries=3, async_=True)
# >>> async def test_func(x):
# >>>     return await x / 0 
# >>> await test_func(-1)
# ERROR:__main__:ValueError exception in 'test_func': x cannot be negative
# Retrying (0/3)...
# """

from functools import wraps
import inspect
import logging
import sys
from typing import Any, Callable, Coroutine


from logger.logger import Logger


# def _get_logger(func: Callable|Coroutine, logger: Logger=None):
#     """
#     Get the logger
#     """
#     return logger or Logger(logger_name=func.__module__, stacklevel=1)

# def _get_exception_list(exception: list) -> tuple:
#     """
#     Get the list of exceptions and retunr it as a tuple.
#     """
#     exception_list = list(exception)
#     if Exception not in exception_list:
#         exception_list.extend([Exception])
#     return tuple(exception_list)

# def _while_loop(func, 
#                 attempts: int, 
#                 exception_tuple: tuple, 
#                 logger: Logger, 
#                 *args, 
#                 retries: int=0, 
#                 **kwargs) -> Exception|None:
#     finally_e = None
#     while attempts <= retries:
#         # Try the function.
#         try:
#             return func(*args, **kwargs)
#         except exception_tuple as e:
#             # Define error variables
#             finally_e: Exception = e
#             error_name: str = e.__class__.__name__
#             func_name = func.__name__
#             error_message = f"{error_name} exception in '{func_name}'\n{e}"
#             retry_message = f"Retrying ({attempts}/{retries})..."

#             # If no retries are specified, log the error and raise it if requested.
#             if retries <= 0: 
#                 logger.exception(error_message)
#                 break
#             else:
#                 # On first attempt, print the error and retry message
#                 if attempts <= 0: 
#                     print(error_message)
#                     print(retry_message)

#                 # On subsequent attempts, print the retry message.
#                 elif attempts > 0 and attempts < retries: 
#                     print(retry_message)

#                 # On the final attempt, log the error and raise it if requested.
#                 else: 
#                     print(f"Function '{func_name}' errored after {attempts + 1} retries.")
#                     logger.exception(f"{error_message}\nretries: {attempts + 1}")
#                     break
#                 attempts += 1
#     return finally_e

# @overload
# async def _while_loop(func, 
#                 attempts: int, 
#                 exception_tuple: tuple, 
#                 logger: Logger, 
#                 *args, 
#                 retries: int=0, 
#                 async_: bool=True, # Dummy argument to differentiate from the sync version. Overloaded functions yay!
#                 **kwargs) -> Exception|None:
#     finally_e = None
#     while attempts <= retries:
#         # Try the function.
#         try:
#             return await func(*args, **kwargs)
#         except exception_tuple as e:
#             # Define error variables
#             finally_e: Exception = e
#             error_name: str = e.__class__.__name__
#             func_name = func.__name__
#             error_message = f"{error_name} exception in '{func_name}'\n{e}"
#             retry_message = f"Retrying ({attempts}/{retries})..."

#             # If no retries are specified, log the error and raise it if requested.
#             if retries <= 0: 
#                 logger.exception(error_message)
#                 break
#             else:
#                 # On first attempt, print the error and retry message
#                 if attempts <= 0: 
#                     print(error_message)
#                     print(retry_message)

#                 # On subsequent attempts, print the retry message.
#                 elif attempts > 0 and attempts < retries: 
#                     print(retry_message)

#                 # On the final attempt, log the error and raise it if requested.
#                 else: 
#                     print(f"Function '{func_name}' errored after {attempts + 1} retries.")
#                     logger.exception(f"{error_message}\nretries: {attempts + 1}")
#                     break
#                 attempts += 1
#     return finally_e


# def _sync_wrapper(func, exception, raise_exception, retries, logger):
#     @wraps(func)
#     def _wrapper(*args, **kwargs):
#         nonlocal logger
#         logger = _get_logger(func, logger=logger)
#         exception_tuple = _get_exception_list(exception)

#         # Determine if func is a method and prepare to use __exit__ or __aexit__ if it is.
#         if inspect.ismethod(func):
#             instance = args[0]  # Get the class instance (self)
#             exit_context: Callable = getattr(instance, '__exit__', None)
#             if not exit_context:
#                 logger.warning(f"{func.__name__} is a method, but the class does not have an __exit__ method.")
#         else:
#             exit_context = None
        
#         try:
#             finally_e = _while_loop(func, 0, exception_tuple, logger, *args, retries=retries, **kwargs)
#         finally:
#             # Raise the exception if requested.
#             if raise_exception:
#                 if exit_context: # Handle the call to __exit__ if the method has it
#                     exception_info = sys.exc_info()
#                     exit_context(exception_info[0], exception_info[1], exception_info[2])
#                 raise finally_e
#             else:
#                 pass
#     return _wrapper


# async def _async_wrapper(func, exception, raise_exception, retries, logger):
#     @wraps(func)
#     async def _async_wrapper(*args, **kwargs):
#         nonlocal logger
#         logger = _get_logger(func, logger=logger)
#         exception_tuple = _get_exception_list(exception)

#         # Determine if func is a method and prepare to use __exit__ or __aexit__ if it is.
#         if inspect.ismethod(func):
#             instance = args[0]  # Get the class instance (self)
#             exit_context: Callable = getattr(instance, '__aexit__') or getattr(instance, '__exit__', None)
#             if not exit_context:
#                 logger.warning(f"{func.__name__} is a method, but the class does not have an __aexit__ or __exit__ method.")
#         else:
#             exit_context = None

#         try:
#             finally_e: Exception = await _while_loop(func, 0, exception_tuple, logger, *args, retries=retries, async_=True, **kwargs)
#         finally:
#             # Raise the exception if requested.
#             if raise_exception:
#                 if exit_context: # Handle the call to __aexit__ if the method has it
#                     exception_info = sys.exc_info()
#                     if inspect.iscoroutinefunction(exit_context):
#                         await exit_context(exception_info[0], exception_info[1], exception_info[2])
#                     else:
#                         exit_context(exception_info[0], exception_info[1], exception_info[2])
#                 raise finally_e
#             else:
#                 pass
#     return _async_wrapper

# # Yes, I know this unconventional, but fuck it!

# from functools import singledispatch
# @singledispatch
# def try_except(exception: list=[Exception],
#                raise_exception: bool=False,
#                retries: int=0,
#                logger: logging.Logger=None,
#                )

# @overload
# def try_except(exception: list=[Exception],
#                raise_exception: bool=False,
#                retries: int=0,
#                logger: logging.Logger=None,
#                ) -> Callable:
#     """
#     A decorator that wraps a function in a try-except block with optional retries and exception raising.

#     This decorator allows you to automatically handle exceptions for a function,
#     with the ability to specify the number of retry attempts and whether to
#     ultimately raise the exception or not.

#     NOTE: 'Exception' is automatically added to the exception argument list if not specified.
#     NOTE: The decorator can also be used for asynchronous functions by using the flag "async_" for asynchronous functions.

#     Args:
#         exception (list): A tuple of exception types to catch. Defaults to [Exception].
#         raise_exception (bool): If True, raises the caught exception after all retries
#                                 have been exhausted. If False, suppresses the exception.
#                                 Defaults to False.
#         retries (int): The number of times to retry the function if an exception occurs.
#                         If None, the function will only be attempted once. Defaults to None.
#         logger (logging.Logger): A logger instance. Defaults to None.
#         async_ (bool): Flag for if the decorated function is a courtine
#                         If not present, the function is treated as synchronous.

#     Returns:
#         function: A decorated function or coroutine that implements the try-except logic.

#     Example:
#     >>> # Synchronous example
#     >>> @try_except(exception=[ValueError, TypeError], raise_exception=True, retries=3)
#     >>> def test_func(x):
#     >>>     return x / 0 
#     >>> test_func(-1)
#     ERROR:__main__:ValueError exception in 'test_func': x cannot be negative
#     Retrying (0/3)...
#     >>> # Asynchronous example
#     >>> @try_except(exception=[ValueError, TypeError], raise_exception=True, retries=3, async_=True)
#     >>> async def test_func(x):
#     >>>     return await x / 0 
#     >>> await test_func(-1)
#     ERROR:__main__:ValueError exception in 'test_func': x cannot be negative
#     Retrying (0/3)...
#     """
#     def decorator(func: Callable):
#         assert not isinstance(func, Coroutine)
#         return _sync_wrapper(func, exception, raise_exception, retries, logger)
#     return decorator

# @overload
# async def try_except(exception: list=[Exception],
#                     raise_exception: bool=False,
#                     retries: int=0,
#                     logger: logging.Logger=None,
#                     async_:bool=True
#                     ) -> Coroutine:
#     """
#     A decorator that wraps a function in a try-except block with optional retries and exception raising.

#     This decorator allows you to automatically handle exceptions for a function,
#     with the ability to specify the number of retry attempts and whether to
#     ultimately raise the exception or not.

#     NOTE: 'Exception' is automatically added to the exception argument list if not specified.
#     NOTE: The decorator can also be used for asynchronous functions by using the flag "async_" for asynchronous functions.

#     Args:
#         exception (list): A tuple of exception types to catch. Defaults to [Exception].
#         raise_exception (bool): If True, raises the caught exception after all retries
#                                 have been exhausted. If False, suppresses the exception.
#                                 Defaults to False.
#         retries (int): The number of times to retry the function if an exception occurs.
#                         If None, the function will only be attempted once. Defaults to None.
#         logger (logging.Logger): A logger instance. Defaults to None.
#         async_ (bool): Flag for if the decorated function is a Coroutine
#                         If not present, the function is treated as synchronous.

#     Returns:
#         function: A decorated function or coroutine that implements the try-except logic.

#     Example:
#     >>> # Synchronous example
#     >>> @try_except(exception=[ValueError, TypeError], raise_exception=True, retries=3)
#     >>> def test_func(x):
#     >>>     return x / 0 
#     >>> test_func(-1)
#     ERROR:__main__:ValueError exception in 'test_func': x cannot be negative
#     Retrying (0/3)...
#     >>> # Asynchronous example
#     >>> @try_except(exception=[ValueError, TypeError], raise_exception=True, retries=3, async_=True)
#     >>> async def test_func(x):
#     >>>     return await x / 0 
#     >>> await test_func(-1)
#     ERROR:__main__:ValueError exception in 'test_func': x cannot be negative
#     Retrying (0/3)...
#     """
#     async def decorator(func: Coroutine):
#         assert isinstance(func, Coroutine)
#         return await _async_wrapper(func, exception, raise_exception, retries, logger)
#     return await decorator


def try_except(exception: list=[Exception],
               raise_exception: bool=False,
               retries: int=0,
               logger: logging.Logger=None,
               ) -> Callable:
    """
    A decorator that wraps a function in a try-except block with optional retries and exception raising.

    This decorator allows you to automatically handle exceptions for a function,
    with the ability to specify the number of retry attempts and whether to
    ultimately raise the exception or not.

    NOTE: 'Exception' is automatically added to the exception argument list if not specified.
    TODO: Figure out how to make this take coroutines as well

    Args:
        exception (list): A tuple of exception types to catch. Defaults to [Exception].
        raise_exception (bool): If True, raises the caught exception after all retries
                                have been exhausted. If False, suppresses the exception.
                                Defaults to False.
        retries (int): The number of times to retry the function if an exception occurs.
                       If None, the function will only be attempted once. Defaults to None.
        logger (logging.Logger): A logger instance. Defaults to None.

    Returns:
        function: A decorated function or coroutine that implements the try-except logic.

    Example:
    >>> @try_except(exception=[ValueError, TypeError], raise_exception=True, retries=3)
    >>> def test_func(x):
    >>>     return x / 0 
    >>> test_func(-1)
    ERROR:__main__:ValueError exception in 'test_func': x cannot be negative
    Retrying (0/3)...
    """
    def decorator(func: Callable|Coroutine) -> Callable|Coroutine:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:

            # Initialize Logger and other variables.
            # NOTE See: https://stackoverflow.com/questions/1261875/what-does-nonlocal-do-in-python-3
            nonlocal logger
            logger = logger or Logger(logger_name=func.__module__, stacklevel=1)
            attempts = 0

            # Since we don't want any uncaught exceptions
            # We add in Exception to the input exception list 
            # if it's not specified in it.
            exception_list = list(exception)
            if Exception not in exception_list:
                exception_list.extend([Exception])
            exception_tuple = tuple(exception_list)

            # Determine if func is a method and prepare to use __exit__ or __aexit__ if it is.
            if inspect.ismethod(func):
                instance = args[0]  # Get the class instance (self)
                exit_context: Callable = getattr(instance, '__exit__') if inspect.iscoroutinefunction(func) else getattr(instance, '__exit__', None)
                if not exit_context:
                    logger.warning(f"{func.__name__} is a method, but the class does not have an __exit__ or __aexit__ method.")
            else:
                exit_context = None

            try:
                while attempts <= retries:
                    # Try the function.
                    try:
                        return func(*args, **kwargs)
                    except exception_tuple as e:
                        # Define error variables
                        finally_e: Exception = e
                        error_name: str = e.__class__.__name__
                        func_name = func.__name__
                        error_message = f"{error_name} exception in '{func_name}'\n{e}"
                        retry_message = f"Retrying ({attempts}/{retries})..."

                        # If no retries are specified, log the error and raise it if requested.
                        if retries <= 0: 
                            logger.exception(error_message)
                            break
                        else:
                            # On first attempt, print the error and retry message
                            if attempts <= 0: 
                                print(error_message)
                                print(retry_message)

                            # On subsequent attempts, print the retry message.
                            elif attempts > 0 and attempts < retries: 
                                print(retry_message)

                            # On the final attempt, log the error and raise it if requested.
                            else: 
                                print(f"Function '{func_name}' errored after {attempts + 1} retries.")
                                logger.exception(f"{error_message}\nretries: {attempts + 1}")
                                break
                            attempts += 1
            finally:
                # Raise the exception if requested.
                if raise_exception:
                    if exit_context: # Handle the call to __exit__ if the method has it
                        exception_info = sys.exc_info()
                        exit_context(exception_info[0], exception_info[1], exception_info[2])
                    raise finally_e
                else:
                    pass
        return wrapper
    return decorator
