# Module: utils

Shared utility functions for custom modules. 
These functions are not specific to any particular module or type, and may be taken from other repos.
Note that many of these functions rely on each other to work, have overlapping implementations, and may not work.
This repo is for standardization purposes

## Functions
### Decorators
- adjust_wait_time_for_executions: Adjust a sleep waiting period to account for the clock time taken to execute a synchronous function.
    Useful for optimizing waiting periods based on a reference value e.g. a robots.txt delay.
- get_exec_time: Decorator to calculate how long a function takes to execute.
- if_not_results: This decorator wraps the given function and checks its return value. 
    If the function returns a falsy value (None, empty list, etc.), 
    it logs a warning message and returns None. 
- try_except: A decorator that wraps a function in a try-except block with optional retries and exception raising.
    This decorator allows you to automatically handle exceptions for a function,
    with the ability to specify the number of retry attempts and whether to
    ultimately raise the exception or not.
### Limiters
- create_tasks_list_with_outer_task_name: Create a list of asyncio Tasks from the given inputs and function.
    This function takes various types of input (list, set, tuple, dict, or pandas DataFrame) and
    creates asyncio Tasks for each element or row. It can optionally enumerate the inputs.
- create_tasks_list: Create a list of coroutine tasks based on input data and a given function.
- Limiter: Create an instance-based custom rate-limiter based on a semaphore.
    Options for a custom stop condition and progress bar.
### Regular Functions
- convert_integer_to_datetime_str: Converts an integer representation of a datetime to a formatted string.
    This function takes an integer input representing a datetime in the format
    YYYYMMDDhhmmss and converts it to a string in the format 'YYYY-MM-DD HH:MM:SS'.
- get_formatted_datetime: Get the exact time in "%Y-%m-%d %H:%M:%S" format.
- make_id: Generate a unique identifier using UUID4.
- make_insert_command_args: Prepare positional and keyword arguments for an MySQL INSERT command.
- make_sha256_hash: Generate a SHA-256 hash from any positional arguments.
- next_step: Log a step message and optionally pause it and prompt for continuation.
    Used in main functions to deliniate sequential steps and pause the program for user input.
- return_s_percent: Produce a comma-separated string of "%s" based on the length of the input.
- safe_format_js_selector: N/A (not implemented)
- safe_format: Safely format a string using the SafeFormatter class.
    Allows for Python values and code to be evaluated first, then inserted into strings.
    Useful for loading in external text and treating that text like an f-string
- sanitize_filename: Sanitize a string to be used as (part of) a filename.
- save_to_csv: Save a list of dictionaries to a CSV file.

## Requirements

multipledispatch
pandas
pyyaml
tqdm



