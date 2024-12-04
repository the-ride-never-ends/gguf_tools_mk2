import asyncio
from typing import Any, Coroutine

import pandas as pd

def _list_set_tuple(inputs: list|set|tuple, func: Coroutine, enum: bool, outer_task_name: str, *args, **kwargs) -> list[asyncio.Task]:
    if enum:
        return [
            asyncio.create_task(
                func(idx, inp, *args, **kwargs),
                name=outer_task_name
            ) for idx, inp in enumerate(inputs)
        ]
    else:
        return [
            asyncio.create_task(
                func(inp, *args, **kwargs),
                name=outer_task_name
            ) for inp in inputs
        ]

def _dict(inputs: dict, func: Coroutine, enum: bool, outer_task_name: str, *args, **kwargs) -> list[asyncio.Task]:
    if enum:
        return [
            asyncio.create_task(
                func(idx, (key, value), *args, **kwargs),
                name=outer_task_name
                ) for idx, (key, value) in enumerate(inputs.items())
            ]
    else:
        return [
            asyncio.create_task(
                func((key, value), *args, **kwargs),
                name=outer_task_name,
                ) for (key, value) in inputs.items()
            ]

def _pd_dataframe(inputs: pd.DataFrame, func: Coroutine, enum: bool, outer_task_name: str, *args, **kwargs) -> list[asyncio.Task]:
    if enum:
        return [
            asyncio.create_task(
                func(idx, row, *args, **kwargs),
                name=outer_task_name
            ) for idx, row in enumerate(inputs.itertuples())
        ]
    else:
        return [
            asyncio.create_task(
                func(row, *args, **kwargs),
                name=outer_task_name
            ) for row in inputs.itertuples()
        ]

async def create_tasks_list_with_outer_task_name(inputs: Any, func: Coroutine, enum: bool, outer_task_name: str, *args, **kwargs) -> list[asyncio.Task]:
    """
    Create a list of asyncio Tasks from the given inputs and function.

    This function takes various types of input (list, set, tuple, dict, or pandas DataFrame) and
    creates asyncio Tasks for each element or row. It can optionally enumerate the inputs.

    Args:
        inputs (Any): The input data structure (list, set, tuple, dict, or pandas DataFrame).
        func (Coroutine): The coroutine function to be executed for each input.
        enum (bool): If True, enumerate the inputs and pass the index to the function.
        outer_task_name (str): The name to be assigned to each created task.
        *args: Additional positional arguments to be passed to the function.
        **kwargs: Additional keyword arguments to be passed to the function.

    Returns:
        list[asyncio.Task]: A list of asyncio Tasks created from the inputs.

    Raises:
        ValueError: If the input type is not supported.

    Note:
        The function behaves differently based on the type of 'inputs':
        - For list, set, or tuple: Creates a task for each element.
        - For dict: Creates a task for each key-value pair.
        - For pandas DataFrame: Creates a task for each row.
    """
    if isinstance(inputs, (list,set,tuple)):
        return _list_set_tuple(inputs, func, enum, outer_task_name, *args, **kwargs) 
    elif isinstance(inputs, dict):
        return _dict(inputs, func, enum, outer_task_name, *args, **kwargs) 
    elif isinstance(inputs, pd.DataFrame):
        return _pd_dataframe(inputs, func, enum, outer_task_name, *args, **kwargs)
    else:
        raise ValueError(f"Argument 'inputs' has an unsupported type '{type(inputs)}'")