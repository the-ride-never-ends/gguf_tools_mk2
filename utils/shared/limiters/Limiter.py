import asyncio
from asyncio import Task
from typing import Any, AsyncGenerator, Callable, Coroutine, Generator, Iterable
import functools 
from functools import partial, partialmethod
import time

from tqdm import tqdm
import pandas as pd

from tqdm import asyncio as tqdmasyncio


from .create_tasks_list import create_tasks_list
from .create_tasks_list_with_outer_task_name import create_tasks_list_with_outer_task_name


class Limiter:
    """
    Create an instance-based custom rate-limiter based on a semaphore.
    Options for a custom stop condition and progress bar.
    """
    def __init__(self, 
                 semaphore: int | asyncio.Semaphore = 2, 
                 stop_condition: Any = "stop_condition", # Replace with your specific stop condition
                 progress_bar: bool=True
                ):
        self.semaphore = asyncio.Semaphore(semaphore) if isinstance(semaphore, int) else semaphore
        self.stop_condition = stop_condition
        self.progress_bar = progress_bar

    # Claude insisted that I include these for compatability/future use purposes.
    # It's probably a good idea. 
    async def __aenter__(self):
        """
        Initialize the Limiter using a context manager.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the limiter using a context manager.
        """
        pass

    @classmethod
    def start(cls, *args, **kwargs):
        """
        Initialize the Limiter using a factory method.
        """
        instance = cls(*args, **kwargs)
        return instance

    def stop():
        """
        Exit the limiter.
        """
        return

    async def async_pbar_listener(self, queue: asyncio.Queue, len_inputs: int):
        """
        Update a tqdm progress bar in multi-process environment.
        """
        pbar = tqdm(total=len_inputs)
        async for _ in iter(queue.get, None):
            pbar.update()


    async def run_task_with_limit(self, task: Coroutine) -> Any:
        """
        Set up rate-limit-conscious functions
        """
        async with self.semaphore:
            result = await task
            if result == self.stop_condition:
                global stop_signal
                stop_signal = True
            return result 


    async def func_worker(self, 
                          name: str, 
                          partial_func: Coroutine, 
                          queue: asyncio.Queue, 
                          run_task_with_limit: bool
                          ) -> None:
        """
        Worker that asynchronously process inputs in a queue.
        """
        while True:
            print(f"Starting Task {name}...")

            inp = await queue.get()

            if inp is None:
                queue.put(inp)
                break

            if run_task_with_limit:
                await partial_func(inp)
            else:
                await self.run_task_with_limit(partial_func(inp))

            queue.task_done()

            print(f"Task {name} completed.")


    async def func_producer(self, inputs: list | pd.DataFrame, queue: asyncio.Queue):
        # Fill the queue with inputs
        if isinstance(inputs, list):
            for inp in inputs:
                if inp is None:
                    break
                await queue.put(inp)
        elif isinstance(inputs, pd.DataFrame):
            for row in inputs.itertuples():
                if row is None:
                    break
                await queue.put(row)
        else:
            raise NotImplementedError(f"run_async_many_in_queue does not currently support type '{type(inputs)}'")

    async def process_asap(self, 
                           name: str, 
                           producer_queue: asyncio.Queue,
                           consumer_queue: asyncio.Queue,
                           func: Callable,
                           executor):
        """Continuously processes content as it becomes available"""
        loop = asyncio.get_event_loop()

        while True:
            inp = await producer_queue.get()
            if inp is None:
                print(f"Processor {name} shutting down...")
                inp.task_done()
                break

            # Process in separate process pool
            result = await loop.run_in_executor(executor, func, func, inp)
            await consumer_queue.put(result)
            print(f"⚡ Processor {name} processed {inp}")
            consumer_queue.task_done()

    # TODO Add in tqdm support.
    async def run_async_many_in_queue(self,
                             *args,
                             inputs: Iterable[Any] = None,
                             func: Callable = None,
                             num_workers: int = 3,
                             outer_task_name: str = None,
                             run_task_with_limit: bool = False,
                             **kwargs
                            ) -> asyncio.Future | AsyncGenerator:
        """
        Asynchronously process multiple inputs using a queue and multiple workers.

        This method creates a queue of inputs and processes them using a specified number of worker tasks.
        It supports both list and pandas DataFrame inputs.

        Args:
            *args: Variable length argument list to be passed to the function.
            inputs (Iterable[Any], optional): An iterable of inputs to be processed. Can be a list or pandas DataFrame.
            func (Callable, optional): The function to be applied to each input.
            num_workers (int, optional): The number of worker tasks to create. Defaults to 3.
            outer_task_name (str, optional): Name for the outer task, if any.
            run_task_with_limit (bool, optional): Whether to run tasks with a rate limit. Defaults to False.
            **kwargs: Arbitrary keyword arguments to be passed to the function.

        Returns:
            asyncio.Future | AsyncGenerator: The result of asyncio.gather() on all worker tasks.

        Raises:
            NotImplementedError: If the input type is not supported.

        Note:
            This method uses asyncio.Queue to manage inputs and asyncio.create_task to spawn worker tasks.
            It waits for all inputs to be processed before cancelling worker tasks and returning results.
        """
        # Instantiate the queues.
        input_queue = asyncio.Queue()
        producer_queue = asyncio.Queue()
        consumer_queue = asyncio.Queue()

        # Make a partial function from the constant args and kwargs.
        partial_func = partial(func, *args, **kwargs)

        # Fill the queue with inputs
        if isinstance(inputs, list):
            for inp in inputs:
                if inp is None:
                    break
                await queue.put(inp)
        elif isinstance(inputs, pd.DataFrame):
            for row in inputs.itertuples():
                if row is None:
                    break
                await queue.put(row)
        else:
            raise NotImplementedError(f"run_async_many_in_queue does not currently support type '{type(inputs)}'")

        # Assign the tasks to the workers.
        tasks = []
        for i in range(num_workers):
            task: Task = asyncio.create_task(
                self.func_worker(
                    f'{func.__name__}-{i}', partial_func, queue, run_task_with_limit
                ),
                name=outer_task_name
            )
            tasks.append(task)

        # Wait until the queue is fully processed.
        await queue.join()
        for task in tasks:
            task.cancel()

        # Wait until all worker tasks are cancelled.
        return await asyncio.gather(*tasks, return_exceptions=True)


    async def run_async_many(self, 
                             *args, 
                             inputs: Any = None, 
                             func: Callable = None,
                             enum: bool = True,
                             outer_task_name: str = None,
                             **kwargs
                            ) -> asyncio.Future | AsyncGenerator:
        if not inputs:
            raise ValueError("input_list was not input as a parameter")

        if not func:
            raise ValueError("func was not input as a parameter")

        # NOTE Adding an outer_task_name changes the tasks list from a list of Coroutines to a list of Tasks.
        # However, running it through the limiter appears to change them back into Coroutines, so maybe it's fine???
        if outer_task_name and self.progress_bar is False:
            tasks = create_tasks_list_with_outer_task_name(inputs, func, enum, *args, **kwargs)
        else:
            tasks = create_tasks_list(inputs, func, enum, *args, **kwargs)

        task_list = [
            self.run_task_with_limit(task) for task in tasks
        ]

        if self.progress_bar:
            for future in tqdmasyncio.tqdm.as_completed(task_list):
                await future
        else:
            return await asyncio.gather(*task_list)

