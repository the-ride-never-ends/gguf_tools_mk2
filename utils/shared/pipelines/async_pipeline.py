import asyncio
from asyncio import Task
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


from logger.logger import Logger
logger = Logger(logger_name=__name__)


@dataclass
class PipelineStage:
    name: str
    worker_count: int
    handler: Callable
    use_executor: bool = False
    max_concurrent: Optional[int] = None

class AsyncPipeline:
    """
    Generic async pipeline system that can handle multiple stages of processing
    with configurable concurrency and execution strategies.
    """
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
        self.queues: Dict[str, asyncio.Queue] = {}
        self.tasks: Dict[str, List[Task]] = {}
        self.executors: Dict[str, ProcessPoolExecutor] = {}
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Set up queues and concurrency controls
        for stage in stages:
            self.queues[stage.name] = asyncio.Queue()
            if stage.max_concurrent:
                self.semaphores[stage.name] = asyncio.Semaphore(stage.max_concurrent)
            if stage.use_executor:
                self.executors[stage.name] = ProcessPoolExecutor(max_workers=stage.worker_count)

    async def stage_worker(self, 
                         stage: PipelineStage,
                         worker_id: int,
                         input_queue: asyncio.Queue,
                         output_queue: Optional[asyncio.Queue] = None):
        """
        Asynchronous worker function for a pipeline stage.

        This function processes items from the input queue, applies the stage's handler,
        and optionally forwards results to the output queue. It handles concurrency limits,
        process pool execution, and graceful shutdown.

        Args:
            stage (PipelineStage): The pipeline stage configuration.
            worker_id (int): Unique identifier for this worker within the stage.
            input_queue (asyncio.Queue): Queue from which to receive input items.
            output_queue (Optional[asyncio.Queue]): Queue to send processed items to the next stage, if any.

        The worker continues processing until it receives a None item, signaling shutdown.
        """
        logger.info(f"Starting {stage.name}_{worker_id}")
        
        while True:
            try:
                # Get input with optional rate limiting
                if stage.name in self.semaphores:
                    async with self.semaphores[stage.name]:
                        item = await input_queue.get()
                else:
                    item = await input_queue.get()

                if item is None:
                    logger.info(f"{stage.name}_{worker_id} received shutdown signal")
                    input_queue.task_done()
                    # Only propagate shutdown signal if this is the last worker to receive it
                    if output_queue and input_queue.empty():
                        for _ in range(self.stages[self.stages.index(stage) + 1].worker_count):
                            await output_queue.put(None)
                    break

                # Process the item with semaphore around the entire processing
                try:
                    if stage.name in self.semaphores:
                        async with self.semaphores[stage.name]:
                            if stage.use_executor:
                                loop = asyncio.get_running_loop()
                                result = await loop.run_in_executor(
                                    self.executors[stage.name],
                                    stage.handler,
                                    item
                                )
                            else:
                                result = await stage.handler(item)
                    else:
                        if stage.use_executor:
                            loop = asyncio.get_running_loop()
                            result = await loop.run_in_executor(
                                self.executors[stage.name],
                                stage.handler,
                                item
                            )
                        else:
                            result = await stage.handler(item)

                    # Forward result to next stage if it exists
                    if output_queue and result is not None:
                        await output_queue.put(result)
                        
                    logger.info(f"{stage.name}_{worker_id} processed item successfully")
                
                except Exception as e:
                    logger.exception(f"Error in {stage.name}_{worker_id}: {e}")
                    
                finally:
                    input_queue.task_done()

            except Exception as e:
                logger.error(f"Critical error in {stage.name}_{worker_id}: {e}")
                break

    async def run(self, input_items: List[Any]):
        """
        Run the pipeline with the given input items
        """
        try:
            # Start all workers for each stage
            for i, stage in enumerate(self.stages):
                output_queue = self.queues[self.stages[i + 1].name] if i < len(self.stages) - 1 else None
                
                self.tasks[stage.name] = [
                    asyncio.create_task(
                        self.stage_worker(
                            stage,
                            worker_id,
                            self.queues[stage.name],
                            output_queue
                        )
                    )
                    for worker_id in range(stage.worker_count)
                ]

            # Feed initial items into first stage
            first_queue = self.queues[self.stages[0].name]
            for item in input_items:
                await first_queue.put(item)

            # Add shutdown signals to first stage only
            for _ in range(self.stages[0].worker_count):
                await first_queue.put(None)

            # Wait for all stages to complete
            for stage in self.stages:
                await self.queues[stage.name].join()

            # Wait for all tasks to complete
            for task_list in self.tasks.values():
                await asyncio.gather(*task_list)

        finally:
            # Clean up executors
            for executor in self.executors.values():
                executor.shutdown()