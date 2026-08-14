import logging
import multiprocessing as mp

from .model_worker import ModelWorker
from .workload_manager import Sequence

logger = logging.getLogger(__name__)


class ModelExecutor:
    def __init__(self):
        # None on the task queue is the shutdown sentinel.
        self.task_queue: mp.Queue[list[Sequence] | None] = mp.Queue()
        self.result_queue: mp.Queue[list[Sequence]] = mp.Queue()
        self.ready_event = mp.Event()  # set by the worker once the model is loaded
        self.worker_process: mp.Process | None = None  # currently single-GPU / worker

    def setup_worker(self, model_name: str, dtype: str):
        # The model runs in its own process so a slow forward pass never
        # blocks the server's event loop; queues are the only link.
        self.worker_process = mp.Process(
            target=ModelWorker.run,
            args=(
                model_name,
                dtype,
                self.task_queue,
                self.result_queue,
                self.ready_event,
            ),
        )
        self.worker_process.start()
        logger.info("worker process started (pid=%s)", self.worker_process.pid)

    def execute_batch(self, batch: list[Sequence]) -> list[Sequence]:
        self.task_queue.put(batch)
        return self.result_queue.get()  # only valid for a single worker

    def shutdown(self):
        # The worker's process exit is what releases the model's GPU memory —
        # the driver frees all of a process's allocations when it dies.
        if self.worker_process is None:
            return
        self.task_queue.put(None)
        self.worker_process.join(timeout=10)
        if self.worker_process.is_alive():
            self.worker_process.terminate()
            self.worker_process.join()
        self.worker_process = None
        logger.info("worker stopped; model memory released")

    def is_alive(self) -> bool:
        return self.worker_process is not None and self.worker_process.is_alive()

    def is_ready(self) -> bool:
        return self.is_alive() and self.ready_event.is_set()
