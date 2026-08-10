import logging
import multiprocessing as mp

from .model_worker import ModelWorker
from .workload_manager import Sequence

logger = logging.getLogger(__name__)


class ModelExecutor:
    def __init__(self):
        self.task_queue: mp.Queue[list[Sequence]] = mp.Queue()
        self.result_queue: mp.Queue[list[Sequence]] = mp.Queue()
        self.worker_process: mp.Process | None = None

    def setup_worker(self, model_name: str):
        # The model runs in its own process so a slow forward pass never
        # blocks the server's event loop; queues are the only link.
        self.worker_process = mp.Process(
            target=ModelWorker.run,
            args=(model_name, self.task_queue, self.result_queue),
        )
        self.worker_process.start()
        logger.info("worker process started (pid=%s)", self.worker_process.pid)

    def execute_batch(self, batch: list[Sequence]) -> list[Sequence]:
        self.task_queue.put(batch)
        return self.result_queue.get()
