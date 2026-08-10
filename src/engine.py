import logging
import time

from model_executor import ModelExecutor
from workload_manager import WorkloadManager

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self):
        self.model_executor = ModelExecutor()
        self.workload_manager = WorkloadManager()

        # Set up the model executor worker for the default model.
        self.model_executor.setup_worker("facebook/opt-125m")

    def generate(self, prompts: list[str]) -> list[str]:
        # Queue every prompt, then pull batches until all of ours are done.
        # Batches may mix in other callers' sequences — that's the point.
        request_ids = [self.workload_manager.add_request(prompt) for prompt in prompts]
        logger.info("queued %d request(s)", len(request_ids))
        while not all(self.workload_manager.is_finished(request_id) for request_id in request_ids):
            batch = self.workload_manager.get_next_batch()
            if not batch:
                # Another caller's batch is carrying our sequences; wait for it.
                time.sleep(0.01)
                continue
            for sequence in self.model_executor.execute_batch(batch):
                self.workload_manager.update_sequence(sequence)
        # Outputs in the same order the prompts came in.
        return [self.workload_manager.pop_finished(request_id).output for request_id in request_ids]
