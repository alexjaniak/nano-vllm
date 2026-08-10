import uuid
from dataclasses import dataclass


@dataclass
class Sequence:
    # One unit of work, flowing engine -> executor -> worker and back.
    # The worker fills in `output` and flips `finished`.
    request_id: str
    prompt: str
    output: str = ""
    finished: bool = False


class WorkloadManager:
    def __init__(self, max_batch_size: int = 8):
        self.sequences: dict[str, Sequence] = {}
        self.waiting: list[str] = []  # FIFO queue
        self.max_batch_size = max_batch_size

    def add_request(self, prompt: str) -> str:
        sequence = Sequence(request_id=str(uuid.uuid4()), prompt=prompt)
        self.sequences[sequence.request_id] = sequence
        self.waiting.append(sequence.request_id)
        return sequence.request_id

    def get_next_batch(self) -> list[Sequence]:
        # Take up to max_batch_size sequences from the front of the queue.
        batch_ids = self.waiting[: self.max_batch_size]
        self.waiting = self.waiting[self.max_batch_size :]
        return [self.sequences[request_id] for request_id in batch_ids]

    def update_sequence(self, sequence: Sequence) -> None:
        # The worker returns pickled copies, so replace rather than mutate.
        self.sequences[sequence.request_id] = sequence

    def is_finished(self, request_id: str) -> bool:
        return self.sequences[request_id].finished

    def pop_finished(self, request_id: str) -> Sequence:
        return self.sequences.pop(request_id)
