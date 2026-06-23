import torch
from torch.utils.data import DataLoader

from .protocols import SizedIterable
from .states import BatchState


class ShortDataLoader:
    def __init__(self, dataloader: SizedIterable[BatchState], num_batches: int):
        self.dl = dataloader
        self.num_batches = num_batches

    def __iter__(self):
        for i, batch in enumerate(self.dl):
            if i >= self.num_batches:
                break
            yield batch

    def __len__(self):
        return min(self.num_batches, len(self.dl))