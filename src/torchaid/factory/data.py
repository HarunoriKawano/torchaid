from collections.abc import Callable

from torch.utils.data import Dataset, DataLoader

from torchaid.core.configs import HyperParameters
from torchaid.core.states import BatchState

def get_dataloader(dataset: Dataset, hyper_parameters: HyperParameters, collate_fn: Callable[..., BatchState], shuffle: bool) -> DataLoader[BatchState]:
    dataloader = DataLoader(
        dataset, shuffle=shuffle, batch_size=hyper_parameters.batch_size, num_workers=hyper_parameters.cpu_num_works,
        pin_memory=True, persistent_workers=True, collate_fn=collate_fn
    )

    return dataloader
