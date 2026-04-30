from typing import Optional

from torch.utils.data import Dataset, random_split
import torch

__all__ = ["split_dataset"]

def split_dataset(dataset: Dataset, ratios: list[float], seed: Optional[int] = None) -> list[Dataset]:
    """Splits a dataset into multiple subsets according to the given ratios.

    The total size is divided proportionally. Any remainder caused by integer
    rounding is added to the last subset.

    Args:
        dataset (Dataset): The dataset to split.
        ratios (list[float]): Relative sizes for each subset. Values do not need
            to sum to 1; they are normalised internally. For example,
            ``[8, 1, 1]`` produces 80 / 10 / 10 percent splits.
        seed (Optional[int]): Random seed for reproducible splits. If ``None``,
            the split is non-deterministic. Defaults to ``None``.

    Returns:
        list[Dataset]: A list of :class:`~torch.utils.data.Subset` objects with
            lengths proportional to ``ratios``.
    """
    total_size = len(dataset)
    ratio_sum = sum(ratios)

    lengths = [int(total_size * (r / ratio_sum)) for r in ratios]
    remainder = total_size - sum(lengths)
    lengths[-1] += remainder
    if seed:
        g = torch.Generator().manual_seed(seed)
    else:
        g = None

    return random_split(dataset, lengths, generator=g)
