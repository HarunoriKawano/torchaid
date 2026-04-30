from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, random_split
import torch

__all__ = ["split_dataset", "split_dataframe"]

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
    if not ratios:
        raise ValueError("ratios must not be empty")
    if any(r <= 0 for r in ratios):
        raise ValueError("all values in ratios must be positive")

    total_size = len(dataset)
    if total_size == 0:
        raise ValueError("dataset is empty")
    if total_size < len(ratios):
        raise ValueError(
            f"dataset size ({total_size}) is smaller than the number of splits ({len(ratios)})"
        )

    ratio_sum = sum(ratios)

    lengths = [int(total_size * (r / ratio_sum)) for r in ratios]
    remainder = total_size - sum(lengths)
    lengths[-1] += remainder
    if seed:
        g = torch.Generator().manual_seed(seed)
    else:
        g = None

    return random_split(dataset, lengths, generator=g)



def split_dataframe(df: pd.DataFrame, ratios: list[float], seed: Optional[int] = None) -> list[pd.DataFrame]:
    """Shuffles a pandas DataFrame and splits it into multiple DataFrames based on specified ratios.

    This function randomly shuffles the rows of the input DataFrame and then
    partitions it according to the proportional weights provided in the `ratios` list.
    The index of each resulting DataFrame is reset to start from 0.

    Args:
        df (pd.DataFrame): The input DataFrame to be split.
        ratios (List[float]): A list of numerical values representing the desired size
            ratio for each split. For example, `[3.0, 1.5, 4.0]` will divide the data
            into three parts with exactly those proportions.
        seed (Optional[int], optional): A seed value for the random number
            generator to ensure reproducibility of the shuffle. Defaults to None.

    Returns:
        List[pd.DataFrame]: A list containing the split DataFrames, where each
        DataFrame's index is cleanly reset.
    """
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    ratios_array = np.array(ratios)
    proportions = ratios_array / ratios_array.sum()

    total_len = len(df_shuffled)
    split_indices = (np.cumsum(proportions) * total_len).astype(int)[:-1]

    split_dfs = np.split(df_shuffled, split_indices)

    return [d.reset_index(drop=True) for d in split_dfs]
