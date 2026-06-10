import math
import pandas as pd
from typing import Tuple, Union


def snap_split_index(ordered_df: pd.DataFrame, idx: int, date_col: str) -> int:
    """ Moves a row-based split index forward to the next date boundary.

        Given a dataframe sorted ascending by ``date_col``, returns the smallest
        ``j >= idx`` such that ``date[j] != date[idx - 1]``. This guarantees that
        all matches sharing a date land on the same side of a temporal split,
        preventing intra-day leakage between train and eval folds.
    """

    total = ordered_df.shape[0]
    if idx <= 0:
        return 0
    if idx >= total:
        return total

    dates = ordered_df[date_col].to_numpy()
    boundary_date = dates[idx - 1]
    j = idx
    while j < total and dates[j] == boundary_date:
        j += 1
    return j


def train_test_split(df: pd.DataFrame, test_size: Union[int, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Selects the first N matches as test set and the remaining matches as train set. """

    # Validate that the matches are provided in descending order, otherwise the calculations will be wrong.
    if not df['Date'].is_monotonic_decreasing:
        raise ValueError('Expected dates to be sorted in a descending order.')

    if isinstance(test_size, float):
        test_size = int(math.floor(test_size*df.shape[0]/100.0))

    df_test = df.iloc[:test_size].reset_index(drop=True)
    df_train = df.iloc[test_size:].reset_index(drop=True)
    return df_train, df_test
