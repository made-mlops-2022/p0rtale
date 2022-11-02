import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split

from project.params import SplitParams


def read_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data


def split_data(data: pd.DataFrame,
               params: SplitParams,
               target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_train, data_test = train_test_split(data, test_size=params.test_size,
                                             random_state=params.random_state,
                                             stratify=data[target_col])
    return data_train, data_test
