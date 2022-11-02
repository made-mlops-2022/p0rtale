import pandas as pd

from project.data import split_data
from project.params import SplitParams


def test_split_data(dataset: pd.DataFrame, split_params: SplitParams, target_col: str):
    data_train, data_test = split_data(dataset, split_params, target_col)

    part_size = dataset.shape[0] / 10

    assert data_train.shape[1] == data_test.shape[1]
    assert data_train.shape[0] > part_size
    assert data_test.shape[0] > part_size
