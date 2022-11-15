import pandas as pd

from ml_project.params.feature_params import FeatureParams
from ml_project.features.build_features import extract_target


def test_extract_target(dataset: pd.DataFrame, feature_params: FeatureParams):
    target = extract_target(dataset, feature_params)

    assert target.shape[0] == dataset.shape[0]
    assert target.apply(lambda x: x == 0 or x == 1).all()
