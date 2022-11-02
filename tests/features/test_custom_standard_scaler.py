import pandas as pd
import numpy as np

from ml_project.params.feature_params import FeatureParams
from ml_project.features.custom_standard_scaler import CustomStandardScaler


def test_custom_standard_scaler(dataset: pd.DataFrame,
                                feature_params: FeatureParams,
                                target_col: str):
    if target_col in dataset.columns:
        dataset = dataset.drop(columns=[target_col])

    transformer = CustomStandardScaler(
        numerical_features=feature_params.numerical_features
    )
    transformer.fit(dataset)
    new_features = transformer.transform(dataset)

    assert new_features.shape == dataset.shape

    for feature in dataset:
        if feature in feature_params.numerical_features:
            assert np.isclose(np.mean(new_features[feature]), 0)
            assert np.isclose(np.var(new_features[feature]), 1)
