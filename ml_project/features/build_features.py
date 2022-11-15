import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml_project.params.feature_params import FeatureParams


def extract_target(data: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return data[params.target_col]


def build_categorical_pipeline() -> Pipeline:
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    encoder = OneHotEncoder(sparse=False)
    pipeline = Pipeline(
        [
            ("imputer", imputer),
            ("encoder", encoder),
        ]
    )
    return pipeline


def build_numerical_pipeline() -> Pipeline:
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    scaler = StandardScaler()
    pipeline = Pipeline(
        [
            ("imputer", imputer),
            ("scaler", scaler),
        ]
    )
    return pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("to-drop", "drop", params.features_to_drop),
            ("numerical", build_numerical_pipeline(), params.numerical_features),
            ("categorical", build_categorical_pipeline(), params.categorical_features)
        ],
        remainder='passthrough',
    )
