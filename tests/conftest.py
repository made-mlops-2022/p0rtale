import pytest
import logging

from project.params import (
    FeatureParams,
    TrainParams,
    PredictParams,
    SplitParams,
    LogisticRegressionParams
)

from project.data import generate_data

from project.train import train


def pytest_addoption(parser):
    parser.addoption(
        "--log-disable", action="append", default=[], help="disable specific loggers"
    )


def pytest_configure(config):
    for name in config.getoption("--log-disable", default=[]):
        logger = logging.getLogger(name)
        logger.propagate = False


@pytest.fixture()
def target_col():
    return "condition"


@pytest.fixture()
def categorical_features():
    return ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@pytest.fixture()
def numerical_features():
    return ["age", "trestbps", "chol", "thalach", "oldpeak"]


@pytest.fixture()
def features_to_drop():
    return []


@pytest.fixture()
def dataset():
    return generate_data()


@pytest.fixture()
def split_params():
    return SplitParams(test_size=0.2, random_state=42)


@pytest.fixture()
def feature_params(categorical_features, numerical_features,
                   features_to_drop, target_col):
    return FeatureParams(categorical_features=categorical_features,
                         numerical_features=numerical_features,
                         features_to_drop=features_to_drop,
                         target_col=target_col)


def train_params_step(tmp_path, split_params, feature_params):
    input_data_path = tmp_path / "data/data.csv"
    input_data_path.parent.mkdir()
    input_data_path.touch()

    output_model_path = tmp_path / "models/model.pkl"
    output_model_path.parent.mkdir()
    output_model_path.touch()

    metric_path = tmp_path / "models/metrics.json"
    metric_path.touch()

    data = generate_data()
    data.to_csv(str(input_data_path), index_label=False)

    return TrainParams(
        input_data_path=str(input_data_path),
        output_model_path=str(output_model_path),
        metric_path=str(metric_path),
        split=split_params,
        features=feature_params,
        model=LogisticRegressionParams(),
    )


@pytest.fixture()
def train_params(tmp_path, split_params, feature_params):
    return train_params_step(tmp_path, split_params, feature_params)


@pytest.fixture()
def predict_params(tmp_path, split_params, feature_params):
    train_params = train_params_step(tmp_path, split_params, feature_params)

    train(train_params)

    output_predict_path = tmp_path / "models/predict.csvs"
    output_predict_path.touch()

    return PredictParams(
        input_data_path=train_params.input_data_path,
        model_path=train_params.output_model_path,
        output_predict_path=str(output_predict_path),
        target_col=train_params.features.target_col,
    )
