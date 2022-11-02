import numpy as np
import pandas as pd
from typing import Dict, Union

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml_project.params import ModelParams


SklearnClassifierModel = Union[LogisticRegression, GaussianNB, KNeighborsClassifier]


def create_pipeline(transformer: ColumnTransformer,
                    model: SklearnClassifierModel) -> Pipeline:
    return Pipeline(
        [
            ("transform", transformer),
            ("model", model),
        ]
    )


def train_model(features: pd.DataFrame,
                target: pd.Series,
                model_params: ModelParams) -> SklearnClassifierModel:
    if model_params.model_type == "LogisticRegression":
        model = LogisticRegression(penalty=model_params.penalty, C=model_params.C,
                                   random_state=model_params.random_state)
    elif model_params.model_type == "GaussianNB":
        model = GaussianNB(var_smoothing=model_params.var_smoothing)
    else:
        raise NotImplementedError()

    model.fit(features, target)

    return model


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return model.predict(features)


def evaluate(target: pd.Series, predicts: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy":  accuracy_score(target, predicts),
        "precision": precision_score(target, predicts),
        "recall":    recall_score(target, predicts),
        "f1":        f1_score(target, predicts, pos_label=1),
        "roc auc":   roc_auc_score(target, predicts),
    }
    return metrics
