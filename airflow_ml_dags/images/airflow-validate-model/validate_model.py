import os
import click
import json
import pickle

import pandas as pd
import numpy as np

from typing import Dict

from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
)


def evaluate(target: pd.Series, predict: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy":  accuracy_score(target, predict),
        "precision": precision_score(target, predict),
        "recall":    recall_score(target, predict),
        "f1":        f1_score(target, predict, pos_label=1),
        "roc_auc":   roc_auc_score(target, predict),
    }
    return metrics


@click.command("validate_model")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def validate_model(input_dir: str, model_dir: str, output_dir: str):
    features = pd.read_csv(os.path.join(input_dir, "val_features.csv"))
    target = pd.read_csv(os.path.join(input_dir, "val_target.csv"))

    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    predict = model.predict(features)

    metrics = evaluate(target, predict)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    validate_model()