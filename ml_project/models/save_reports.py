import numpy as np
from typing import Dict
import json


def save_predictions(predictions: np.ndarray, file_path: str):
    np.savetxt(file_path, predictions, delimiter=",", fmt="%d")


def save_metrics(metrics: Dict[str, float], file_path: str):
    with open(file_path, "w") as metric_file:
        json.dump(metrics, metric_file)
