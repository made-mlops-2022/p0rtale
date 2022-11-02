import os

from project.train import train
from project.params import TrainParams


def test_train(train_params: TrainParams):
    metrics = train(train_params)

    assert metrics["accuracy"] > 0
    assert metrics["precision"] > 0
    assert metrics["recall"] > 0
    assert metrics["f1"] > 0
    assert metrics["roc auc"] > 0

    assert os.path.exists(train_params.output_model_path)
