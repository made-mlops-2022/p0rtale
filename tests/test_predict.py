import os

from project.predict import predict
from project.params import PredictParams

import pandas as pd


def test_predict(predict_params: PredictParams):
    assert os.path.exists(predict_params.output_predict_path)

    predict(predict_params)
    predictions = pd.read_csv(predict_params.output_predict_path)

    assert predictions.shape[1] == 1
    assert predictions.apply(lambda pred: pred[0] == 0 or pred[0] == 1, axis=1).all()
