from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictParams:
    input_data_path: str
    model_path: str
    output_predict_path: str
    target_col: str


PredictParamsSchema = class_schema(PredictParams)


def read_predict_params(path: str) -> PredictParams:
    with open(path, "r") as file:
        schema = PredictParamsSchema()
        return schema.load(yaml.safe_load(file))
