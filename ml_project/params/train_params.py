from dataclasses import dataclass
from .split_params import SplitParams
from .feature_params import FeatureParams
from .model_params import ModelParams
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig, OmegaConf


@dataclass()
class TrainParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    split: SplitParams
    features: FeatureParams
    model: ModelParams


TrainParamsSchema = class_schema(TrainParams)


def convert_to_train_params(config: DictConfig) -> TrainParams:
    container = OmegaConf.to_container(config, resolve=True)
    schema = TrainParamsSchema()
    return schema.load(container)
