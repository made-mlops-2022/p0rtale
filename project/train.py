import logging
import hydra
from omegaconf import DictConfig

from project.params import TrainParams, convert_to_train_params
from project.data.make_dataset import read_data, split_data
from project.features.build_features import build_transformer, extract_target
from project.models.model_fit_predict import (
    create_pipeline, train_model, predict_model, evaluate
)
from project.models.model_data import serialize_model
from project.models.save_reports import save_metrics


logger = logging.getLogger(__name__)


def train(config: TrainParams):
    logger.info(f"starting train with parameters: {config}")

    logger.info(f"reading the data using path: {config.input_data_path}")
    data = read_data(config.input_data_path)
    logger.info(f"data size: {data.shape}")

    logger.info("splitting the data into train and test")
    data_train, data_test = split_data(data, config.split, config.features.target_col)
    logger.info(f"train data size: {data_train.shape}")
    logger.info(f"test data size: {data_test.shape}")

    logger.info("extracting a target from data")
    target_train = extract_target(data_train, config.features)
    target_test = extract_target(data_test, config.features)

    data_train = data_train.drop(columns=[config.features.target_col])
    data_test = data_test.drop(columns=[config.features.target_col])

    logger.info("building a transformer")
    transformer = build_transformer(config.features)

    logger.info("fitting and transforming training data")
    features_train = transformer.fit_transform(data_train)

    logger.info(f"model training ({config.model.model_type})")
    model = train_model(features_train, target_train, config.model)

    logger.info("creating a pipeline of the final model")
    pipeline = create_pipeline(transformer, model)

    logger.info("getting model predictions on test data")
    predicts = predict_model(pipeline, data_test)

    logger.info("calculating metrics")
    metrics = evaluate(target_test, predicts)

    logger.info(f"saving metrics using file path: {config.metric_path}")
    save_metrics(metrics, config.metric_path)

    logger.info(f"serialization of the model using file path: {config.output_model_path}")
    serialize_model(pipeline, config.output_model_path)

    return metrics


@hydra.main(version_base=None, config_path="../configs/train", config_name="train")
def train_with_hydra(dict_config: DictConfig):
    config = convert_to_train_params(dict_config)
    train(config)


if __name__ == "__main__":
    train_with_hydra()
