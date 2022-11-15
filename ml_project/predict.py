import logging
import click

from ml_project.data.make_dataset import read_data
from ml_project.models.save_reports import save_predictions
from ml_project.models.model_data import load_model
from ml_project.params.predict_params import read_predict_params, PredictParams


logging.basicConfig(
    format='[%(asctime)s][%(levelname)s] - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def predict(config: PredictParams):
    logger.info(f"starting predict with parameters: {config}")

    logger.info(f"reading the data using path: {config.input_data_path}")
    data = read_data(config.input_data_path)
    if config.target_col in data.columns:
        data = data.drop(columns=[config.target_col])
    logger.info(f"data size: {data.shape}")

    logger.info(f"reading the model using path: {config.model_path}")
    model = load_model(config.model_path)
    logger.info(f"model: {model['model'].__class__.__name__}")

    logger.info("getting model predictions")
    predictions = model.predict(data)

    logger.info(f"saving predictions using path: {config.output_predict_path}")
    save_predictions(predictions, config.output_predict_path)


@click.command(name="predict")
@click.argument("config_path")
def predict_command(config_path: str):
    config = read_predict_params(config_path)
    predict(config)


if __name__ == "__main__":
    predict_command()
