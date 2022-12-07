import os
import click
import pickle
import pandas as pd


@click.command("predict")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    processed_data = pd.read_csv(os.path.join(input_dir, "processed_data.csv"))

    with open(os.path.join(model_dir, "model.pkl"), "rb") as file:
        model = pickle.load(file)

    predictions = model.predict(processed_data)
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(predictions).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    predict()
