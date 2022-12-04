import os
import click
import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression


@click.command("train_model")
@click.option("--input-dir")
@click.option("--model-dir")
def train_model(input_dir: str, model_dir: str):
    features = pd.read_csv(os.path.join(input_dir, "train_features.csv"))
    target = pd.read_csv(os.path.join(input_dir, "train_target.csv"))

    model = LogisticRegression(penalty="l2", C=1.0, random_state=42)
    model.fit(features, target)

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model()
