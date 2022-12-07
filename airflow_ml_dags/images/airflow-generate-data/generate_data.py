import os
import click
import pandas as pd

from sklearn.datasets import make_classification


@click.command("generate_data")
@click.option("--output-dir")
def generate_data(output_dir: str):
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.DataFrame(y, columns=["target"])

    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == "__main__":
    generate_data()
