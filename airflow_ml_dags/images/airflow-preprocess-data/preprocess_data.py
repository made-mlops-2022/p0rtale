import os
import click
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_numerical_pipeline() -> Pipeline:
    imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    scaler = StandardScaler()
    pipeline = Pipeline(
        [
            ("imputer", imputer),
            ("scaler", scaler),
        ]
    )
    return pipeline


@click.command("preprocess_data")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess_data(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    pipeline = build_numerical_pipeline()
    processed_data = pd.DataFrame(pipeline.fit_transform(data))

    os.makedirs(output_dir, exist_ok=True)
    processed_data.to_csv(os.path.join(output_dir, "processed_data.csv"), index=False)
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    preprocess_data()