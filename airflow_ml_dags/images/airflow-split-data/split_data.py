import os
import click
import pandas as pd

from sklearn.model_selection import train_test_split


@click.command("split_data")
@click.option("--input-dir")
@click.option("--output-dir")
def split_data(input_dir: str, output_dir: str):
    X = pd.read_csv(os.path.join(input_dir, "processed_data.csv"))
    y = pd.read_csv(os.path.join(input_dir, "target.csv"))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "train_features.csv"), index=False)
    X_val.to_csv(os.path.join(output_dir, 'val_features.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'train_target.csv'), index=False)
    y_val.to_csv(os.path.join(output_dir, 'val_target.csv'), index=False)


if __name__ == '__main__':
    split_data()
