import pickle
from sklearn.pipeline import Pipeline


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(model_path: str) -> Pipeline:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
