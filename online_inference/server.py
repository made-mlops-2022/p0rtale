import os
import pickle
import gdown
import pandas as pd
import uvicorn

from typing import List, Optional
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi_health import health

from sklearn.pipeline import Pipeline

from heart_disease_data import HeartDiseaseData


class ConditionResponse(BaseModel):
    condition: int


app = FastAPI()

model: Optional[Pipeline] = None

def is_model_ready():
    return model is not None

app.add_api_route('/health', health(conditions=[is_model_ready]))


@app.on_event("startup")
def load_model():
    global model

    model_path = os.getenv("PATH_TO_MODEL", default="models/model.pkl")
    if model_path is None:
        raise RuntimeError("PATH_TO_MODEL is not specified")

    url = "https://drive.google.com/uc?id=1FfRG_vPj84INaZvC3tln332aTnzfBq_d"
    gdown.download(url, model_path, quiet=True)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)


@app.get("/")
def main():
    return "This is a service for the model inference"


@app.post('/predict', response_model=List[ConditionResponse])
def predict(request_data: HeartDiseaseData):
    data = pd.DataFrame([request_data.dict()])
    predictions = model.predict(data)
    response_list = [ConditionResponse(condition=pred) for pred in predictions]
    return response_list


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=os.getenv("PORT", 8000), reload=True)
