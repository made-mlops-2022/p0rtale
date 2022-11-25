from fastapi.testclient import TestClient
from server import app


def test_predict_ok():
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "age": 37,
            "sex": 1,
            "cp": 2,
            "trestbps": 130,
            "chol": 250,
            "fbs": 0,
            "restecg": 0,
            "thalach": 187,
            "exang": 0,
            "oldpeak": 3.5,
            "slope": 2,
            "ca": 0,
            "thal": 0
        })

        assert response.status_code == 200
        assert response.json() == [{"condition": 0}]

def test_predict_wrong_value():
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "age": 10,
            "sex": 1,
            "cp": 2,
            "trestbps": 130,
            "chol": 250,
            "fbs": 0,
            "restecg": 0,
            "thalach": 187,
            "exang": 0,
            "oldpeak": 3.5,
            "slope": 2,
            "ca": 0,
            "thal": 0
        })

        assert response.status_code == 422
        assert response.json()["detail"] == [{
            "loc": ["body", "age"],
            "msg": "must be between 20 and 90",
            "type": "value_error",
        }]
