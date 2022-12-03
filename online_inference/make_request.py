import json
import logging
import sys
import requests

import pandas as pd


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(stream_handler)


data_path = "data/heart_disease.csv"
url = "http://localhost:8000/predict"

if __name__ == "__main__":
    data = pd.read_csv(data_path).to_dict(orient='records')
    for record in data:
        request = json.dumps(record)
        response = requests.post(url, request)
        logger.info(f"response code: {response.status_code}")
        logger.info(f"response data: {response.json()}")
