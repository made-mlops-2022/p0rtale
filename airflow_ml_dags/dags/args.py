from datetime import timedelta
from airflow.models import Variable


LOCAL_DATA_PATH = "/home/portale/Документы/TP/MLOPS/p0rtale/airflow_ml_dags/data/"
RAW_DATA_PATH = "/data/raw/{{ ds }}"
PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
SPLITTED_DATA_PATH = "/data/splitted/{{ ds }}"
MODEL_PATH = Variable.get("model_path")
PREDICT_PATH = "/data/predicted/{{ ds }}"


default_args = {
    "owner": "p0rtale",
    "email": ["portale888@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retry_delay": timedelta(minutes=5),
    "retries": 1,
}
