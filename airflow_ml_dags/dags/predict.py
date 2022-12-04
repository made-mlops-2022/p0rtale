import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

from args import (
    LOCAL_DATA_PATH,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    PREDICT_PATH,
    default_args
)


def _wait_for_file(file_name):
    return os.path.exists(file_name)

with DAG(
    "predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(7),
) as dag:
    wait_data = PythonSensor(
        task_id='wait-for-predict-data',
        python_callable=_wait_for_file,
        op_args=['/opt/airflow/data/raw/{{ ds }}/data.csv'],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    preprocess_data = DockerOperator(
        image="airflow-preprocess-data",
        command=f"--input-dir {RAW_DATA_PATH} --output-dir {PROCESSED_DATA_PATH}",
        task_id="docker-airflow-preprocess-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=LOCAL_DATA_PATH, target="/data", type="bind")]
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input-dir {PROCESSED_DATA_PATH} --model-dir {MODEL_PATH} --output-dir {PREDICT_PATH}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=LOCAL_DATA_PATH, target="/data", type="bind")]
    )

    wait_data >> preprocess_data >> predict
