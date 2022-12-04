import os
from airflow import DAG
from airflow.sensors.python import PythonSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

from args import (
    LOCAL_DATA_PATH,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    SPLITTED_DATA_PATH,
    MODEL_PATH,
    default_args
)


def _wait_for_file(file_name):
    return os.path.exists(file_name)

with DAG(
    "train_model",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(7),
) as dag:
    wait_data = PythonSensor(
        task_id='wait-for-data',
        python_callable=_wait_for_file,
        op_args=['/opt/airflow/data/raw/{{ ds }}/data.csv'],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    wait_target = PythonSensor(
        task_id='wait-for-target',
        python_callable=_wait_for_file,
        op_args=['/opt/airflow/data/raw/{{ ds }}/target.csv'],
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

    split_data = DockerOperator(
        image="airflow-split-data",
        command=f"--input-dir {PROCESSED_DATA_PATH} --output-dir {SPLITTED_DATA_PATH}",
        task_id="docker-airflow-split-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=LOCAL_DATA_PATH, target="/data", type="bind")]
    )

    train_model = DockerOperator(
        image="airflow-train-model",
        command=f"--input-dir {SPLITTED_DATA_PATH} --model-dir {MODEL_PATH}",
        task_id="docker-airflow-train-model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=LOCAL_DATA_PATH, target="/data", type="bind")]
    )

    validate_model = DockerOperator(
        image="airflow-validate-model",
        command=f"--input-dir {SPLITTED_DATA_PATH} --model-dir {MODEL_PATH} --output-dir {MODEL_PATH}",
        task_id="validate_model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=LOCAL_DATA_PATH, target="/data", type="bind")]
    )

    [wait_data, wait_target] >> preprocess_data >> split_data >> train_model >> validate_model
