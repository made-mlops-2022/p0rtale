from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

from args import LOCAL_DATA_PATH, RAW_DATA_PATH, default_args


with DAG(
    "generate_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(7),
) as dag:
    generate = DockerOperator(
        image="airflow-generate-data",
        command=f"--output-dir {RAW_DATA_PATH}",
        task_id="docker-airflow-generate-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=LOCAL_DATA_PATH, target="/data", type="bind")]
    )
