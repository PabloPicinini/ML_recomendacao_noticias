from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Defina os argumentos padrão para o DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "processamento_data",
    default_args=default_args,
    description="DAG para processar os dados",
    schedule_interval=None,
    catchup=False,
) as dag:
    download_data = BashOperator(
        task_id="download_data",
        bash_command="python -m pipelines.download_data",
    )

    process_data = BashOperator(
        task_id="process_data",
        bash_command="python -m pipelines.process_data",
    )


    download_data >> process_data
