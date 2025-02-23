from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

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
    "avaliacao_modelos",
    default_args=default_args,
    description="DAG para avaliar os modelos de recomendação",
    schedule_interval=None,
    catchup=False,
) as dag:
    evaluate_logged = BashOperator(
        task_id="avaliar_modelo_logged",
        bash_command="python -m pipelines.evaluate.evaluate_logged",
    )

    evaluate_semianon = BashOperator(
        task_id="avaliar_modelo_semianon",
        bash_command="python -m pipelines.evaluate.evaluate_semianon",
    )

    evaluate_logged >> evaluate_semianon
