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
    "treinamento_modelos",
    default_args=default_args,
    description="DAG para treinar os modelos de recomendação",
    schedule_interval=None, 
    catchup=False,
) as dag:
    train_logged = BashOperator(
        task_id="treinar_modelo_logged",
        bash_command="python -m pipelines.train.train_logged",
    )

    train_semianon = BashOperator(
        task_id="treinar_modelo_semianon",
        bash_command="python -m pipelines.train.train_semianon",
    )

    train_anon = BashOperator(
        task_id="treinar_modelo_anon",
        bash_command="python -m pipelines.train.train_anon",
    )

    train_logged >> train_semianon >> train_anon
