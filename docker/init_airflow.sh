#!/bin/bash
set -e

echo "Instalando gdown..."
pip install --user gdown

echo "Aguardando o Airflow estar pronto..."
sleep 30

echo "Criando usuário admin (se ainda não existir)..."
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com || true

echo "Despausando as DAGs necessárias..."
airflow dags unpause processamento_data

echo "Disparando a DAG processamento_data..."
# Definir a execution_date (em UTC) no formato ISO sem fuso
execution_date=$(date -u +"%Y-%m-%dT%H:%M:%S")
echo "Execution date: $execution_date"

# Disparar a DAG utilizando a execution_date definida
trigger_output=$(airflow dags trigger processamento_data --exec-date "$execution_date")
echo "$trigger_output"

check_status() {
  airflow dags state processamento_data "$execution_date"
}

while true; do
  state=$(check_status)
  echo "Estado atual da DAG processamento_data: $state"
  if [ "$state" == "success" ]; then
    echo "DAG processamento_data finalizada com sucesso."
    break
  elif [ "$state" == "failed" ]; then
    echo "DAG processamento_data falhou."
    exit 1
  fi
  sleep 10
done

echo "Disparando a DAG treinamento_modelos..."
airflow dags unpause treinamento_modelos
airflow dags trigger treinamento_modelos --exec-date "$execution_date"

echo "Processo finalizado, container saindo."
