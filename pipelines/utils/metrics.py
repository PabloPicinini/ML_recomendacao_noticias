import os
import datetime
import pandas as pd


def salvar_metricas_csv(
    metrics: dict,
    type_model: str = "logged",
) -> None:
    """
    Salva (ou adiciona) as métricas de avaliação em um arquivo CSV.

    Se o arquivo já existir, a nova linha é adicionada sem sobrescrever os dados anteriores.
    Verifica se o diretório existe e o cria, se necessário.

    Args:
        metrics: Dicionário com as métricas de avaliação (ex.: {"mean_recall": ..., "mean_ndcg": ...}).
        type_model: Tipo de modelo avaliado (ex.: "logged").
        csv_path: Caminho do arquivo CSV onde as métricas serão salvas.
    """
    csv_path = os.path.abspath("/opt/airflow/shared/script_shared/evaluation/evaluation_metrics.csv")
    print(f"Salvando métricas em {csv_path}...")

    # Adiciona um timestamp para registrar quando a avaliação foi feita
    timestamp = datetime.datetime.now().isoformat()
    # Cria uma nova linha com as colunas desejadas
    new_row = {
        "type_model": type_model,
        "mean_recall": metrics.get("mean_recall", 0.0),
        "mean_ndcg": metrics.get("mean_ndcg", 0.0),
        "timestamp": timestamp,
    }
    df_novo = pd.DataFrame([new_row])

    try:
        if os.path.exists(csv_path):
            df_novo.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df_novo.to_csv(csv_path, index=False)

        print(f"Métricas salvas com sucesso em {csv_path}")
    except Exception as e:
        print(f"Erro ao salvar métricas: {e}")
