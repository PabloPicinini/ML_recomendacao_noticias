import os
import pickle
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np
from script_shared import config

MODEL_DIR_ANON_HEURISTICO = config.MODEL_DIR_ANON_HEURISTICO
DEFAULT_W_ISSUED = config.DEFAULT_W_ISSUED
DEFAULT_W_MODIFIED = config.DEFAULT_W_MODIFIED


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def calcular_score_heuristico(
    df_item: pd.DataFrame,
    w_issued: float = DEFAULT_W_ISSUED,
    w_modified: float = DEFAULT_W_MODIFIED,
) -> pd.DataFrame:
    """
    Calcula um score para cada notícia combinando a recência da publicação ('issued')
    e a data de modificação ('modified') com pesos definidos.

    Parâmetros:
      - df_item: DataFrame com as colunas 'issued', 'modified' e 'page'.
      - w_issued: peso para a data de publicação.
      - w_modified: peso para a data de modificação.

    Retorna:
      - DataFrame com as colunas calculadas e ordenado de forma decrescente pelo score.
    """
    logger.info("Calculando score heurístico dos itens.")

    # Converter colunas para datetime e garantir que sejam tz-naive
    df_item["issued_timestamp"] = pd.to_datetime(
        df_item["issued"], errors="coerce"
    ).apply(lambda x: x.replace(tzinfo=None) if pd.notnull(x) else x)
    df_item["modified_timestamp"] = pd.to_datetime(
        df_item["modified"], errors="coerce"
    ).apply(lambda x: x.replace(tzinfo=None) if pd.notnull(x) else x)

    now = pd.Timestamp.now(tz=None)  # Timestamp atual tz-naive

    # Calcular a diferença em horas para as datas
    df_item["hours_diff_issued"] = (
        now - df_item["issued_timestamp"]
    ).dt.total_seconds() / 3600.0
    df_item["hours_diff_modified"] = (
        now - df_item["modified_timestamp"]
    ).dt.total_seconds() / 3600.0

    # Aplicar função de decaimento exponencial (constante de 24 horas)
    df_item["time_decay_issued"] = np.exp(-df_item["hours_diff_issued"] / 24)
    df_item["time_decay_modified"] = np.exp(-df_item["hours_diff_modified"] / 24)

    # Calcular score final combinando os decaimentos com os respectivos pesos
    df_item["score"] = (
        w_issued * df_item["time_decay_issued"]
        + w_modified * df_item["time_decay_modified"]
    )

    # Ordenar o DataFrame pelo score de forma decrescente e resetar o índice
    df_item_sorted = df_item.sort_values(by="score", ascending=False).reset_index(
        drop=True
    )
    logger.info("Score heurístico calculado com sucesso.")
    return df_item_sorted


def treinar_modelo_anon_heuristico(
    df_item: pd.DataFrame, model_dir: str = MODEL_DIR_ANON_HEURISTICO
) -> Dict[str, Any]:
    """
    Treina o modelo anônimo baseado em heurísticas, calculando o ranking dos itens
    a partir das datas de publicação e modificação, e salva o ranking em disco.

    Parâmetros:
      - df_item: DataFrame com informações dos itens (deve conter 'issued', 'modified' e 'page').
      - model_dir: diretório para salvar o ranking calculado.

    Retorna:
      - Dicionário com o ranking e o método utilizado.
    """
    logger.info("Treinando modelo anônimo heurístico...")
    df_ranked = calcular_score_heuristico(df_item)
    ranking_item_ids = df_ranked["page"].tolist()

    os.makedirs(model_dir, exist_ok=True)
    ranking_path = os.path.join(model_dir, "ranking_anon_heuristico.pkl")
    with open(ranking_path, "wb") as f:
        pickle.dump(ranking_item_ids, f)

    logger.info("Modelo anônimo heurístico treinado e salvo com sucesso.")
    return {"ranking_anon": ranking_item_ids, "method": "heurístico"}


def main():
    model_dir = MODEL_DIR_ANON_HEURISTICO
    try:
        df_item = pd.read_parquet("/opt/airflow/shared/script_shared/data/refined/items.parquet")
        treinar_modelo_anon_heuristico(df_item, model_dir)
    except Exception as e:
        logger.warning(
            "Não foi possível treinar o modelo anônimo heurístico.", exc_info=e
        )


if __name__ == "__main__":
    main()
