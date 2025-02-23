import logging
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

from script_shared.models.model_semianon import recomendar_semianon, load_model_semianon
from pipelines.utils.metrics import salvar_metricas_csv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def avaliar_modelo_semianon(
    model_objs: Dict[str, Any],
    df_validacao: pd.DataFrame,
    df_users_semianon: pd.DataFrame,
    top_k: int = 10,
    max_users: Optional[int] = 10,
) -> Dict[str, float]:
    """
    Avalia o desempenho do modelo semianon usando mean_recall e mean_ndcg.

    Para cada usuário presente no conjunto de validação (df_validacao) e no modelo semianon,
    gera recomendações (com base no clustering) e compara com as interações reais (ground truth)
    contidas em df_validacao. Calcula a média de recall e NDCG.

    Args:
        model_objs: Dicionário com os artefatos do modelo semianon.
        df_validacao: DataFrame de validação contendo, ao menos, as colunas 'userId' e 'page'.
        df_users_semianon: DataFrame com as features dos usuários semianon (utilizado para gerar recomendações).
        top_k: Número de recomendações consideradas (default 10).
        max_users: Se definido, limita a avaliação aos primeiros N usuários.
        csv_path: Caminho do CSV onde as métricas serão salvas.

    Returns:
        Um dicionário com as métricas: "mean_recall" e "mean_ndcg".
    """
    recall_scores = []
    ndcg_scores = []

    # Considera os usuários presentes no modelo semianon (df_features salvo em model_objs)
    users_model = set(model_objs["df_features"]["userId"].unique())
    users_valid = set(df_validacao["userId"].unique())
    users_avaliacao = list(users_model.intersection(users_valid))

    if max_users is not None:
        users_avaliacao = users_avaliacao[:max_users]

    if not users_avaliacao:
        logger.warning("Nenhum usuário de validação encontrado no modelo semianon.")
        metrics = {"mean_recall": 0.0, "mean_ndcg": 0.0}
        salvar_metricas_csv(metrics, type_model="semianon")
        return metrics

    for user in users_avaliacao:
        # Obter recomendações para o usuário a partir do modelo semianon
        recs = recomendar_semianon(user, model_objs, top_k)

        # Ground truth: itens com os quais o usuário interagiu no conjunto de validação
        ground_truth = list(set(df_validacao[df_validacao["userId"] == user]["page"]))
        if not ground_truth:
            continue

        # Cálculo do Recall para o usuário
        intersec = set(recs).intersection(set(ground_truth))
        recall = len(intersec) / float(len(ground_truth))
        recall_scores.append(recall)

        # Cálculo do DCG (Discounted Cumulative Gain) para as recomendações (relevância binária)
        dcg = 0.0
        for i, item in enumerate(recs):
            if item in ground_truth:
                dcg += 1.0 / np.log2(i + 2)  # i+2 pois a posição inicia em 1
        # Cálculo do IDCG (Ideal DCG) para as primeiras top_k posições
        ideal_count = min(len(ground_truth), top_k)
        idcg = sum([1.0 / np.log2(i + 2) for i in range(ideal_count)])
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    mean_recall = np.mean(recall_scores) if recall_scores else 0.0
    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    metrics = {"mean_recall": mean_recall, "mean_ndcg": mean_ndcg}

    logger.info(f"Métricas de avaliação semianon: {metrics}")
    salvar_metricas_csv(metrics, type_model="semianon")
    return metrics


def main():
    try:
        df_validacao = pd.read_parquet("/opt/airflow/shared/script_shared/data/refined/validacao.parquet")
        df_users_semianon = pd.read_parquet("/opt/airflow/shared/script_shared/data/refined/users_semianon.parquet")
    except Exception as e:
        logger.error("Erro ao carregar dados de validação ou semianon", exc_info=e)
        return

    try:
        model_objs = load_model_semianon()
    except Exception as e:
        logger.error("Erro ao carregar o modelo semianon", exc_info=e)
        return

    avaliar_modelo_semianon(
        model_objs, df_validacao, df_users_semianon, top_k=10, max_users=50
    )


if __name__ == "__main__":
    main()
