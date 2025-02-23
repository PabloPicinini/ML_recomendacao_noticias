import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from script_shared.models.model_logged import recomendar_logged, load_model_logged
from pipelines.utils.metrics import salvar_metricas_csv

MODEL_DIR_LOGGED = "/opt/airflow/shared/script_shared/models/logged"
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def avaliar_modelo_logged(
    model_objs: Dict[str, Any],
    df_validacao: pd.DataFrame,
    df_users_logged: pd.DataFrame,
    top_k: int = 10,
    max_users: Optional[int] = 10,
) -> Dict[str, float]:
    """
    Avalia o desempenho do modelo logado usando mean_recall e mean_ndcg.

    Para cada usuário presente no conjunto de validação (df_validacao) e no modelo,
    gera recomendações (com base no modelo ALS) e compara com as interações reais (ground truth)
    contidas em df_validacao. Calcula a média de recall e NDCG.

    Args:
        model_objs: Dicionário com os artefatos do modelo logado.
        df_validacao: DataFrame de validação contendo, ao menos, as colunas 'userId' e 'page'.
        df_users_logged: DataFrame com as interações dos usuários logados (para gerar recomendações).
        top_k: Número de recomendações consideradas (default 10).
        max_users: Se definido, limita a avaliação aos primeiros N usuários.

    Returns:
        Um dicionário com as métricas: "mean_recall" e "mean_ndcg".
    """
    recall_scores = []
    ndcg_scores = []

    # Usuários presentes tanto no modelo quanto no conjunto de validação
    users_model = set(model_objs["aux_dict"]["user_to_idx"].keys())
    users_valid = set(df_validacao["userId"].unique())
    users_avaliacao = list(users_model.intersection(users_valid))

    if max_users is not None:
        users_avaliacao = users_avaliacao[:max_users]

    if not users_avaliacao:
        logger.warning("Nenhum usuário de validação encontrado no modelo.")
        return {"mean_recall": 0.0, "mean_ndcg": 0.0}

    for user in users_avaliacao:
        # Obter recomendações para o usuário
        recs = recomendar_logged(user, model_objs, df_users_logged, top_k)

        # Ground truth: itens com os quais o usuário interagiu no conjunto de validação
        ground_truth = list(set(df_validacao[df_validacao["userId"] == user]["page"]))
        if not ground_truth:
            continue

        # Cálculo do Recall para o usuário
        intersec = set(recs).intersection(set(ground_truth))
        recall = len(intersec) / float(len(ground_truth))
        recall_scores.append(recall)

        # Cálculo do DCG para as recomendações (relevância binária: 1 se item estiver no ground truth, 0 caso contrário)
        dcg = 0.0
        for i, item in enumerate(recs):
            if item in ground_truth:
                dcg += 1.0 / np.log2(
                    i + 2
                )  # i+2 pois a posição inicia em 1 (log2(1+1)=log2(2))

        # Cálculo do IDCG: soma dos melhores valores possíveis até top_k
        ideal_count = min(len(ground_truth), top_k)
        idcg = sum([1.0 / np.log2(i + 2) for i in range(ideal_count)])

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    mean_recall = np.mean(recall_scores) if recall_scores else 0.0
    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

    metrics = {"mean_recall": mean_recall, "mean_ndcg": mean_ndcg}
    logger.info(f"Métricas de avaliação: {metrics}")

    # Salva as métricas em CSV sem apagar dados anteriores
    salvar_metricas_csv(metrics, type_model="logged")
    return metrics


def main():
    try:
        df_validacao = pd.read_parquet("/opt/airflow/shared/script_shared/data/refined/validacao.parquet")
        df_users_logged = pd.read_parquet(
            "/opt/airflow/shared/script_shared/data/refined/users_logged.parquet"
        )
    except Exception as e:
        logger.error(
            "Erro ao carregar dados de validação ou usuários logados", exc_info=e
        )
        return

    try:
        model_objs = load_model_logged(MODEL_DIR_LOGGED)
    except Exception as e:
        logger.error("Erro ao carregar o modelo logado", exc_info=e)
        return

    avaliar_modelo_logged(
        model_objs, df_validacao, df_users_logged, top_k=10, max_users=50
    )


if __name__ == "__main__":
    main()
