import os
import sys
import pickle
import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from implicit.als import AlternatingLeastSquares
from script_shared import config


MODEL_DIR_LOGGED = config.MODEL_DIR_LOGGED
ALS_DEFAULT_PARAMS = config.ALS_DEFAULT_PARAMS

# Configuração do logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model_logged(model_dir: str = MODEL_DIR_LOGGED) -> Dict[str, Any]:
    """
    Carrega os artefatos salvos do modelo para usuários logados.
    """
    logger.info(f"Carregando artefatos do modelo logado a partir de: {model_dir}")
    with open(os.path.join(model_dir, "objetos_logged_auxiliares.pkl"), "rb") as f:
        aux_dict = pickle.load(f)

    model_als = AlternatingLeastSquares(
        factors=ALS_DEFAULT_PARAMS["factors"],
        regularization=ALS_DEFAULT_PARAMS["regularization"],
        iterations=ALS_DEFAULT_PARAMS["iterations"],
        random_state=42,
    )
    data = np.load(os.path.join(model_dir, "model_logged_als.npz"))
    model_als.user_factors = data["user_factors"]
    model_als.item_factors = data["item_factors"]
    model_als._YtY = model_als.item_factors.T.dot(model_als.item_factors)

    tfidf_matrix = load_npz(os.path.join(model_dir, "tfidf_logged_matrix.npz"))
    logger.info("Modelo e artefatos carregados com sucesso.")
    return {"model_als": model_als, "aux_dict": aux_dict, "tfidf_matrix": tfidf_matrix}


def get_user_vector(
    user_id: str, df_historico: pd.DataFrame, item_to_idx: Dict[str, int]
) -> Optional[csr_matrix]:
    """
    Constrói o vetor de interações do usuário a partir do histórico.
    """
    df_user = df_historico[df_historico["userId"] == user_id]
    if df_user.empty:
        logger.warning("Histórico do usuário não encontrado.")
        return None

    item_scores = df_user.groupby("history")["final_score"].sum()
    indices, data = [], []
    for item, score in item_scores.items():
        if item in item_to_idx:
            indices.append(item_to_idx[item])
            data.append(score)

    n_items = len(item_to_idx)
    user_vector = csr_matrix((data, ([0] * len(indices), indices)), shape=(1, n_items))
    return user_vector


def recomendar_logged(
    user_id: str,
    model_objs: Dict[str, Any],
    df_historico: pd.DataFrame,
    top_k: int = 10,
) -> List[str]:
    """
    Gera recomendações para o usuário logado com base no modelo ALS.
    """
    aux_dict = model_objs["aux_dict"]
    user_to_idx = aux_dict["user_to_idx"]
    item_to_idx = aux_dict["item_to_idx"]

    if user_id not in user_to_idx:
        logger.warning("Usuário não encontrado; retornando fallback vazio.")
        return []

    user_vector = get_user_vector(user_id, df_historico, item_to_idx)
    if user_vector is None:
        logger.warning(
            "Histórico do usuário não encontrado; retornando fallback vazio."
        )
        return []

    recs_raw = model_objs["model_als"].recommend(
        user_to_idx[user_id], user_vector, N=top_k, recalculate_user=True
    )
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    rec_ids = [idx_to_item[r[0]] for r in recs_raw if r[0] in idx_to_item]

    return rec_ids