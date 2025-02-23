import os
import pickle
import logging
from typing import Dict, Any, Optional
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import load_npz, save_npz
from implicit.als import AlternatingLeastSquares
from sklearn.feature_extraction.text import TfidfVectorizer
from script_shared import config

ALS_DEFAULT_PARAMS = config.ALS_DEFAULT_PARAMS
SPARSE_MATRIX_PATH = config.SPARSE_MATRIX_PATH
MODEL_DIR_LOGGED = config.MODEL_DIR_LOGGED

logger = logging.getLogger(__name__)


def build_sparse_matrix(df_users: pd.DataFrame) -> sp.csr_matrix:
    """
    Constrói a matriz esparsa a partir do DataFrame de usuários logados.
    """
    logger.info("Construindo a matriz esparsa a partir do DataFrame...")
    users = df_users["userId"].unique()
    items = df_users["history"].unique()
    rows = df_users["user_idx"].values
    cols = df_users["item_idx"].values
    vals = df_users["final_score"].values
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
    logger.info("Matriz esparsa construída.")
    return matrix


def treinar_modelo_logged(
    df_users_logged: pd.DataFrame,
    df_item: pd.DataFrame,
    als_params: Optional[Dict[str, Any]] = None,
    weight_cf: float = 0.25,
    top_n_cf: int = 120,
) -> Dict[str, Any]:
    """
    Treina e salva os modelos ALS e TF-IDF para usuários logados.
    """
    als_params = als_params or ALS_DEFAULT_PARAMS

    num_users = df_users_logged["userId"].nunique()
    logger.info(f"{num_users} usuários logados encontrados.")

    # Carregar ou construir a matriz esparsa
    if os.path.exists(SPARSE_MATRIX_PATH):
        logger.info(f"Carregando matriz esparsa de: {SPARSE_MATRIX_PATH}")
        sparse_mat = load_npz(SPARSE_MATRIX_PATH)
        logger.info("Matriz esparsa carregada.")
    else:
        sparse_mat = build_sparse_matrix(df_users_logged)

    # Treinamento do modelo ALS
    logger.info("Treinando modelo ALS...")
    model_als = AlternatingLeastSquares(
        factors=als_params["factors"],
        regularization=als_params["regularization"],
        iterations=als_params["iterations"],
        random_state=42,
    )
    model_als.fit(sparse_mat * als_params["alpha"])
    logger.info("Modelo ALS treinado.")

    # Treinamento do modelo TF‑IDF
    logger.info("Preparando dados para TF‑IDF...")
    df_item["text_content"] = (
        df_item["title"].fillna("") + " " + df_item["body"].fillna("")
    )
    logger.info("Treinando modelo TF‑IDF...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = tfidf.fit_transform(df_item["text_content"])
    logger.info("Modelo TF‑IDF treinado.")

    # Montar mapeamentos para inferência
    logger.info("Montando mapeamentos para inferência...")
    user_ids = df_users_logged["userId"].unique()
    item_ids_history = df_users_logged["history"].unique()
    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    item_to_idx = {item: idx for idx, item in enumerate(item_ids_history)}
    item_to_idx_content = {p: i for i, p in enumerate(df_item["page"].values)}

    aux_dict = {
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "item_to_idx_content": item_to_idx_content,
        "weight_cf": weight_cf,
        "top_n_cf": top_n_cf,
        "tfidf": tfidf,
    }

    # Salvando os artefatos treinados
    os.makedirs(MODEL_DIR_LOGGED, exist_ok=True)
    logger.info(f"Salvando modelo e artefatos em: {MODEL_DIR_LOGGED}")
    model_als.save(os.path.join(MODEL_DIR_LOGGED, "model_logged_als.npz"))
    with open(
        os.path.join(MODEL_DIR_LOGGED, "objetos_logged_auxiliares.pkl"), "wb"
    ) as f:
        pickle.dump(aux_dict, f)
    save_npz(os.path.join(MODEL_DIR_LOGGED, "tfidf_logged_matrix.npz"), tfidf_matrix)

    logger.info("Modelo logado treinado e salvo com sucesso.")
    return {"model_als": model_als, "aux_dict": aux_dict, "tfidf_matrix": tfidf_matrix}


def main():
    logger.info("Carregando dados processados...")
    df_users_logged = pd.read_parquet("/opt/airflow/shared/script_shared/data/refined/users_logged.parquet")
    df_item = pd.read_parquet("/opt/airflow/shared/script_shared/data/refined/items.parquet")
    treinar_modelo_logged(df_users_logged, df_item)


if __name__ == "__main__":
    main()
