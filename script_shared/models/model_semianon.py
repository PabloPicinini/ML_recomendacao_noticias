import os
import pickle
import logging
from typing import Dict, Any, List
import pandas as pd
from script_shared import config

MODEL_DIR_SEMIANON = config.MODEL_DIR_SEMIANON

# Configuração do logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model_semianon(model_dir: str = MODEL_DIR_SEMIANON) -> Dict[str, Any]:
    """
    Carrega os artefatos salvos para o modelo de usuários semi-logados.
    """
    with open(os.path.join(model_dir, "modelo_semianon_kmeans.pkl"), "rb") as f:
        kmeans_model = pickle.load(f)
    with open(os.path.join(model_dir, "scaler_semianon.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, "pca_semianon.pkl"), "rb") as f:
        pca = pickle.load(f)
    with open(os.path.join(model_dir, "cluster_top_items.pkl"), "rb") as f:
        cluster_top_items = pickle.load(f)
    df_features = pd.read_csv(os.path.join(model_dir, "df_features_semianon.csv"))

    return {
        "kmeans_model": kmeans_model,
        "scaler": scaler,
        "pca": pca,
        "cluster_top_items": cluster_top_items,
        "df_features": df_features,
    }


def recomendar_semianon(
    user_id: str, model_objs: Dict[str, Any], top_k: int = 10
) -> List[Any]:
    """
    Para um usuário semi-logado, retorna os top itens para o cluster ao qual o usuário pertence.
    Se o usuário não estiver presente no df_features, retorna um fallback.
    """
    df_features = model_objs["df_features"]
    cluster_top_items = model_objs["cluster_top_items"]

    user_info = df_features[df_features["userId"] == user_id]
    if not user_info.empty:
        cluster = int(user_info["cluster"].values[0])
        top_items = cluster_top_items.get(cluster, [])
        return top_items[:top_k]
    else:
        return []
