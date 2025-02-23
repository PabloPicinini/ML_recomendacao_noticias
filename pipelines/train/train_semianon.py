import os
import pickle
import logging
from typing import Optional, Dict, Any, List

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from script_shared import config

MODEL_DIR_SEMIANON = config.MODEL_DIR_SEMIANON
FEATURE_COLUMNS_SEMIANON = config.FEATURE_COLUMNS_SEMIANON

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def cap_outliers(
    df: pd.DataFrame, columns: List[str], quantile: float = 0.99
) -> pd.DataFrame:
    """
    Limita os valores de outliers para as colunas especificadas usando o quantil definido.
    """
    df = df.copy()
    for col in columns:
        upper_limit = df[col].quantile(quantile)
        df[col] = df[col].clip(upper=upper_limit)
    return df.fillna(0)


def treinar_modelo_semianon(
    df_semianon: pd.DataFrame,
    df_item: pd.DataFrame,
    df_semianon_raw: Optional[pd.DataFrame] = None,
    dias_limite: int = 30,
    best_k: int = 5,
    best_init: str = "random",
    best_max_iter: int = 600,
    n_components: int = 5,
) -> Dict[str, Any]:
    """
    Treina o modelo para usuários semi-logados usando dados já processados.

    Parâmetros:
      - df_semianon: DataFrame agregado dos usuários semi-logados (ex.: "users_semianon.parquet").
      - df_item: DataFrame dos itens (já processado).
      - df_semianon_raw: DataFrame com as interações individuais dos usuários semi-logados.
                        Se fornecido, será usado para gerar o mapeamento de itens populares por cluster.
      - dias_limite, best_k, best_init, best_max_iter, n_components: parâmetros para filtragem, PCA e clustering.

    Retorna um dicionário com:
      - kmeans_model: modelo KMeans treinado.
      - scaler: objeto StandardScaler.
      - pca: objeto PCA.
      - cluster_top_items: mapeamento dos itens populares por cluster.
      - df_features: DataFrame com as features e cluster para cada usuário.
    """
    logger.info(f"{df_semianon.shape[0]} usuários semi-logados agregados encontrados.")

    # Filtrar usuários recentes
    df_features = df_semianon[df_semianon["days_since_last"] <= dias_limite].copy()
    logger.info(
        f"{df_features.shape[0]} usuários semi-logados após filtragem por dias."
    )

    # Aplicar cap nos outliers e preencher valores faltantes
    df_features = cap_outliers(df_features, FEATURE_COLUMNS_SEMIANON)

    # Normalização e redução de dimensionalidade via PCA
    scaler = StandardScaler()
    X_raw = df_features[FEATURE_COLUMNS_SEMIANON].values
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(
        f"Soma da variância explicada pelo PCA: {pca.explained_variance_ratio_.sum():.2f}"
    )

    # Treinar KMeans
    kmeans_model = KMeans(
        n_clusters=best_k, init=best_init, max_iter=best_max_iter, random_state=42
    )
    kmeans_model.fit(X_pca)
    df_features["cluster"] = kmeans_model.labels_
    logger.info("Distribuição dos clusters:")
    logger.info(df_features["cluster"].value_counts())

    # Gerar mapeamento de itens populares por cluster (se dados individuais disponíveis)
    cluster_top_items: Dict[int, List[Any]] = {}
    if df_semianon_raw is not None:
        logger.info("Gerando mapeamento de itens populares por cluster...")
        df_merged = df_semianon_raw.merge(
            df_features[["userId", "cluster"]], on="userId", how="left"
        )
        for cluster in df_merged["cluster"].dropna().unique():
            top_items = (
                df_merged[df_merged["cluster"] == cluster]["history"]
                .value_counts()
                .head(10)
                .index.tolist()
            )
            cluster_top_items[int(cluster)] = top_items
        logger.info("Mapa de itens por cluster gerado.")
    else:
        logger.info("Dados individuais não fornecidos; usando mapeamento vazio.")

    # Salvar artefatos do modelo
    os.makedirs(MODEL_DIR_SEMIANON, exist_ok=True)
    with open(
        os.path.join(MODEL_DIR_SEMIANON, "modelo_semianon_kmeans.pkl"), "wb"
    ) as f:
        pickle.dump(kmeans_model, f)
    with open(os.path.join(MODEL_DIR_SEMIANON, "scaler_semianon.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR_SEMIANON, "pca_semianon.pkl"), "wb") as f:
        pickle.dump(pca, f)
    with open(os.path.join(MODEL_DIR_SEMIANON, "cluster_top_items.pkl"), "wb") as f:
        pickle.dump(cluster_top_items, f)
    df_features.to_csv(
        os.path.join(MODEL_DIR_SEMIANON, "df_features_semianon.csv"), index=False
    )

    logger.info("Modelo de usuários semi-logados treinado e salvo!")
    return {
        "kmeans_model": kmeans_model,
        "scaler": scaler,
        "pca": pca,
        "cluster_top_items": cluster_top_items,
        "df_features": df_features,
    }


def main():
    try:
        df_semianon = pd.read_parquet("/opt/airflow/shared/script_shared/data/refined/users_semianon.parquet")
        df_item = pd.read_parquet("/opt/airflow/shared/script_shared/data/refined/items.parquet")
        df_semianon_raw = pd.read_parquet("/opt/airflow/shared/script_shared/data/refined/users_semianon_raw.parquet")
    except Exception as e:
        logger.error("Erro ao carregar dados para treinamento semianon", exc_info=e)
        return

    treinar_modelo_semianon(
        df_semianon,
        df_item,
        df_semianon_raw,
        dias_limite=30,
        best_k=5,
        best_init="random",
        best_max_iter=600,
        n_components=5,
    )
    logger.info("Treinamento concluído.")


if __name__ == "__main__":
    main()
