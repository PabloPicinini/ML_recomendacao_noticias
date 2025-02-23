import logging
from typing import Dict, Any

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz

# Configuração do logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_type_logged(
    df_user_clean: pd.DataFrame, engagement_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Processa os dados para usuários logados.
    Calcula o score de engajamento, cria a coluna 'final_score' e mapeia usuários e itens.

    Args:
        df_user_clean: DataFrame contendo os dados de usuários.
        engagement_params: Parâmetros de engajamento com chaves:
            - 'w_time'
            - 'w_clicks'
            - 'w_scroll'
            - 'w_visits'

    Returns:
        DataFrame com os usuários logados processados, contendo as colunas:
        'engagement', 'final_score', 'user_idx' e 'item_idx'.
    """
    logger.info("📌 Processando usuários logados...")

    # Filtrar usuários logados e criar cópia
    df_users_logged = df_user_clean[df_user_clean["userType"] == "Logged"].copy()

    # Calcular engagement score
    df_users_logged["engagement"] = (
        np.log1p(df_users_logged["timeOnPageHistory"] / 60000.0)
        * engagement_params["w_time"]
        + df_users_logged["numberOfClicksHistory"] * engagement_params["w_clicks"]
        + df_users_logged["scrollPercentageHistory"] * engagement_params["w_scroll"]
        + df_users_logged["pageVisitsCountHistory"] * engagement_params["w_visits"]
    )

    # Criar final_score (atualmente igual ao engagement)
    df_users_logged["final_score"] = df_users_logged["engagement"]

    # Mapear usuários e itens para IDs numéricos
    user_to_idx = {u: i for i, u in enumerate(df_users_logged["userId"].unique())}
    item_to_idx = {p: i for i, p in enumerate(df_users_logged["history"].unique())}
    df_users_logged["user_idx"] = df_users_logged["userId"].map(user_to_idx)
    df_users_logged["item_idx"] = df_users_logged["history"].map(item_to_idx)

    logger.info(f"✅ {len(df_users_logged)} usuários logados processados.")
    return df_users_logged


def process_type_semianon(
    df_user_clean: pd.DataFrame, engagement_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Processa os dados para usuários semi-logados.

    Args:
        df_user_clean: DataFrame contendo os dados dos usuários.
        engagement_params: Parâmetros de engajamento com, ao menos, a chave 'dias_limite'.

    Returns:
        DataFrame com os usuários semi-logados agregados e processados.
    """
    logger.info("📌 Processando usuários semi-logados...")

    # Selecionar usuários não logados e filtrar para usuários com mais de 1 interação
    df_anon = df_user_clean[df_user_clean["userType"] == "Non-Logged"].copy()
    valid_users = df_anon["userId"].value_counts()[lambda x: x > 1].index
    df_multi = df_anon[df_anon["userId"].isin(valid_users)].copy()

    logger.info(f"✅ {len(valid_users)} usuários semi-logados identificados.")

    # Agregar dados por usuário
    df_users_semianon = (
        df_multi.groupby("userId")
        .agg(
            sum_time=(
                "timeOnPageHistory",
                lambda x: np.sum(x) / 60000.0,
            ),  # tempo total em minutos
            mean_scroll=("scrollPercentageHistory", "mean"),
            sum_clicks=("numberOfClicksHistory", "sum"),
            unique_pages=("history", "nunique"),
            last_ts=("timestampHistory", "max"),
        )
        .reset_index()
    )

    # Processamento de timestamps e cálculo de dias
    df_users_semianon["last_ts"] = pd.to_numeric(
        df_users_semianon["last_ts"], errors="coerce"
    )
    df_users_semianon["last_ts_days"] = df_users_semianon["last_ts"] / (
        1000 * 60 * 60 * 24
    )
    max_global_days = df_users_semianon["last_ts_days"].max()
    df_users_semianon["days_since_last"] = (
        max_global_days - df_users_semianon["last_ts_days"]
    )

    # Filtrar usuários dentro do limite de dias
    df_users_semianon_filtered = df_users_semianon[
        df_users_semianon["days_since_last"] <= engagement_params["dias_limite"]
    ].copy()

    # Calcular métricas por página e tratar possíveis divisões por zero
    df_users_semianon_filtered["time_per_page"] = (
        (
            df_users_semianon_filtered["sum_time"]
            / df_users_semianon_filtered["unique_pages"]
        )
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    df_users_semianon_filtered["clicks_per_page"] = (
        (
            df_users_semianon_filtered["sum_clicks"]
            / df_users_semianon_filtered["unique_pages"]
        )
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    # Aplicar transformação logarítmica nas colunas selecionadas
    for col in ["sum_time", "sum_clicks", "time_per_page", "clicks_per_page"]:
        df_users_semianon_filtered[f"log_{col}"] = np.log1p(
            df_users_semianon_filtered[col]
        )

    # Limitar outliers para as features
    feature_cols = [
        "sum_time",
        "sum_clicks",
        "mean_scroll",
        "unique_pages",
        "time_per_page",
        "clicks_per_page",
        "log_sum_time",
        "log_sum_clicks",
        "log_time_per_page",
        "log_clicks_per_page",
    ]
    for col in feature_cols:
        upper_p = df_users_semianon_filtered[col].quantile(0.99)
        df_users_semianon_filtered[col] = df_users_semianon_filtered[col].clip(
            upper=upper_p
        )

    df_users_semianon_filtered.fillna(0, inplace=True)
    logger.info(
        f"✅ {len(df_users_semianon_filtered)} usuários semi-logados processados."
    )
    return df_users_semianon_filtered


def build_and_save_sparse_matrix(
    df_users_logged: pd.DataFrame, output_path: str
) -> csr_matrix:
    """
    Constrói a matriz esparsa usuário-item a partir do DataFrame de usuários logados
    e a salva em formato NPZ.

    Args:
        df_users_logged: DataFrame processado de usuários logados, contendo as colunas 'user_idx',
                         'item_idx' e 'final_score'.
        output_path: Caminho onde a matriz esparsa será salva.

    Returns:
        A matriz esparsa construída (csr_matrix).
    """
    # Assume que os mapeamentos já foram criados em process_type_logged
    users = df_users_logged["userId"].unique()
    items = df_users_logged["history"].unique()

    rows = df_users_logged["user_idx"].values
    cols = df_users_logged["item_idx"].values
    vals = df_users_logged["final_score"].values

    sparse_mat = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
    save_npz(output_path, sparse_mat)
    logger.info(f"✅ Matriz esparsa salva em: {output_path}")
    return sparse_mat
