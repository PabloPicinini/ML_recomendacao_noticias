import os
import logging
import glob

import pandas as pd
import numpy as np
from pipelines.process_type_user import (
    process_type_logged,
    process_type_semianon,
    build_and_save_sparse_matrix,
)

# Configuração do logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constantes
RAW_USER_PATH = ("/opt/airflow/shared/script_shared/data/raw/files/treino/")
RAW_ITEM_PATH = ("/opt/airflow/shared/script_shared/data/raw/itens/itens/")
VALIDACAO_PATH = ("/opt/airflow/shared/script_shared/data/raw/validacao.csv")

REFINED_DIR = ("/opt/airflow/shared/script_shared/data/refined")
STAGE_DIR = ("/opt/airflow/shared/script_shared/data/stage")

def tratar_outliers_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers do dataset de usuários com base no percentil 99 para colunas específicas.

    Args:
        df: DataFrame contendo os dados dos usuários.

    Returns:
        DataFrame sem os registros considerados outliers.
    """
    df = df.copy()
    registros_iniciais = len(df)
    logger.info(f"Registros iniciais para remoção de outliers: {registros_iniciais}")

    limites = {
        "historySize": np.percentile(df["historySize"], 99),
        "numberOfClicksHistory": np.percentile(df["numberOfClicksHistory"], 99),
        "timeOnPageHistory": 1_800_000,  # 30 minutos em milissegundos
        "scrollPercentageHistory": 100,
        "pageVisitsCountHistory": np.percentile(df["pageVisitsCountHistory"], 99),
    }

    for col, limite in limites.items():
        registros_removidos = len(df[df[col] > limite])
        df = df[df[col] <= limite]
        logger.info(
            f"{col}: {registros_removidos} registros removidos (limite: {limite})"
        )

    registros_finais = len(df)
    logger.info(f"Registros finais sem outliers: {registros_finais}")
    logger.info(
        f"Total de registros removidos: {registros_iniciais - registros_finais}"
    )
    return df


def salvar_dataframe(df: pd.DataFrame, nome_arquivo: str) -> None:
    """
    Salva um DataFrame em formato Parquet no diretório refined.

    Args:
        df: DataFrame a ser salvo.
        nome_arquivo: Nome do arquivo (sem extensão).
    """
    caminho_arquivo =f"{REFINED_DIR}/{nome_arquivo}.parquet"
    df.to_parquet(caminho_arquivo, index=False)
    logger.info(f"✅ {nome_arquivo} salvo em: {caminho_arquivo}")


def processar_usuarios(engagement_params: dict = None) -> None:
    """
    Processa e salva os dados de usuários.

    Args:
        engagement_params: Dicionário com parâmetros de engajamento. Se None, utiliza os valores padrão.
                          Valores padrão: {"w_time": 0.25, "w_clicks": 1.7, "w_scroll": 0.35, "w_visits": 2.2, "dias_limite": 30}
    """
    engagement_params = engagement_params or {
        "w_time": 0.25,
        "w_clicks": 1.7,
        "w_scroll": 0.35,
        "w_visits": 2.2,
        "dias_limite": 30,
    }

    logger.info("Carregando dados de usuários...")
    csv_paths = glob.glob((RAW_USER_PATH) + "*.csv") 

    if not csv_paths:
        logger.error(f"Nenhum arquivo CSV encontrado em: {str(RAW_USER_PATH)}")
        return

    dfs = [pd.read_csv(p, delimiter=",") for p in csv_paths]
    if not dfs:
        logger.error("Nenhum DataFrame foi carregado. Abortando.")
        return

    dfs = [pd.read_csv(p, delimiter=",") for p in csv_paths]
    df_users_raw = pd.concat(dfs, ignore_index=True)

    # Explodir colunas com listas
    list_columns = [
        "history",
        "timestampHistory",
        "numberOfClicksHistory",
        "timeOnPageHistory",
        "scrollPercentageHistory",
        "pageVisitsCountHistory",
        "timestampHistory_new",
    ]
    for col in list_columns:
        df_users_raw[col] = df_users_raw[col].astype(str).str.split(",")

    df_users_stage = df_users_raw.explode(list_columns).reset_index(drop=True)

    # Converter colunas numéricas
    numeric_cols = [
        "historySize",
        "numberOfClicksHistory",
        "timeOnPageHistory",
        "scrollPercentageHistory",
        "pageVisitsCountHistory",
    ]
    df_users_stage[numeric_cols] = df_users_stage[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Remover outliers
    df_users_clean = tratar_outliers_users(df_users_stage)

    # Limpar espaços em branco em colunas de string
    string_cols = df_users_clean.select_dtypes(include="object").columns
    df_users_clean[string_cols] = df_users_clean[string_cols].apply(
        lambda x: x.str.strip()
    )

    # Processar usuários logados e semi-anônimos
    df_users_logged = process_type_logged(df_users_clean, engagement_params)
    df_users_semianon = process_type_semianon(df_users_clean, engagement_params)

    # Salvar DataFrames processados
    salvar_dataframe(df_users_clean, "users_clean")
    salvar_dataframe(df_users_logged, "users_logged")
    salvar_dataframe(df_users_semianon, "users_semianon")

    # Salvar interações individuais dos usuários semi-logados
    df_semianon_raw = df_users_clean[df_users_clean["userType"] == "Non-Logged"].copy()
    valid_users = df_semianon_raw["userId"].value_counts()[lambda x: x > 1].index
    df_semianon_raw = df_semianon_raw[
        df_semianon_raw["userId"].isin(valid_users)
    ].copy()
    salvar_dataframe(df_semianon_raw, "users_semianon_raw")

    # Construir e salvar a matriz esparsa dos usuários logados
    output_sparse_path = f"{REFINED_DIR}/user_item_sparse_mat_logged.npz"
    build_and_save_sparse_matrix(df_users_logged, output_sparse_path)


def processar_itens() -> None:
    """
    Processa e salva os dados dos itens.
    """
    logger.info("Processando itens...")
    csv_paths = glob.glob(RAW_ITEM_PATH + "*.csv")
    
    dfs = [pd.read_csv(p, delimiter=",") for p in csv_paths]
    df_items = pd.concat(dfs, ignore_index=True)

    # Limpar espaços em branco em colunas de string
    string_cols = df_items.select_dtypes(include="object").columns
    df_items[string_cols] = df_items[string_cols].apply(lambda x: x.str.strip())

    salvar_dataframe(df_items, "items")


def processar_validacao() -> None:
    """
    Processa e salva os dados de validação.
    """
    if not os.path.exists(VALIDACAO_PATH):
        logger.error(f"❌ Arquivo de validação não encontrado: {VALIDACAO_PATH}")
        return

    logger.info("Processando dados de validação...")
    df_validacao = pd.read_csv(VALIDACAO_PATH, delimiter=",")

    # Tratar colunas de listas
    list_columns = ["history", "timestampHistory"]
    for col in list_columns:
        df_validacao[col] = (
            df_validacao[col]
            .astype(str)
            .str.replace(r"[\n\[\]\'\"]", "", regex=True)
            .str.strip()
            .str.split()
        )

    # Explodir colunas para criar uma linha por interação do usuário
    df_validacao = df_validacao.explode(list_columns)

    # Converter timestampHistory para datetime
    df_validacao["timestampHistory"] = pd.to_numeric(
        df_validacao["timestampHistory"], errors="coerce"
    )
    df_validacao["timestampHistory"] = pd.to_datetime(
        df_validacao["timestampHistory"], unit="ms"
    )

    # Renomear a coluna 'history' para 'page'
    df_validacao.rename(columns={"history": "page"}, inplace=True)

    # Limpar espaços em branco em colunas de string
    string_cols = df_validacao.select_dtypes(include="object").columns
    df_validacao[string_cols] = df_validacao[string_cols].apply(lambda x: x.str.strip())

    salvar_dataframe(df_validacao, "validacao")


def main() -> None:
    """
    Função principal que orquestra o processamento dos dados de usuários, itens e validação.
    """
    processar_usuarios()
    processar_itens()
    processar_validacao()


if __name__ == "__main__":
    main()
