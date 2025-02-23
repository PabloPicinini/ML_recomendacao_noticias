import logging
import pandas as pd
from typing import Dict, Any

# Importar funções de carregamento dos modelos
from script_shared.models.model_logged import load_model_logged
from script_shared.models.model_semianon import load_model_semianon
from script_shared.models.model_anon import load_model_anon_heuristico
from script_shared import config

MODEL_DIR_LOGGED = config.MODEL_DIR_LOGGED
USERS_LOGGED = config.USERS_LOGGED
MODEL_DIR_SEMIANON = config.MODEL_DIR_SEMIANON
MODEL_DIR_ANON_HEURISTICO = config.MODEL_DIR_ANON_HEURISTICO

logger = logging.getLogger(__name__)


def load_all_models() -> Dict[str, Any]:
    """
    Carrega todos os modelos e artefatos (logged, semianon, anônimo) e retorna um dicionário com eles.

    Returns:
        Um dicionário com as chaves:
         - "logged": modelo e artefatos do usuário logado.
         - "semianon": modelo e artefatos do usuário semi-logado.
         - "anon": modelo e artefatos do usuário anônimo (heurístico).
         - "df_users_logged": DataFrame com as interações dos usuários logados (necessário para a inferência do modelo logged).
    """
    models = {}

    # Carregar modelo logged
    try:
        df_users_logged = pd.read_parquet(USERS_LOGGED)
        models["logged"] = load_model_logged(MODEL_DIR_LOGGED)
        models["df_users_logged"] = df_users_logged
        logger.info("Modelo logged carregado com sucesso.")
    except Exception as e:
        logger.error("Erro ao carregar modelo logged.", exc_info=e)
        models["logged"] = None
        models["df_users_logged"] = None

    # Carregar modelo semianon
    try:
        models["semianon"] = load_model_semianon(MODEL_DIR_SEMIANON)
        logger.info("Modelo semianon carregado com sucesso.")
    except Exception as e:
        logger.error("Erro ao carregar modelo semianon.", exc_info=e)
        models["semianon"] = None

    # Carregar modelo anônimo (heurístico)
    try:
        models["anon"] = load_model_anon_heuristico(MODEL_DIR_ANON_HEURISTICO)
        logger.info("Modelo anônimo (heurístico) carregado com sucesso.")
    except Exception as e:
        logger.error("Erro ao carregar modelo anônimo.", exc_info=e)
        models["anon"] = None

    return models
