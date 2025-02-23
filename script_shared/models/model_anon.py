import os
import pickle
import logging
from typing import Dict, Any, List
from script_shared import config

MODEL_DIR_ANON_HEURISTICO = config.MODEL_DIR_ANON_HEURISTICO

# Configuração do logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model_anon_heuristico(
    model_dir: str = MODEL_DIR_ANON_HEURISTICO,
) -> Dict[str, Any]:
    """
    Carrega o ranking salvo para o modelo anônimo heurístico.

    Parâmetros:
      - model_dir: diretório onde o ranking está salvo.

    Retorna:
      - Dicionário com o ranking e o método utilizado.
    """
    ranking_path = os.path.join(model_dir, "ranking_anon_heuristico.pkl")
    with open(ranking_path, "rb") as f:
        ranking_anon = pickle.load(f)

    logger.info("Modelo anônimo heurístico carregado com sucesso.")
    return {"ranking_anon": ranking_anon, "method": "heurístico"}


def recomendar_anon_heuristico(
    model_objs: Dict[str, Any], top_k: int = 10
) -> List[Any]:
    """
    Retorna as top_k recomendações para usuários anônimos com base no ranking heurístico.

    Parâmetros:
      - model_objs: dicionário contendo o ranking do modelo.
      - top_k: número de itens a serem retornados.

    Retorna:
      - Lista com as top_k recomendações.
    """
    ranking = model_objs.get("ranking_anon", [])
    return ranking[:top_k]
