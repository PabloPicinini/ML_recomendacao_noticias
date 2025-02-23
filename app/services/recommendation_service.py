import logging
from typing import List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


def get_recommendations_for_user(
    user_id: str,
    num_recs: int,
    logged_model: Dict[str, Any],
    semianon_model: Dict[str, Any],
    anon_model: Dict[str, Any],
    df_users_logged: pd.DataFrame,
) -> List[str]:
    """
    Verifica qual modelo utilizar para o usuário informado e retorna a lista de recomendações.

    Se o usuário estiver presente no modelo logged, utiliza o modelo logged.
    Se estiver presente no modelo semianon, utiliza o modelo semianon.
    Caso contrário, utiliza o modelo anônimo (heurístico).

    Args:
        user_id: ID do usuário.
        num_recs: Número de recomendações a retornar.
        logged_model: Artefatos do modelo logged.
        semianon_model: Artefatos do modelo semianon.
        anon_model: Artefatos do modelo anônimo.
        df_users_logged: DataFrame de usuários logados (necessário para a inferência do modelo logged).

    Returns:
        Uma lista de recomendações (IDs dos itens).
    """
    # Se o usuário estiver no modelo logged
    if user_id in logged_model["aux_dict"].get("user_to_idx", {}):
        from script_shared.models.model_logged import recomendar_logged

        recs = recomendar_logged(user_id, logged_model, df_users_logged, top_k=num_recs)
        logger.info(f"Recomendações geradas para usuário logged: {user_id}")
        return recs

    # Se o usuário estiver no modelo semianon
    elif user_id in semianon_model["df_features"]["userId"].unique():
        from script_shared.models.model_semianon import recomendar_semianon

        recs = recomendar_semianon(user_id, semianon_model, top_k=num_recs)
        logger.info(f"Recomendações geradas para usuário semianon: {user_id}")
        return recs

    # Caso contrário, assume usuário anônimo completo
    else:
        from script_shared.models.model_anon import recomendar_anon_heuristico

        recs = recomendar_anon_heuristico(anon_model, top_k=num_recs)
        logger.info(f"Recomendações geradas para usuário anônimo (fallback): {user_id}")
        return recs
