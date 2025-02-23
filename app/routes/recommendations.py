from fastapi import APIRouter, HTTPException, Query
from typing import List
from services.recommendation_service import get_recommendations_for_user
from core.models_loader import load_all_models


router = APIRouter()

# Carrega os modelos na inicialização
models = load_all_models()

@router.get("/recommendations", response_model=List[str])
def get_recommendations(user_id: str, num_recs: int = Query(5, gt=0)) -> List[str]:
    """
    Retorna uma lista de recomendações para o usuário informado.
    """
    if not all(
        models.get(k) is not None for k in ["logged", "semianon", "anon"]
    ) or models["df_users_logged"] is None or models["df_users_logged"].empty:
        raise HTTPException(status_code=500, detail="Modelos não carregados corretamente ou DataFrame vazio.")


    return get_recommendations_for_user(
        user_id=user_id,
        num_recs=num_recs,
        logged_model=models["logged"],
        semianon_model=models["semianon"],
        anon_model=models["anon"],
        df_users_logged=models["df_users_logged"],
    )
