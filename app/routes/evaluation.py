from fastapi import APIRouter, HTTPException
from typing import List
from services.evaluation_service import get_evaluation_metrics
from schemas.evaluation_model import EvaluationMetric

router = APIRouter()

@router.get("/evaluation/metrics", response_model=List[EvaluationMetric])
def evaluation_metrics():
    """
    Endpoint que lê o CSV gerado pelo Airflow e retorna as métricas de avaliação.
    """
    try:
        return get_evaluation_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
