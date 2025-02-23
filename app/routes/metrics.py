from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter()

@router.get("/metrics")
def metrics() -> Response:
    """Exposição das métricas Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
