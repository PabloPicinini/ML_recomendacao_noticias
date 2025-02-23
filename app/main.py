import uvicorn
import logging
import time
from fastapi import FastAPI, Request
from routes.recommendations import router as recommendations_router
from routes.metrics import router as metrics_router
from routes.evaluation import router as evaluation_router
from prometheus_client import Counter, Histogram

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Sistema de Recomendação",
    description=(
        "API para previsão de recomendações de notícias do G1 com base em modelos para diversos tipos de usuários. "
        "A pipeline de inferência determina se o usuário é logado, semianônimo ou anônimo, retornando o modelo adequado."
    ),
    version="1.0",
)

# Definição de Métricas Prometheus
REQUEST_COUNT = Counter("api_request_count", "Contagem de requisições por endpoint e status", ["endpoint", "http_status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Tempo de resposta por endpoint", ["endpoint"])

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    REQUEST_COUNT.labels(endpoint=request.url.path, http_status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(process_time)
    
    return response

app.include_router(recommendations_router)
app.include_router(metrics_router)
app.include_router(evaluation_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
