from pydantic import BaseModel

class EvaluationMetric(BaseModel):
    type_model: str
    mean_recall: float
    mean_ndcg: float
    timestamp: str
