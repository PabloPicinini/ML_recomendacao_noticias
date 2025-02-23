import csv
from schemas.evaluation_model import EvaluationMetric

CSV_PATH = "/app/script_shared/evaluation/evaluation_metrics.csv"

def get_evaluation_metrics():
    """
    Lê o arquivo CSV e retorna os dados como uma lista de EvaluationMetric.
    """
    results = []
    try:
        with open(CSV_PATH, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                results.append(EvaluationMetric(
                    type_model=row["type_model"],
                    mean_recall=float(row["mean_recall"]),
                    mean_ndcg=float(row["mean_ndcg"]),
                    timestamp=row["timestamp"]
                ))
    except Exception as e:
        raise RuntimeError(f"Erro ao ler o CSV: {str(e)}")
    
    return results
