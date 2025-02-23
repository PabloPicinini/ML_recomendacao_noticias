import os

# Identifica o ambiente
ENVIRONMENT = os.getenv("APP_ENV", "api")  # Padrão: "api"

# Define os caminhos conforme o ambiente
if ENVIRONMENT == "api":
    BASE_PATH = "/app/script_shared"
elif ENVIRONMENT == "airflow":
    BASE_PATH = "/opt/airflow/shared/script_shared"
else:
    raise ValueError(f"Ambiente '{ENVIRONMENT}' desconhecido")

# Configuração Treino Anônimo
MODEL_DIR_ANON_HEURISTICO = os.path.join(BASE_PATH, "models", "anon_heuristico")

DEFAULT_W_ISSUED = 0.8
DEFAULT_W_MODIFIED = 0.2

# Configuração Treino Logged
ALS_DEFAULT_PARAMS = {
    "factors": 40,
    "regularization": 0.04,
    "iterations": 15,
    "alpha": 12,
}
SPARSE_MATRIX_PATH = os.path.join(BASE_PATH, "data", "refined", "user_item_sparse_mat_logged.npz")
MODEL_DIR_LOGGED = os.path.join(BASE_PATH, "models", "logged")

# Configuração Treino Semi-Anônimo
MODEL_DIR_SEMIANON = os.path.join(BASE_PATH, "models", "semianon")
FEATURE_COLUMNS_SEMIANON = [
    "sum_time",
    "sum_clicks",
    "mean_scroll",
    "unique_pages",
    "time_per_page",
    "clicks_per_page",
    "log_sum_time",
    "log_sum_clicks",
    "log_time_per_page",
    "log_clicks_per_page",
]

# Arquivos parquet
USERS_LOGGED = os.path.join(BASE_PATH, "data", "refined", "users_logged.parquet")

# Caminho do modelo Logged específico
MODEL_LOGGED_PATH = os.path.join(MODEL_DIR_LOGGED, "model_logged_als.npz.pkl")
