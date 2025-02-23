#!/bin/sh
set -e

# Lista de arquivos que a API precisa
REQUIRED_FILES="script_shared/data/refined/users_logged.parquet \
                script_shared/data/refined/items.parquet \
                script_shared/data/refined/users_logged.parquet \
                script_shared/data/refined/users_semianon.parquet \
                script_shared/models/logged/model_logged_als.npz \
                script_shared/models/logged/objetos_logged_auxiliares.pkl \
                script_shared/models/logged/tfidf_logged_matrix.npz \
                script_shared/models/semianon/cluster_top_items.pkl \
                script_shared/models/semianon/df_features_semianon.csv \
                script_shared/models/semianon/modelo_semianon_kmeans.pkl \
                script_shared/models/semianon/pca_semianon.pkl \
                script_shared/models/semianon/scaler_semianon.pkl \
                script_shared/models/anon_heuristico/ranking_anon_heuristico.pkl"

for file in $REQUIRED_FILES; do
  echo "Aguardando o arquivo: $file ..."
  while [ ! -f "$file" ]; do
    sleep 5
  done
  echo "Arquivo encontrado: $file"
done

echo "Todos os arquivos necessários foram encontrados. Iniciando a API."
exec uvicorn main:app --host 0.0.0.0 --port 8000
