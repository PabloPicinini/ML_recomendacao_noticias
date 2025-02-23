# Sistema de Recomendação de Notícias G1 com MLOps

## Objetivo do Projeto

Este projeto tem como objetivo desenvolver um pipeline completo de MLOps para recomendar notícias do G1 a diferentes perfis de usuários, utilizando seus padrões de leitura. Um dos principais desafios é mitigar o problema de Cold Start para usuários com poucas ou nenhuma interação, garantindo recomendações precisas mesmo em cenários de baixa informação. Além disso, o sistema integra todas as etapas do MLOps, desde a atualização contínua dos dados, treinamento e inferência dos modelos, até o monitoramento e avaliação do desempenho. O fluxo de trabalho envolve:

### 1. Coleta dos Dados
- Download automático dos dados brutos de um link,
- Armazenamento organizado dos dados.

### 2. Exploração e Tratamento dos Dados
- Análise exploratória (EDA).
- Tratamento de dados, remoção de outliers, normalização e refinamento.
- Separação dos dados em pastas raw (dados brutos) e refined (dados processados).

### 3. Treinamento de Modelos
    
Modelos desenvolvidos para diferentes perfis de usuários:
- Usuários Logados: Modelo híbrido utilizando Filtragem Colaborativa + Conteúdo.
- Usuários Semi-Logados: Aplicação de Clusterização para recomendações personalizadas.
- Usuários Anônimos: Uso de Heurísticas baseadas em popularidade e recência.


### 4. Avaliação dos Modelos
- Avaliação periódica via Airflow.
- Cálculo de métricas como Recall@K e NDCG@K para medir a qualidade das recomendações,
- Para fins didáticos, a avaliação atualmente considera apenas 50 usuários, o que pode ocasionar métricas com valores zerados.

### 5. Criação da API de Inferência
- API construída em FastAPI para servir as recomendações.
- Endpoint /recommendations retorna recomendações personalizadas.

### 6. Orquestração com Airflow
- Airflow automatiza o fluxo de dados e treinamento dos modelos.
- DAGs implementadas para coleta, tratamento, treinamento e avaliação.

### 7. Monitoramento com Grafana e Prometheus
- Grafana exibe dashboards para acompanhamento dos modelos.
- Prometheus coleta métricas da API.

### 8. Deploy com Docker & Docker Compose
- Dockerfiles para criação das imagens da API e do Airflow.
- Docker Compose gerencia a execução de toda a infraestrutura.

## Estratégias e Abordagens Adotadas

Para garantir um sistema eficiente e escalável, foram adotadas as seguintes estratégias:

### 1. Separação de Usuários
-  Usuários Logados: Possuem um histórico completo de interações. São recomendados via Filtragem Colaborativa e Conteúdo.
-  Usuários Semi-Logados: Possuem múltiplas interações, mas não estão logados. São recomendados através de Clusterização + Heurísticas.
-  Usuários Anônimos: Sem histórico suficiente. São recomendados com base em popularidade e recência.

### 2. Arquitetura do Sistema (Containers)

| Container              | Função                                                                 |
  |------------------------|------------------------------------------------------------------------|
  | **API (api_inferencia)**       | Fornece recomendações para os usuários via FastAPI.                  |
  | **Airflow Webserver**  | Interface do Airflow para monitoramento e disparo de DAGs.               |
  | **Airflow Scheduler**  | Agendamento e execução automática das DAGs.                            |
  | **PostgreSQL**         | Banco de dados para persistência do Airflow.                           |
  | **Prometheus**         | Coleta métricas de desempenho da API e do Airflow.                     |
  | **Grafana**            | Exibe dashboards para monitoramento das métricas.                      |


## Estrutura do Projeto

```
/recomendacao_noticias
│── app/
│   ├── __init__.py
│   ├── main.py                       # Arquivo principal da API
│   ├── routes/                       # Endpoints organizados
│   │   ├── recommendations.py        # Endpoint de recomendações
│   │   ├── metrics.py                # Endpoint de métricas Prometheus
│   │   ├── evaluation.py             # Endpoint para métricas de avaliação (CSV)
│   ├── services/                     # Lógica
│   │   ├── recommendation_service.py # Lógica de recomendação
│   │   ├── evaluation_service.py     # Lógica para leitura do CSV
│   ├── core/                         # Parte central da aplicação
│   │   ├── models_loader.py          # Carregamento de modelos de recomendação
│   ├── schemas/                      # Modelos Pydantic
│   │   ├── evaluation_model.py       # Modelo para JSON do CSV
│── docker/
│   ├── Dockerfile.api          # Dockerfile da API
│   ├── Dockerfile.airflow      # Dockerfile do Airflow
|   │── entrypoint.sh           # Arquivo para scripts de build da api
|   │── init_airflow.sh         # Arquivo para scripts de build do airflow
|   │── prometheus.yml          # Arquivo de configuração do prometheus
│── grafana/                    # Pasta com Indicadores do Grafana
│── pipelines/
│   ├── dags/                   # DAGs do Airflow
│   ├── evaluate/               # Scripts de avaliuação dos modelos
│   ├── train/                  # Scripts de treinamento
│   ├── utils/                  # Script para salvar as métricas de avaliação
│   ├── download_data.py        # Script para realizar download dos dados
│   ├── process_data.py         # Script para realizar processamento dos dados
│   ├── process_data.py         # Script para realizar processamento dos dados de tipo de usuários
│── script_shared/              # Pasta que é compartilhada entre a API e o Airflow
|   │── data/                   # Pasta que é compartilhada entre a API e o Airflow
|   │   ├── raw/                # Dados brutos (coletados)
|   │   ├── refined/            # Dados processados (limpos)
|   │── evaluation/             # Pasta que que contém as avaliações dos modelos salvas em CSV
│   ├── __init__.py
|   │── models/                 # Modelos treinados, salvos e scripts para carregar e recomendar notícias
│   ├── config.py               # Constantes para API e Airflow
│── test_model/                 # Notebooks para EDA, tratamento e testes de modelos
│── __init__.py
│── .dockerignore          
│── .gitignore     
│── docker-compose.yml          # Arquivo para subir os containers
│── LICENSE     
│── README.md
│── requirements_airflow.txt    # Dependências do projeto Airflow
│── requirements.txt            # Dependências do projeto API
```

Na pasta test_model foi criado um arquivo ipynb com o objetivo de realizar a análise exploratória dos dados, tratar outliers e testar modelos. Nesse notebook, foram salvas métricas de diferentes combinações obtidas via grid search para a melhoria dos modelos e a experimentação com diversos parâmetros. Esse trabalho foi fundamental, servindo como base para o tratamento de dados final e para a seleção dos modelos que foram implementados no projeto definitivo.

## Como Rodar o Projeto

### 1. Clonagem e Acesso ao Repositório

```
https://github.com/PabloPicinini/ML_recomendacao_noticias.git
cd ML_recomendacao_noticias
```

### 2. Subida dos Containers Docker
Execute os comando nessa ordem:

- Para construir a imagem:
  ```
  docker-compose build
  ```

- Para dar permissão de baixar arquivo zip, extrair e apagá-lo em seguida (Pelo Airflow, para alterar a pasta do host para a API)
  ```
  sudo chmod -R 775 script_shared
  sudo chmod -R 777 script_shared
  ```

- Para iniciar o banco do airflow
  ```
  docker-compose run airflow-webserver airflow db init
  ```

- Para subir os containers
  ```
  docker-compose up -d
  ```



Esses comandos inicia a construção e o deploy dos containers, de acordo com o arquivo ```docker-compose.yml``` e os respectivos Dockerfiles.

#### 2.1 Processo de Inicialização Detalhado

**Build e Inicialização da API:**

- ```Dockerfile.api```:

  - Durante o build da API, o Dockerfile copia os arquivos do repositório e o script ```entrypoint.sh``` para a imagem.
  - O ```entrypoint.sh``` é responsável por aguardar a existência dos arquivos necessários para a API iniciar, ou seja:

      - Dados Processados: ```script_shared/data/refined/*```
      - Modelos Treinados: ```script_shared/models/*```

      O script entra em loop verificando a existência desses arquivos (checando a cada 5 segundos) e só então inicia a API via Uvicorn, garantindo que a API só suba quando os dados tratados e modelos treinados estiverem presentes.

      - ```entrypoint.sh```:
      Garante a integridade do ambiente de execução da API, impedindo a inicialização se os arquivos necessários não forem encontrados.


**Build e Inicialização do Airflow:**

- ```Dockerfile.airflow```:

  O Dockerfile do Airflow prepara o ambiente instalando dependências (inclusive Java e outros pacotes necessários), copia os DAGs e scripts, e configura as permissões e variáveis de ambiente.

  - ```init_airflow.sh```:
  
    Ao subir o container do Airflow (airflow-webserver), o script ```init_airflow.sh``` é executado automaticamente. Esse script realiza os seguintes passos:

    - Instala a ferramenta gdown para download de arquivos.
    - Limpa o histórico antigo removendo dados e modelos previamente gerados, garantindo um ambiente limpo.
    - Aguarda 30 segundos para que o Airflow esteja totalmente operacional.
    - Cria um usuário admin, caso não exista.
    - Despausa as DAGs ```processamento_data``` e ```treinamento_modelos```.
    - Dispara a DAG ```processamento_data``` com um run_id customizado e monitora seu estado, aguardando até que ela finalize com sucesso.
    - Após o sucesso da DAG de processamento dos dados, dispara a DAG ```treinamento_modelos```, que é responsável por treinar os modelos.

    Essa sequência é crucial, pois as DAGs baixam os dados brutos, tratam os dados e treinam os modelos, gerando os arquivos que o entrypoint.sh da API espera para subir com sucesso.

**Integração com Prometheus e Grafana:**

- Prometheus:

  - O container do Prometheus é configurado para coletar métricas da API, utilizando o arquivo de configuração ```docker/prometheus.yml```.

- Grafana:

  - O Grafana está configurado para exibir dashboards já prontos, os quais se encontram em um volume compartilhado na raiz do projeto.
  - Esse volume compartilhado (mapeado para ```./grafana_data``` e ```./grafana/provisioning```) permite que as dashboards pré-configuradas sejam utilizadas imediatamente, facilitando a visualização das métricas sem necessidade de configuração adicional.

- Compartilhamento de Volume:
  - Tanto o Prometheus quanto o Grafana utilizam volumes compartilhados para persistência dos dados e dashboards. Essa configuração garante que as métricas e dashboards sejam mantidas atualizadas e acessíveis mesmo após reinicializações dos containers.


**Orquestração com docker-compose:**

O arquivo ```docker-compose.yml``` orquestra a execução dos containers:
- Define os serviços de API, Airflow (webserver e scheduler) e PostgreSQL.
- Configura volumes e redes para compartilhamento de dados e comunicação entre os containers.
- Garante que o Airflow dependa do PostgreSQL para a persistência dos dados.
- Estabelece que o container do Airflow execute o init_airflow.sh na inicialização, integrando todo o fluxo de processamento e treinamento de dados.

#### 2.2 Acesso aos Serviços

- API: http://localhost:8001/docs
- Airflow: http://localhost:8080
  - Usuário padrão criado automaticamente: admin / admin.
- Grafana: [http://localhost:3000](http://localhost:3000/d/fedvlqolwrw8we/monitoramento?orgId=1&from=now-10d&to=now&timezone=browser)
- Prometheus: http://localhost:9090/

  ***Nota: As DAGs possuem atualizações configuradas para None, permitindo modificação conforme necessário, sendo essa configuração mantida para fins didáticos.***

## Próximos passos

1. Infraestrutura:

    Substituir o download manual dos dados por uma solução em nuvem, utilizando AWS S3 para armazenamento, AWS Lambda para processamento e AWS Glue para tratamento dos dados.

2. Modelos:

    Modelo Logado:
    - Paralelizar a avaliação utilizando joblib.Parallel.
    - Amostrar um subconjunto do dataset para reduzir o tempo de validação.
    - Adicionar métricas adicionais como cobertura e novidade das recomendações.
    - Ajustar melhor os pesos das features, recência e engajamento.

    Modelo Semi-Logado:
    - Criar novas features para melhorar a clusterização, como desvio-padrão, mediana e valores máximos de interação.
    - Testar variações no número de componentes do PCA para melhorar a segmentação de usuários.

3. Monitoramento:
  Implementar um ambiente de produção escalável utilizando serviços na nuvem e expandir o monitoramento com métricas mais detalhadas sobre tempo de inferência e taxa de erro.