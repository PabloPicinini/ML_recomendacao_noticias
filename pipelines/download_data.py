import logging
import os
import zipfile
from pathlib import Path
from typing import Union

import gdown

# Configuração do logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def download_and_extract_zip(file_id: str, extract_to: Union[str, Path] = '.') -> None:
    """
    Baixa um arquivo ZIP do Google Drive usando o file_id, extrai os arquivos e remove
    arquivos que não sejam CSV da pasta de extração.

    Args:
        file_id: ID do arquivo no Google Drive.
        extract_to: Diretório onde o arquivo ZIP será baixado e extraído.
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    zip_path = extract_to / "arquivo.zip"

    logger.info(f"Iniciando o download do arquivo {file_id} para {zip_path}")
    gdown.download(url, str(zip_path), quiet=False)

    if zip_path.exists():
        if zipfile.is_zipfile(zip_path):
            logger.info("Arquivo ZIP válido. Iniciando a extração...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info("Extração concluída com sucesso!")
        else:
            logger.error("O arquivo baixado não é um arquivo ZIP válido.")
        try:
            zip_path.unlink()
            logger.info("Arquivo ZIP removido após extração.")
        except Exception as e:
            logger.error(f"Erro ao remover o arquivo ZIP: {e}")

        clean_folder(extract_to)
    else:
        logger.error("Falha no download do arquivo.")

def clean_folder(directory: Union[str, Path]) -> None:
    """
    Remove todos os arquivos que não são CSV e apaga subpastas vazias dentro do diretório especificado.

    Args:
        directory: Diretório a ser limpo.
    """
    directory = Path(directory)
    for root, dirs, files in os.walk(directory, topdown=False):
        root_path = Path(root)
        # Remove arquivos que não são CSV
        for file in files:
            if not file.lower().endswith('.csv'):
                file_path = root_path / file
                try:
                    file_path.unlink()
                    logger.info(f"Removido: {file_path}")
                except Exception as e:
                    logger.error(f"Erro ao remover o arquivo {file_path}: {e}")
        # Remove subpastas vazias
        for dir in dirs:
            dir_path = root_path / dir
            try:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    logger.info(f"Removida pasta vazia: {dir_path}")
            except Exception as e:
                logger.error(f"Erro ao remover a pasta {dir_path}: {e}")

def main() -> None:
    # ID do arquivo no Google Drive
    file_id = '10zmuxXi05ayfiREA9hi-xSrcYJy0iW-r&export'
    # Diretório onde o arquivo será baixado e extraído
    extract_to = Path("/opt/airflow/shared/script_shared/data/raw")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    download_and_extract_zip(file_id, extract_to)

if __name__ == "__main__":
    main()
