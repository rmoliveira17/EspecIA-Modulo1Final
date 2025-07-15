import pandas as pd
import logging

def load_csv_data(file_path: str):
    """Carrega dados de um arquivo CSV."""
    logging.info(f"Carregando dados de '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo não foi encontrado em: {file_path}")
        return None