import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

def create_ratio_feature(df: pd.DataFrame, col1: str, col2: str, new_col_name: str) -> Optional[pd.DataFrame]:
    """
    Cria uma nova feature baseada na razão de duas colunas existentes.

    Args:
        df (pd.DataFrame): O DataFrame de entrada.
        col1 (str): O nome da coluna do numerador.
        col2 (str): O nome da coluna do denominador.
        new_col_name (str): O nome da nova coluna a ser criada.

    Returns:
        Optional[pd.DataFrame]: O DataFrame com a nova coluna adicionada
                                ou None se ocorrer um erro.
    """
    logger.info(f"Criando a feature '{new_col_name}' a partir de '{col1}' / '{col2}'.")
    try:
        # Cria uma cópia para evitar o SettingWithCopyWarning
        df_copy = df.copy()
        
        if col1 not in df_copy.columns or col2 not in df_copy.columns:
            raise KeyError(f"Uma ou ambas as colunas '{col1}', '{col2}' não foram encontradas no DataFrame.")

        # Evita divisão por zero
        if (df_copy[col2] == 0).any():
            logger.warning(f"A coluna '{col2}' contém zeros. A razão será infinita (inf) nesses casos.")
            # Substituímos 0 por um valor muito pequeno para evitar erro, ou poderíamos tratar de outra forma
            df_copy[new_col_name] = df_copy[col1] / df_copy[col2].replace(0, 1e-9)
        else:
            df_copy[new_col_name] = df_copy[col1] / df_copy[col2]
        
        logger.info(f"Feature '{new_col_name}' criada com sucesso.")
        return df_copy

    except KeyError as e:
        logger.error(f"Erro de chave ao criar feature: {e}")
        return None
    except Exception as e:
        logger.exception(f"Erro inesperado ao criar feature '{new_col_name}': {e}")
        return None