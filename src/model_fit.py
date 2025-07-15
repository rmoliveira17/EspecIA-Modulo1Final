import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import logging

# Configuração básica do logging para ver as mensagens
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_fit_model(df: pd.DataFrame, model_name: str, target_col: str):
    """
    Prepara os dados, treina um modelo com TODOS os dados disponíveis e retorna o modelo treinado.

    Args:
        df (pd.DataFrame): O DataFrame de entrada. Deve conter a coluna alvo e, opcionalmente, uma coluna 'Date'.
        model_name (str): O nome do modelo a ser treinado ('LinearRegression', 'RandomForest', 'SVR').
        target_col (str): O nome da coluna que será usada como alvo e feature.

    Returns:
        tuple: Uma tupla contendo (modelo_treinado, X, y) ou (None, None, None) em caso de erro.
               - modelo_treinado: O objeto do modelo sklearn treinado.
               - X: O DataFrame de features usado para o treino.
               - y: A Series do alvo usada para o treino.
    """
    logging.info(f"Iniciando preparação de dados para o treino final. Coluna alvo: '{target_col}'")

    try:
        # --- LÓGICA DE PREPARAÇÃO DE DADOS (idêntica à sua) ---
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values(by='Date', inplace=True)
            df.reset_index(drop=True, inplace=True)

        dados = df[[target_col]].copy()
        dados['Target'] = dados[target_col].shift(-1)
        dados.dropna(inplace=True)

        if dados.empty:
            logging.error("DataFrame ficou vazio após preparação. Não é possível treinar.")
            return None, None, None

        X = dados[[target_col]]
        y = dados['Target']

    except KeyError:
        logging.error(f"ERRO: A coluna alvo '{target_col}' não foi encontrada no DataFrame.")
        return None, None, None
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado durante a preparação dos dados: {e}")
        return None, None, None

    # --- SELEÇÃO E TREINAMENTO FINAL DO MODELO ---
    logging.info(f"Treinando o modelo '{model_name}' em {len(X)} amostras de dados.")
    
    if model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'RandomForest':
        # Você pode usar aqui os melhores parâmetros que encontrou na fase de avaliação
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == 'SVR':
        model = SVR()
    else:
        logging.error(f"Modelo '{model_name}' não é suportado.")
        raise ValueError(f"Modelo '{model_name}' não é suportado.")

    # A "mágica" acontece aqui: treinamos o modelo com todos os dados X e y
    model.fit(X, y)
    
    logging.info("Treinamento finalizado com sucesso!")
    
    # Retornamos o modelo treinado e os dados usados, para referência
    return model, X, y