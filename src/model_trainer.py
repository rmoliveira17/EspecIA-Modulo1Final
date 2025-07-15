
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import logging

def train_and_evaluate(df: pd.DataFrame, model_name: str, n_splits: int, target_col: str):
    logging.info(f"Iniciando treinamento. Coluna alvo: '{target_col}'")
    
    try:
        # ✅ LÓGICA DE PREPARAÇÃO CORRIGIDA E ROBUSTA ✅
        
        # 1. GARANTE A ORDENAÇÃO POR DATA
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values(by='Date', inplace=True)
            df.reset_index(drop=True, inplace=True) # Reseta o índice para garantir a ordem

        # 2. Isola a coluna de interesse para evitar efeitos colaterais.
        dados = df[[target_col]].copy()

        # 3. Cria a feature (X) e o alvo (y) de forma explícita.
        dados['Target'] = dados[target_col].shift(-1)
        
        # 4. Remove apenas a última linha que terá um NaN no alvo.
        dados.dropna(inplace=True)

        if dados.empty:
            logging.error("DataFrame ficou vazio após preparação.")
            return None

        X = dados[[target_col]]
        y = dados['Target']

    except KeyError:
        logging.error(f"ERRO: A coluna alvo '{target_col}' não foi encontrada.")
        return None

    # O restante do código de treinamento...
    if model_name == 'LinearRegression': model = LinearRegression()
    elif model_name == 'RandomForest': model = RandomForestRegressor(n_estimators=10, random_state=42)
    elif model_name == 'SVR': model = SVR()
    else: raise ValueError(f"Modelo '{model_name}' não é suportado.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    
    rmse_scores = np.sqrt(-cv_scores)
    return np.mean(rmse_scores)
