
import argparse
import os
import sys
import logging

# Add project path to find the 'src' module
sys.path.append('/content/drive/MyDrive/Trabalho_final')

from src.data_load import load_csv_data
from src.model_trainer import train_and_evaluate

# --- Robust Logging Setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
file_handler = logging.FileHandler("cli_execution.log", mode='w')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# --- Mappings and Constants ---
MAPA_CRIPTO_ARQUIVO = {
    'AAVE': 'dataset01.csv', 'ADA': 'dataset02.csv', 'AERGO': 'dataset03.csv',
    'AGLD': 'dataset04.csv', 'AKITA': 'dataset05.csv', 'ALPACA': 'dataset06.csv',
    'ALPHA': 'dataset07.csv', 'APE': 'dataset08.csv', 'APX': 'dataset09.csv',
    'ATLAS': 'dataset10.csv'
}
CRIPTO_CHOICES = list(MAPA_CRIPTO_ARQUIVO.keys())

def main():
    parser = argparse.ArgumentParser(description="Script de Treinamento de Modelos.")
    parser.add_argument("--crypto", type=str.upper, default="AAVE", choices=CRIPTO_CHOICES, help="Ticker da criptomoeda.")
    parser.add_argument("--model", type=str, default="LinearRegression", choices=["LinearRegression", "RandomForest", "SVR"], help="Modelo a ser utilizado.")
    parser.add_argument("--kfolds", type=int, default=5, help="Número de folds.")
    parser.add_argument("--target_col", type=str, default="close", help="Nome da coluna alvo.")

    # ✅ THIS IS THE FIX ✅
    # Change .parse_args() to .parse_known_args()
    # It will parse the arguments it knows and ignore the internal Colab ones.
    args, unknown = parser.parse_known_args()

    logging.info(f"Parâmetros: Cripto={args.crypto}, Modelo={args.model}, K-Folds={args.kfolds}, Coluna Alvo={args.target_col}")
    if unknown:
        logging.warning(f"Argumentos desconhecidos foram ignorados: {unknown}")

    nome_arquivo = MAPA_CRIPTO_ARQUIVO.get(args.crypto)
    if nome_arquivo is None:
        logging.error(f"ERRO DE MAPEAMENTO: Criptomoeda '{args.crypto}' não encontrada.")
        return

    file_path = os.path.join('/content/drive/MyDrive/Trabalho_final/data', nome_arquivo)
    
    df = load_csv_data(file_path=file_path)
    if df is None:
        return

    mean_rmse = train_and_evaluate(df=df, model_name=args.model, n_splits=args.kfolds, target_col=args.target_col)
    if mean_rmse is None:
        logging.critical("Pipeline interrompido: falha no treinamento.")
        return

    logging.info(f"RESULTADO: Cripto={args.crypto}, RMSE Médio=${mean_rmse:.4f}")
    print(f"\n*** RESULTADO FINAL ***\nCriptomoeda: {args.crypto} | Modelo: {args.model}\nDesempenho (RMSE Médio): ${mean_rmse:.4f}\n*********************")

if __name__ == "__main__":
    main()
