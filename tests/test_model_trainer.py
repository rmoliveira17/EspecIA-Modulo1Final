
import pytest
import pandas as pd
import sys
# Adiciona o caminho do projeto para que o 'from src...' funcione
sys.path.insert(0, '/content/drive/MyDrive/Trabalho Final IA')
from src.model_trainer import train_and_evaluate

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({'close': range(100, 200)})

def test_train_and_evaluate_successful_run(sample_dataframe):
    rmse = train_and_evaluate(df=sample_dataframe, model_name='LinearRegression', n_splits=3, target_col='close')
    assert isinstance(rmse, float) and rmse > 0

def test_train_and_evaluate_invalid_column(sample_dataframe):
    assert train_and_evaluate(df=sample_dataframe, model_name='RandomForest', n_splits=3, target_col='ColunaInvalida') is None

def test_train_and_evaluate_unsupported_model(sample_dataframe):
    with pytest.raises(ValueError, match="Modelo 'MeuModeloRuim' não é suportado."):
        train_and_evaluate(df=sample_dataframe, model_name='MeuModeloRuim', n_splits=3, target_col='close')
