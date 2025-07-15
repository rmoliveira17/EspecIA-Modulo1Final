
import pytest
import pandas as pd
import sys
# Adiciona o caminho do projeto para que o 'from src...' funcione
sys.path.insert(0, '/content/drive/MyDrive/Trabalho Final IA')
from src.data_load import load_csv_data

def test_load_csv_data_success(tmp_path):
    file_path = tmp_path / "test_data.csv"
    df_original = pd.DataFrame({'col1': [10], 'col2': ['X']})
    df_original.to_csv(file_path, index=False)
    df_loaded = load_csv_data(file_path)
    assert df_loaded is not None
    pd.testing.assert_frame_equal(df_original, df_loaded)

def test_load_csv_data_file_not_found():
    assert load_csv_data("caminho/invalido/arquivo.csv") is None
