import pytest
from pathlib import Path
import pandas as pd
from defense_predictor import load_model, predict
import numpy as np

@pytest.fixture
def X():
    x_path = Path(__file__).parent / 'data' / 'GCF_003333385_X.csv'
    x = pd.read_csv(x_path, index_col=0)
    return x

@pytest.fixture
def true_y_df():
    true_y_path = Path(__file__).parent / 'data' / 'GCF_003333385_beaker_predictions.csv'
    true_y_df = pd.read_csv(true_y_path, index_col=0)
    print('Head of true Y df:', true_y_df.head())
    print('Columns of true Y df:', true_y_df.columns)
    true_y_df = true_y_df[['beaker_prediction']]
    return true_y_df

def test_load_model():
    model = load_model()
    assert model is not None

def test_predict(X, true_y_df):
    output_df = predict(X)
    merged_df = output_df.merge(true_y_df, left_index=True, right_index=True)
    assert np.allclose(merged_df['probability'], merged_df['beaker_prediction'], atol=1e-2)