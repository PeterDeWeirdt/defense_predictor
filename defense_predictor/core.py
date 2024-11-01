from importlib import resources
from joblib import load
import pandas as pd
import numpy as np

def load_model():
    with resources.files('defense_predictor').joinpath('beaker_v3.pkl').open('rb') as f:
        return load(f)

def predict(data):
    model = load_model()
    probs = model.predict_proba(data)[:, 1]
    output_df = pd.DataFrame(index=data.index)
    output_df['probability'] = probs
    output_df['log-odds'] = np.log(probs / (1 - probs))
    return output_df