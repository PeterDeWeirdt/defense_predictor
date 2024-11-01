import pandas as pd
from pathlib import Path
import os

if __name__ == '__main__':
    os.system('ls -l tests/data')
    true_y_path = Path(__file__).parent / 'data' / 'GCF_003333385_beaker_predictions.csv'
    true_y_df = pd.read_csv(true_y_path, index_col=0)
    print('Head of true Y df:', true_y_df.head())
    print('Columns of true Y df:', true_y_df.columns)
    true_y_df = true_y_df[['beaker_prediction']]
    print('Head of true Y df:', true_y_df.head())
