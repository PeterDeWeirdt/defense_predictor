import pytest
from pathlib import Path
import pandas as pd
from defense_predictor import download_weights, defense_predictor
import numpy as np


@pytest.fixture(scope='session', autouse=True)
def download_weights_fixture():
    download_weights()


@pytest.fixture
def reference_predictions():
    ref_pred_path = Path(__file__).parent / 'data' / 'GCF_000005845_predictions.pq'
    ref_pred_df = pd.read_parquet(ref_pred_path)
    return ref_pred_df


def test_defense_predictor(reference_predictions):
    data_top_dir = Path(__file__).parent / 'data'
    ncbi_feature_table = data_top_dir / 'GCF_000005845.2_ASM584v2_feature_table.txt'
    ncbi_cds_from_genomic = data_top_dir / 'GCF_000005845.2_ASM584v2_cds_from_genomic.fna'
    ncbi_protein_fasta = data_top_dir / 'GCF_000005845.2_ASM584v2_protein.faa'
    output_df, _ = defense_predictor(ft_file=ncbi_feature_table,  
                                  fna_file=ncbi_cds_from_genomic,
                                  faa_file=ncbi_protein_fasta)
    merged_df = output_df.merge(reference_predictions[['protein_context_id', 'mean_log_odds']], 
                                on='protein_context_id', suffixes=('_curr', '_reference'))
    merged_df['delta'] = merged_df['mean_log_odds_curr'] - merged_df['mean_log_odds_reference']
    assert merged_df['delta'].abs().mean() < 1e-2
    