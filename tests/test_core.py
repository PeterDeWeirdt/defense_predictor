import pytest
from pathlib import Path
import pandas as pd
from defense_predictor import load_model, predict, get_ncbi_seq_info, get_prokka_seq_info, run_defense_predictor
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
    true_y_df = true_y_df[['beaker_prediction']]
    return true_y_df


def test_get_seq_info():
    data_top_dir = Path(__file__).parent / 'data'
    ncbi_seq_info = get_ncbi_seq_info(data_top_dir / 'GCF_003333385.1_ASM333338v1_feature_table.txt', 
                                      data_top_dir / 'GCF_003333385.1_ASM333338v1_cds_from_genomic.fna',
                                      data_top_dir / 'GCF_003333385.1_ASM333338v1_protein.faa')
    prokka_seq_info = get_prokka_seq_info(data_top_dir / 'GCF_003333385.1_11042024.gff', 
                                          data_top_dir / 'GCF_003333385.1_11042024.ffn', 
                                          data_top_dir / 'GCF_003333385.1_11042024.faa')
    protein_overlap = ncbi_seq_info['protein_seq'].isin(prokka_seq_info['protein_seq']).sum()
    dna_overlap = ncbi_seq_info['dna_seq'].isin(prokka_seq_info['dna_seq']).sum()
    assert protein_overlap/len(ncbi_seq_info) > 0.8
    assert protein_overlap/len(prokka_seq_info) > 0.8
    assert dna_overlap/len(ncbi_seq_info) > 0.8
    assert dna_overlap/len(prokka_seq_info) > 0.8


def test_load_model():
    model = load_model()
    assert model is not None


def test_predict(X, true_y_df):
    output_df = predict(X)
    merged_df = output_df.merge(true_y_df, left_index=True, right_index=True)
    assert np.allclose(merged_df['defense_probability'], merged_df['beaker_prediction'], atol=1e-2)


def test_run_defense_predictor(true_y_df):
    data_top_dir = Path(__file__).parent / 'data'
    ncbi_feature_table = data_top_dir / 'GCF_003333385.1_ASM333338v1_feature_table.txt'
    ncbi_cds_from_genomic = data_top_dir / 'GCF_003333385.1_ASM333338v1_cds_from_genomic.fna'
    ncbi_protein_fasta=None = data_top_dir / 'GCF_003333385.1_ASM333338v1_protein.faa'
    output_df = run_defense_predictor(ncbi_feature_table=ncbi_feature_table,  
                                      ncbi_cds_from_genomic=ncbi_cds_from_genomic,
                                      ncbi_protein_fasta=ncbi_protein_fasta)
    merged_df = output_df.merge(true_y_df, left_on='protein_context_id', right_index=True)
    merged_df['delta'] = merged_df['defense_probability'] - merged_df['beaker_prediction']
    assert merged_df['delta'].abs().mean() < 1e-3
    