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


def _decompose_protein_context_id(df):
    parts = df['protein_context_id'].str.split('|', expand=True)
    df = df.copy()
    df['_ref_genomic'] = parts[1]
    df['_ref_start'] = parts[2].astype(int)
    df['_ref_strand'] = parts[3]
    return df


def test_defense_predictor_pgap_gff(reference_predictions):
    data_top_dir = Path(__file__).parent / 'data'
    pgap_gff = data_top_dir / 'annot_with_genomic_fasta.gff'
    output_df, _ = defense_predictor(pgap_gff=pgap_gff)
    # PGAP uses locus_tag (pgaptmp_XXXXXX) as the identifier while the NCBI reference uses
    # NP_/WP_ accessions, so join on (genomic_accession, start, strand) instead of
    # protein_context_id. For the vast majority of genes PGAP and RefSeq agree on coordinates.
    ref = _decompose_protein_context_id(reference_predictions[['protein_context_id',
                                                               'mean_log_odds']])
    merged_df = output_df.merge(
        ref.rename(columns={'mean_log_odds': 'mean_log_odds_reference'}),
        left_on=['genomic_accession', 'start', 'strand'],
        right_on=['_ref_genomic', '_ref_start', '_ref_strand'],
    )
    # Expect nearly all reference genes to match PGAP annotations on coordinates.
    recovery = len(merged_df) / len(reference_predictions)
    assert recovery > 0.95, f'Only recovered {recovery:.2%} of reference genes'
    merged_df['delta'] = merged_df['mean_log_odds'] - merged_df['mean_log_odds_reference']
    assert merged_df['delta'].abs().mean() < 1e-2
