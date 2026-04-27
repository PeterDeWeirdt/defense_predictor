from pathlib import Path

import pandas as pd
import pytest

from defense_predictor import gff


GFF_FIXTURE = Path(__file__).parent / 'data' / 'annot_with_genomic_fasta.gff'


def test_reverse_complement():
    assert gff.reverse_complement('ATGC') == 'GCAT'
    assert gff.reverse_complement('AAAA') == 'TTTT'
    assert gff.reverse_complement('ACGTN') == 'NACGT'


def test_translate_cds_basic():
    # ATG-AAA-TAA → M, K, stop
    assert gff.translate_cds('ATGAAATAA') == 'MK'


def test_translate_cds_alternative_start():
    # GTG is a valid bacterial start codon; transl_table=11 rule forces first residue to M.
    assert gff.translate_cds('GTGAAATAA') == 'MK'
    assert gff.translate_cds('TTGAAATAA') == 'MK'


def test_translate_cds_ambiguous_codon():
    # N-containing codons become X.
    assert gff.translate_cds('ATGNNNAAATAA') == 'MXK'


@pytest.fixture(scope='module')
def parsed_gff():
    return gff.parse_gff(GFF_FIXTURE)


def test_parse_gff_fasta_loaded(parsed_gff):
    _, contig_seqs = parsed_gff
    assert 'NC_000913.3' in contig_seqs
    assert len(contig_seqs['NC_000913.3']) == 4641652


def test_parse_gff_counts(parsed_gff):
    cds_records, _ = parsed_gff
    # Every CDS record must have a unique ID (frameshift pairs merged into one).
    ids = [r['id'] for r in cds_records]
    assert len(ids) == len(set(ids))
    # Reasonable count for E. coli K-12 (roughly 4,200-4,300 protein-coding CDS).
    assert 4000 < len(cds_records) < 4500
    # No pseudogenes should slip through.
    for r in cds_records:
        assert r['id'].startswith('cds-')


def test_frameshift_merged(parsed_gff):
    cds_records, _ = parsed_gff
    matches = [r for r in cds_records if r['locus_tag'] == 'pgaptmp_000020']
    assert len(matches) == 1
    rec = matches[0]
    assert rec['strand'] == '-'
    assert rec['start'] == 19811
    assert rec['end'] == 20508
    assert len(rec['segments']) == 2
    assert rec['exception'] == 'ribosomal slippage'


def test_build_feature_df_schema(parsed_gff):
    cds_records, _ = parsed_gff
    feature_df = gff.build_feature_df(cds_records)
    expected_cols = {'# feature', 'product_accession', 'genomic_accession',
                     'start', 'end', 'strand', 'attributes', 'protein_context_id'}
    assert expected_cols.issubset(set(feature_df.columns))
    # Same order contract as NCBI feature table: sorted by contig, then start.
    assert feature_df['start'].is_monotonic_increasing or (
        feature_df.groupby('genomic_accession')['start'].apply(
            lambda s: s.is_monotonic_increasing).all())
    # protein_context_id matches the documented composition.
    first = feature_df.iloc[0]
    assert first['protein_context_id'] == (
        f"{first['product_accession']}|{first['genomic_accession']}|"
        f"{first['start']}|{first['strand']}"
    )


def test_thrL_translation(parsed_gff):
    # pgaptmp_000001 = thrL: NC_000913.3:190-255, + strand, 66 bp → 21 aa starting with M.
    cds_records, contig_seqs = parsed_gff
    seq_df = gff.build_cds_seq_df(cds_records, contig_seqs)
    thrL_row = seq_df[seq_df['locus_tag'] == 'pgaptmp_000001'].iloc[0]
    assert len(thrL_row['seq']) == 66
    protein = gff.translate_cds(thrL_row['seq'])
    assert protein.startswith('M')
    assert len(protein) == 21


def test_minus_strand_extraction(parsed_gff):
    # Any minus-strand CDS: the extracted sequence should be the reverse-complement of the
    # genomic slice (and should translate to a protein starting with M).
    cds_records, contig_seqs = parsed_gff
    seq_df = gff.build_cds_seq_df(cds_records, contig_seqs)
    feature_df = gff.build_feature_df(cds_records)
    merged = feature_df.merge(seq_df, on='protein_context_id')
    minus = merged[merged['strand'] == '-'].iloc[0]
    # For a single-segment minus CDS, extracted seq = RC(genomic[start-1:end]).
    contig = contig_seqs[minus['genomic_accession']]
    expected = gff.reverse_complement(contig[minus['start'] - 1:minus['end']])
    assert minus['seq'] == expected
    protein = gff.translate_cds(minus['seq'])
    assert protein.startswith('M')


def test_prepare_inputs(tmp_path):
    feature_df, cds_seq_df, len_df, faa_path = gff.prepare_inputs(
        GFF_FIXTURE, str(tmp_path))
    # FAA file was written and is non-empty.
    assert Path(faa_path).exists()
    assert Path(faa_path).stat().st_size > 0
    # Schemas.
    assert set(feature_df.columns) >= {'product_accession', 'genomic_accession',
                                       'start', 'end', 'strand', 'protein_context_id'}
    assert set(cds_seq_df.columns) >= {'protein_context_id', 'locus_tag', 'seq'}
    assert set(len_df.columns) == {'product_accession', 'len'}
    # Counts line up: one len_df row per unique protein.
    assert len(len_df) == cds_seq_df['locus_tag'].nunique()
    # protein_context_id is consistent between feature_df and cds_seq_df.
    assert set(cds_seq_df['protein_context_id']) == set(feature_df['protein_context_id'])
