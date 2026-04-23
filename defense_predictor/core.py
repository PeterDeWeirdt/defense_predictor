import pandas as pd
from esm import pretrained, FastaBatchedDataset
import torch
from tqdm import tqdm
import re
from joblib import load
import os
import numpy as np
import argparse
from datetime import datetime
import tempfile
from importlib import resources
from pathlib import Path
import warnings

from . import pgap as _pgap


def get_feature_df(ft_file):
    """Read an NCBI feature table and return a filtered dataframe with one new column:
    "protein_context_id".

    Filters out pseudogenes and rows that are not CDS, then builds a
    "protein_context_id" column of the form
    `product_accession|genomic_accession|start|strand`.

    Args:
        ft_file: path to an NCBI "*_feature_table.txt" file.

    Returns:
        DataFrame of CDS rows with all original feature-table columns plus a
        "protein_context_id" column.
    """
    feature_df = pd.read_table(ft_file)
    feature_df['attributes'] = feature_df['attributes'].astype(str)
    filtered_feature_df = (feature_df[(feature_df['# feature'] == 'CDS') &
                                      ~(feature_df['attributes'].str.contains('pseudo', na=False))]
                           .reset_index(drop=True))
    filtered_feature_df['protein_context_id'] = (filtered_feature_df['product_accession'].astype(str) + '|' +
                                                 filtered_feature_df['genomic_accession'].astype(str) + '|' +
                                                 filtered_feature_df['start'].astype(str) + '|' +
                                                 filtered_feature_df['strand'])
    return filtered_feature_df


def get_neighbor_df(feature_df):
    """Get neighbors +/-2 genes away from each CDS on the same-contig.

    Relative positions are flipped on minus-strand genes so ``-1`` is always the
    upstream (5') neighbor.

    Args:
        feature_df: DataFrame that contains "protein_context_id",
            "product_accession", "genomic_accession", "strand", "start",
            "end", sorted by "genomic_accession" and "start".

    Returns:
        Long-format DataFrame with columns "center_id", "center_strand",
        "relative_position" (-2..+2), "protein_context_id",
        "product_accession", "strand", "start", "end".
    """
    n_neighbors = 2
    protein_neighbor_list = list()
    for i, center_row in tqdm(feature_df.iterrows(),
                              total=len(feature_df), 
                              position=0):
        center_id = center_row['protein_context_id']
        center_genomic_accession = center_row['genomic_accession']
        center_strand = center_row['strand']
        protein_neighbor_df = feature_df.iloc[max(i - n_neighbors, 0):(i + n_neighbors + 1), :]
        protein_neighbor_df = protein_neighbor_df[protein_neighbor_df['genomic_accession'] == center_genomic_accession]
        protein_neighbor_out = (protein_neighbor_df[['product_accession', 'protein_context_id', 'strand', 'start', 'end']].reset_index()
                                .rename(columns={'index': 'relative_position'}))
        protein_neighbor_out['relative_position'] = protein_neighbor_out['relative_position'] - i
        protein_neighbor_out['center_strand'] = center_strand
        if center_strand == '-':
            protein_neighbor_out['relative_position'] = -protein_neighbor_out['relative_position']
        protein_neighbor_out['center_id'] = center_id
        protein_neighbor_list.append(protein_neighbor_out)
    protein_neighbor_df = pd.concat(protein_neighbor_list)
    return protein_neighbor_df


def get_representations(faa_file):
    """Compute mean-pooled ESM2 embeddings for every protein in a FASTA file.

    Loads "esm2_t30_150M_UR50D", runs a forward pass, and
    averages the final layer (30) token representations across each sequence (truncated to
    1022 residues). Uses CUDA when available.

    Args:
        faa_file: path to a protein FASTA file.

    Returns:
        DataFrame of shape "(n_proteins, 640)" with columns "ft1".."ft640"
        and index values equal to each record's FASTA ID.
    """
    model_location = 'esm2_t30_150M_UR50D.pt'
    model_path = str(Path(__file__).parent / model_location)
    toks_per_batch = 4096
    truncation_seq_length = 1022
    repr_layer = 30
    model, alphabet = pretrained.load_model_and_alphabet(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    assert (-(model.num_layers + 1) <= repr_layer <= model.num_layers)
    repr_layer = (repr_layer + model.num_layers + 1) % (model.num_layers + 1)
    print('repr layer', repr_layer)
    dataset = FastaBatchedDataset.from_file(faa_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )
    rep_list = list()
    label_list = list()
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader),
                                                    total=len(batches), 
                                                    position=0):
            if torch.cuda.is_available():
                toks = toks.to(device='cuda', non_blocking=True)
            out = model(toks, repr_layers=[repr_layer], return_contacts=False)
            representations = out['representations'][repr_layer]
            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                mean_rep = representations[i, 1:truncate_len + 1].mean(0).cpu().numpy()
                label_list.append(label.split(' ')[0])
                rep_list.append(mean_rep)
    rep_df = pd.DataFrame(rep_list)
    rep_df.columns = ['ft' + str(i+1) for i in range(rep_df.shape[1])]
    rep_df.index = label_list
    return rep_df


def _compute_motif_features(seq_df):
    """Compute z-scored GC and mono/di-nucleotide content.

    For each nucleotide sequence, computes the fraction of G+C, the fraction of each
    of the 4 nucleotides, and the fraction of each of the 16 dinucleotides,
    then z-scores each feature across the full set of input sequences, corresponding
    to all sequences for a genome.

    Args:
        seq_df: DataFrame with at least "protein_context_id" and "seq"
            (uppercase nucleotide string) columns.

    Returns:
        DataFrame with "protein_context_id" plus one "scaled_<motif>_frac"
        column per motif ("scaled_gc_frac", "scaled_A_frac", ...,
        "scaled_GG_frac").
    """
    nts = ['A', 'C', 'T', 'G']
    di_nts = [n1 + n2 for n1 in nts for n2 in nts]
    motifs = nts + di_nts
    out = seq_df[['protein_context_id', 'seq']].copy()
    out['gc_frac'] = out['seq'].str.count('G|C') / out['seq'].str.len()
    out['scaled_gc_frac'] = (out['gc_frac'] - out['gc_frac'].mean()) / out['gc_frac'].std()
    out_cols = ['protein_context_id', 'scaled_gc_frac']
    for motif in motifs:
        col = motif + '_frac'
        out[col] = out['seq'].str.count(motif) / out['seq'].str.len()
    for motif in motifs:
        col = 'scaled_' + motif + '_frac'
        unscaled_col = motif + '_frac'
        out[col] = (out[unscaled_col] - out[unscaled_col].mean()) / out[unscaled_col].std()
        out_cols.append(col)
    return out[out_cols]


def get_motifs(fna_file):
    """Parse an NCBI CDS-from-genomic FASTA and compute nucleotide features.

    Extracts per-record bracketed attributes (e.g. "[protein_id=...]",
    "[location=...]"), drops pseudogenes, builds a "protein_context_id" for
    each CDS.

    Args:
        fna_file: path to an NCBI "*_cds_from_genomic.fna" file.

    Returns:
        DataFrame with "protein_context_id" plus the z-scored GC and
        mono/di-nucleotide content.
    """
    seq = ''
    seq_list = []
    seq_info = dict()
    for line in open(fna_file):
        line = line.strip()
        if line.startswith('>'):
            if seq:
                seq_info['seq'] = seq
                seq_list.append(seq_info)
                seq_info = dict()
                seq = ''
            seq_info['id'] = line.split(' ')[0][1:]
            regex = r'\[([^=]+)=([^=]+)\]'
            attributes = re.findall(regex, line)
            for key, value in attributes:
                seq_info[key] = value
        else:
            seq += line
    seq_info['seq'] = seq
    seq_list.append(seq_info)
    seq_df = pd.DataFrame(seq_list)
    if 'pseudo' in seq_df.columns:
        filtered_seq_df = seq_df[seq_df['pseudo'].isna()].copy()
    else:
        filtered_seq_df = seq_df
    filtered_seq_df['strand'] = ['-' if x else '+' for x in
                                 filtered_seq_df['location'].str.contains('complement')]
    filtered_seq_df['start'] = (filtered_seq_df['location']
                                .str.extract(r'([0-9]+)\.\.').astype(int))
    filtered_seq_df['genomic_locus'] = filtered_seq_df['id'].str.extract(r'lcl\|(.+)_cds')
    filtered_seq_df['protein_context_id'] = (filtered_seq_df['protein_id'] + '|' +
                                           filtered_seq_df['genomic_locus'] + '|' +
                                           filtered_seq_df['start'].astype(str) + '|' +
                                           filtered_seq_df['strand'])
    return _compute_motif_features(filtered_seq_df[['protein_context_id', 'seq']])


def parse_fasta(fasta_file):
    """Parse a FASTA file into a DataFrame of "id"/"sequence" rows.

    Args:
        fasta_file: path to a FASTA file.

    Returns:
        DataFrame with columns "id" and "sequence".
    """
    records = []
    current_id = None
    current_sequence_parts = []
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            if line.startswith('>'):
                # If we have a previous sequence, save it
                if current_id is not None:
                    sequence = "".join(current_sequence_parts)
                    records.append({'id': current_id, 'sequence': sequence})
                # Start a new record
                # The ID is the string after '>' and before the first space
                current_id = line[1:].split()[0]
                current_sequence_parts = []
            else:
                # Append sequence line to the current record
                if current_id is not None:
                    current_sequence_parts.append(line)
        # After the loop, save the very last record in the file
        if current_id is not None:
            sequence = "".join(current_sequence_parts)
            records.append({'id': current_id, 'sequence': sequence})
    records_df = pd.DataFrame(records)
    return records_df


def get_seq_len(fasta_file):
    """Return the sequence length of every record in a FASTA file.

    Args:
        fasta_file: path to a FASTA file.

    Returns:
        DataFrame with columns "product_accession" and "len".
    """
    seq_df = parse_fasta(fasta_file)
    seq_df['len'] = seq_df['sequence'].str.len()
    seq_df = (seq_df.rename(columns={'id': 'product_accession'})
              .drop(columns='sequence')
              .drop_duplicates())
    return seq_df


def get_directionality(neighbor_df):
    """Indicate whether each neighbor has the same orientation as its center gene.

    Args:
        neighbor_df: long-format neighbor DataFrame 
            with the columns "strand" and "center_strand".

    Returns:
        DataFrame with columns "protein_context_id", "center_id", and "co_directional".
    """
    out_df = neighbor_df.copy()
    out_df['co_directional'] = (out_df['strand'] == out_df['center_strand']).astype(int)
    out_df = out_df[['protein_context_id', 'center_id', 'co_directional']]
    return out_df


def get_gene_dist(center_seq_id, context_df):
    """Compute inter-gene distances within one center gene's window.

    Compute the gap between consecutive genes as 'next.start - curr.end'.

    Args:
        center_seq_id: the center gene's "protein_context_id".
        context_df: dataframe with columns "relative_position", "strand", 
            "start", "end"

    Returns:
        Dict with "center_id" and one "dist_<a>:<b>" key per adjacent pair
        of relative positions (e.g. "dist_-2:-1", "dist_-1:0", ...).
    """
    center_strand = context_df.loc[(context_df['relative_position'] == 0), 'strand'].item()
    if center_strand == '+':
        context_df = context_df.sort_values('relative_position', ascending=True)
    else:
        context_df = context_df.sort_values('relative_position', ascending=False)
    curr_end = context_df['end']
    next_start = context_df['start'].shift(-1)
    if (context_df['end'] < context_df['start']).any():
        context_df['wraparound'] = context_df['end'] < context_df['start']
        next_wraparound = context_df['wraparound'].shift(-1)
        distances = np.where(next_wraparound, 
                             -curr_end,
                             next_start - curr_end)
    else:
        distances = next_start - curr_end
    distances = list(distances)
    out_dict = {'center_id': center_seq_id}
    relative_positions = context_df['relative_position'].to_list()
    for i in range(len(context_df) - 1):
        pos_i = relative_positions[i]
        pos_j = relative_positions[i+1]
        out_dict['dist_' + 
                 ':'.join([str(min(pos_i, pos_j)), 
                           str(max(pos_i, pos_j))])] = distances[i]
    return out_dict


def get_distances(neighbor_df):
    """Compute inter-gene distances for every center gene in a neighbor table.

    Args:
        neighbor_df: long-format neighbor DataFrame
        
    Returns:
        DataFrame indexed by "center_id", with one "dist_<a>:<b>" column
        per adjacent pair of genes.
    """
    distance_list = [get_gene_dist(center_seq_id, context_df)
                     for center_seq_id, context_df in tqdm(neighbor_df.groupby('center_id'),
                                                           position=0)]
    distance_df = pd.DataFrame(distance_list)
    distance_df = distance_df.set_index('center_id')
    return distance_df


def load_model(model_f):
    """Load a pickled LightGBM model.

    Args:
        model_f: filename.

    Returns:
        The unpickled model object.
    """
    with resources.files('defense_predictor').joinpath(model_f).open('rb') as f:
        return load(f)
    
    
def defense_predictor(ft_file=None, fna_file=None, faa_file=None, pgap_gff=None,
                      rep_df=None, model_feature_df=None):
    """Run the DefensePredictor pipeline end-to-end on a genome.

    Accepts either the three NCBI input files or a single PGAP GFF3 with
    embedded genomic FASTA (exactly one of the two input modes must be
    provided). Builds the feature matrix (ESM2 protein embeddings, nucleotide
    motif composition, protein length, neighbor co-directionality, and
    inter-gene distances), then averages log-odds predictions from a 5-fold
    LightGBM ensemble.

    Args:
        ft_file: path to an NCBI "*_feature_table.txt". NCBI mode.
        fna_file: path to an NCBI "*_cds_from_genomic.fna". NCBI mode.
        faa_file: path to an NCBI "*_protein.faa". NCBI mode.
        pgap_gff: path to a PGAP "annot_with_genomic_fasta.gff" file. PGAP
            mode. Mutually exclusive with the three NCBI args.
        rep_df: optional precomputed ESM2 embedding DataFrame. 
            If provided, the ESM2 forward pass is skipped.
        model_feature_df: optional precomputed full feature matrix. If
            provided, all feature-extraction steps are skipped and the
            ensemble is run directly on this matrix.

    Returns:
        Tuple `(out_df, model_feature_df)` where `out_df` is a DataFrame
        with one row per gene containing "protein_context_id",
        "mean_log_odds", "sd_log_odds", "min_log_odds",
        "max_log_odds", and all feature-table columns; and
        "model_feature_df" is the full feature matrix used for prediction
        (indexed by "center_id").

    Raises:
        ValueError: if neither input mode (or both) is provided.
        FileNotFoundError: if any of the model weights are missing.
    """
    ncbi_args = (ft_file, fna_file, faa_file)
    ncbi_any = any(x is not None for x in ncbi_args)
    ncbi_all = all(x is not None for x in ncbi_args)
    if pgap_gff is not None and ncbi_any:
        raise ValueError('Provide either pgap_gff or the three NCBI inputs, not both.')
    if pgap_gff is None and not ncbi_all:
        raise ValueError('Must provide either pgap_gff or all three NCBI inputs '
                         '(ft_file, fna_file, faa_file).')
    model_fs = ['beaker_fold_0.pkl', 'beaker_fold_1.pkl', 'beaker_fold_2.pkl',
                'beaker_fold_3.pkl', 'beaker_fold_4.pkl']
    for f in model_fs + ['esm2_t30_150M_UR50D.pt', 'esm2_t30_150M_UR50D-contact-regression.pt']:
        if not Path(__file__).parent.joinpath(f).exists():
            raise FileNotFoundError(f)

    tmpdir_ctx = tempfile.TemporaryDirectory() if pgap_gff is not None else None
    try:
        if pgap_gff is not None:
            feature_df, cds_seq_df, len_df, faa_path = _pgap.prepare_pgap_inputs(
                pgap_gff, tmpdir_ctx.name)
        else:
            feature_df = get_feature_df(ft_file)
            cds_seq_df = None
            len_df = None
            faa_path = faa_file
        if model_feature_df is None:
            print('Getting neighbors')
            neighbor_df = get_neighbor_df(feature_df)
            if rep_df is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    print('Getting representations')
                    rep_df = get_representations(faa_path)
    finally:
        if tmpdir_ctx is not None:
            tmpdir_ctx.cleanup()
    if model_feature_df is None:
        wide_rep_df = (neighbor_df[['product_accession', 'center_id', 'relative_position']]
                       .set_index('product_accession')
                       .merge(rep_df, how='left', left_index=True, right_index=True)
                       .pivot(index='center_id', columns='relative_position'))
        wide_rep_df = wide_rep_df.fillna(0)
        wide_rep_df.columns = [x[0] + '_' + str(x[1]) for x in wide_rep_df.columns]
        # NT df
        if pgap_gff is not None:
            nt_df = _compute_motif_features(cds_seq_df[['protein_context_id', 'seq']])
        else:
            nt_df = get_motifs(fna_file)
        wide_nt_df = (neighbor_df[['protein_context_id', 'center_id', 'relative_position']]
                      .merge(nt_df, how='left', on='protein_context_id')
                      .drop(columns='protein_context_id')
                      .pivot(index='center_id', columns='relative_position'))
        wide_nt_df = wide_nt_df.fillna(1.1)
        wide_nt_df.columns = [x[0] + '_' + str(x[1]) for x in wide_nt_df.columns]
        # Len df
        if len_df is None:
            len_df = get_seq_len(faa_file)
        wide_len_df = (neighbor_df[['product_accession', 'center_id', 'relative_position']]
                       .merge(len_df, how='left', on='product_accession')
                       .drop(columns='product_accession')
                       .pivot(index='center_id', columns='relative_position'))
        wide_len_df = wide_len_df.fillna(0)
        wide_len_df.columns = [x[0] + '_' + str(x[1]) for x in wide_len_df.columns]
        # Directionality df
        directionality_df = get_directionality(neighbor_df)
        wide_directionality_df = (neighbor_df[['protein_context_id', 'center_id', 'relative_position']]
                                  .merge(directionality_df, how='left', on=['protein_context_id', 'center_id'])
                                  .drop(columns='protein_context_id')
                                  .pivot(index='center_id', columns='relative_position'))
        wide_directionality_df = wide_directionality_df.fillna(2)
        wide_directionality_df.columns = [x[0] + '_' + str(x[1]) for x in wide_directionality_df.columns]
        # Distance df
        print('Calculating distances')
        distance_df = get_distances(neighbor_df)
        distance_df = distance_df.fillna(-200)
        model_feature_df = wide_rep_df
        for df in [wide_nt_df, wide_len_df, wide_directionality_df, distance_df]:
            model_feature_df = (model_feature_df.merge(df, left_index=True, right_index=True, how='inner'))
    model_feature_mat = model_feature_df.to_numpy()
    print('Making predictions')
    pred_list = list()
    for model_f in tqdm(model_fs, position=0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = load_model(model_f)
            prob = model.predict_proba(model_feature_mat, num_iteration=model.best_iteration_)[:, 1]
        log_odds = np.log(prob/(1-prob))
        pred_list.append(log_odds)
    cat_preds = np.stack(pred_list)
    out_df = (model_feature_df.reset_index()
              .rename(columns={'center_id': 'protein_context_id'})
              [['protein_context_id']]).copy()
    out_df['mean_log_odds'] = cat_preds.mean(axis=0)
    out_df['sd_log_odds'] = cat_preds.std(axis=0)
    out_df['min_log_odds'] = cat_preds.min(axis=0)
    out_df['max_log_odds'] = cat_preds.max(axis=0)
    out_df = (out_df.merge(feature_df, how='inner', on='protein_context_id'))
    return out_df, model_feature_df
    

def main():
    """CLI entry point for `defense_predictor`.

    Parses command-line arguments, runs `defense_predictor`, and writes
    the output DataFrame to CSV.
    """
    parser = argparse.ArgumentParser(description='Run defense predictor')
    parser.add_argument('--ncbi_feature_table', type=str, help='Path to NCBI feature table')
    parser.add_argument('--ncbi_cds_from_genomic', type=str, help='Path to NCBI CDS from genomic file')
    parser.add_argument('--ncbi_protein_fasta', type=str, help='Path to NCBI protein FASTA file')
    parser.add_argument('--pgap_gff', type=str,
                        help='Path to a PGAP GFF3 file with embedded ##FASTA section '
                             '(alternative to the three --ncbi_* inputs)')
    parser.add_argument('--output', type=str, help='Filepath for csv output file')
    args = parser.parse_args()
    out_df, model_feature_df = defense_predictor(ft_file=args.ncbi_feature_table,
                                                 fna_file=args.ncbi_cds_from_genomic,
                                                 faa_file=args.ncbi_protein_fasta,
                                                 pgap_gff=args.pgap_gff)
    if args.output is None:
        output = f"defense_predictions_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    else:
        output = args.output
    out_df.to_csv(output, index=False)


if __name__ == '__main__':
    main()