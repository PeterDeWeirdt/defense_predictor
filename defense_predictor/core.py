import pandas as pd
from esm import pretrained, FastaBatchedDataset
import torch
from tqdm import tqdm
import re
from joblib import load
import os
import numpy as np
import argparse
import datetime
from importlib import resources
from pathlib import Path
import warnings


def get_feature_df(ft_file):
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


def get_motifs(fna_file):
    nts = ['A', 'C', 'T', 'G']
    di_nts = []
    for n1 in nts:
        for n2 in nts:
            di_nts.append(n1 + n2)
    motifs = nts + di_nts
    seq = ''
    seq_list = []
    seq_info = dict()
    for line in open(fna_file):
        line = line.strip()
        if '>' in line:
            if seq:
                seq_info['seq'] = seq
                seq_list.append(seq_info)
                seq_info = dict()
                seq = ''
            seq_info['id'] = line.split(' ')[0][1:]
            regex = '\[([^=]+)=([^=]+)\]'
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
                                .str.extract('([0-9]+)\.\.').astype(int))
    filtered_seq_df['genomic_locus'] = filtered_seq_df['id'].str.extract('lcl\|(.+)_cds')
    filtered_seq_df['protein_context_id'] = (filtered_seq_df['protein_id'] + '|' + 
                                           filtered_seq_df['genomic_locus'] + '|' +
                                           filtered_seq_df['start'].astype(str) + '|' +
                                           filtered_seq_df['strand'])
    filtered_seq_df = filtered_seq_df.drop(columns=['strand', 'start', 'genomic_locus', 'protein_id'])
    filtered_seq_df['gc_frac'] = (filtered_seq_df['seq'].str.count('G|C')/
                                  filtered_seq_df['seq'].str.len())
    filtered_seq_df['scaled_gc_frac'] = ((filtered_seq_df['gc_frac'] - 
                                          filtered_seq_df['gc_frac'].mean())/
                                         filtered_seq_df['gc_frac'].std())
    out_cols = ['protein_context_id', 'scaled_gc_frac']
    for motif in motifs:
        col = motif + '_frac'
        filtered_seq_df[col] = (filtered_seq_df['seq'].str.count(motif)/
                                filtered_seq_df['seq'].str.len())
    for motif in motifs:
        col = 'scaled_' + motif + '_frac'
        unscaled_col = motif + '_frac'
        filtered_seq_df[col] = ((filtered_seq_df[unscaled_col] - 
                                 filtered_seq_df[unscaled_col].mean())/
                                filtered_seq_df[unscaled_col].std())
        out_cols.append(col)
    filtered_seq_df = filtered_seq_df[out_cols]
    return filtered_seq_df


def parse_fasta(fasta_file):
    """
    Parses a FASTA file using only base Python and returns a list of dictionaries.
    Args:
        fasta_file (str): The path to the FASTA file.
    Returns:
        list: A list of dictionaries, where each dictionary represents a
              sequence record with 'id' and 'sequence' keys. Returns an
              empty list if the file is not found or is empty.
    """
    records = []
    current_id = None
    current_sequence_parts = []
    try:
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
    except FileNotFoundError:
        print(f"Error: The file '{fasta_file}' was not found.")
        return [] # Return an empty list on error
    except Exception as e:
        print(f"An error occurred while parsing the file: {e}")
        return []
    records_df = pd.DataFrame(records)
    return records_df


def get_seq_len(fasta_file):
    seq_df = parse_fasta(fasta_file)
    seq_df['len'] = seq_df['sequence'].str.len()
    seq_df = (seq_df.rename(columns={'id': 'product_accession'})
              .drop(columns='sequence')
              .drop_duplicates())
    return seq_df


def get_directionality(neighbor_df):
    out_df = neighbor_df.copy()
    out_df['co_directional'] = (out_df['strand'] == out_df['center_strand']).astype(int)
    out_df = out_df[['protein_context_id', 'center_id', 'co_directional']]
    return out_df


def get_gene_dist(center_seq_id, context_df):
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
    distance_list = [get_gene_dist(center_seq_id, context_df) 
                     for center_seq_id, context_df in tqdm(neighbor_df.groupby('center_id'),
                                                           position=0)]
    distance_df = pd.DataFrame(distance_list)
    distance_df = distance_df.set_index('center_id')
    return distance_df


def load_model(model_f):
    with resources.files('defense_predictor').joinpath(model_f).open('rb') as f:
        return load(f)
    
    
def defense_predictor(ft_file, fna_file, faa_file, rep_df=None, model_feature_df=None):
    model_fs = ['beaker_fold_0.pkl', 'beaker_fold_1.pkl', 'beaker_fold_2.pkl', 
                'beaker_fold_3.pkl', 'beaker_fold_4.pkl']
    for f in model_fs + ['esm2_t30_150M_UR50D.pt', 'esm2_t30_150M_UR50D-contact-regression.pt']:
        if not Path(__file__).parent.joinpath(f).exists():
            raise FileNotFoundError(f)
    feature_df = get_feature_df(ft_file)
    if model_feature_df is None:
        print('Getting neighbors')
        neighbor_df = get_neighbor_df(feature_df)
        # Rep df
        if rep_df is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print('Getting representations')
                rep_df = get_representations(faa_file)
        wide_rep_df = (neighbor_df[['product_accession', 'center_id', 'relative_position']]
                       .set_index('product_accession')
                       .merge(rep_df, how='left', left_index=True, right_index=True)
                       .pivot(index='center_id', columns='relative_position'))
        wide_rep_df = wide_rep_df.fillna(0)
        wide_rep_df.columns = [x[0] + '_' + str(x[1]) for x in wide_rep_df.columns]
        # NT df
        nt_df = get_motifs(fna_file)
        wide_nt_df = (neighbor_df[['protein_context_id', 'center_id', 'relative_position']]
                      .merge(nt_df, how='left', on='protein_context_id')
                      .drop(columns='protein_context_id')
                      .pivot(index='center_id', columns='relative_position'))
        wide_nt_df = wide_nt_df.fillna(1.1)
        wide_nt_df.columns = [x[0] + '_' + str(x[1]) for x in wide_nt_df.columns]
        # Len df
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
    parser = argparse.ArgumentParser(description='Run defense predictor')
    parser.add_argument('--ncbi_feature_table', type=str, help='Path to NCBI feature table')
    parser.add_argument('--ncbi_cds_from_genomic', type=str, help='Path to NCBI CDS from genomic file')
    parser.add_argument('--ncbi_protein_fasta', type=str, help='Path to NCBI protein FASTA file')
    parser.add_argument('--output', type=str, help='Filepath for csv output file')
    args = parser.parse_args()
    out_df, model_feature_df = defense_predictor(ft_file=args.ncbi_feature_table, 
                                                 fna_file=args.ncbi_cds_from_genomic, 
                                                 faa_file=args.ncbi_protein_fasta)
    if args.output is None:
        output = f"defense_predictions_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    else:
        output = args.output
    out_df.to_csv(output, index=False)


if __name__ == '__main__':
    main()