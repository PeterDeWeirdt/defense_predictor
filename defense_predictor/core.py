from importlib import resources
from joblib import load
import pandas as pd
import numpy as np
import re
import torch
from pathlib import Path
from esm import pretrained, FastaBatchedDataset
from tqdm import tqdm
import warnings
import argparse
from datetime import datetime


def parse_ncbi_cds_from_genomic(cds_from_genomic_f):
    with open(cds_from_genomic_f) as f:
        out_dict = None
        out_list = list()
        for line in f:
            if line.startswith('>'):
                if out_dict is not None:
                    out_list.append(out_dict)
                out_dict = {'dna_seq': ''}
                out_dict['id'] = line.split()[0][1:]
                attributes = re.findall('\[([^=]+)=([^=]+)\]', line)
                for k, v in attributes:
                    out_dict[k] = v
            else:
                out_dict['dna_seq'] += line.strip()
        out_list.append(out_dict)
    out_df = pd.DataFrame(out_list)
    if 'pseudo' in out_df.columns:
        out_df = out_df[out_df['pseudo'].isna()]
    out_df['strand'] = ['-' if x else '+' for x in 
                        out_df['location'].str.contains('complement')]
    out_df['start'] = out_df['location'].str.extract('([0-9]+)\.\.').astype(int)
    out_df['genomic_accession'] = out_df['id'].str.extract('lcl\|(.+)_cds')
    out_df = out_df[['genomic_accession', 'start', 'protein_id', 'strand', 'dna_seq']].rename(columns={'protein_id': 'product_accession'})
    return out_df
                

def parse_ncbi_protein_fasta(protein_fasta_f):
    with open(protein_fasta_f) as f:
        out_dict = None
        out_list = list()
        for line in f:
            if line.startswith('>'):
                if out_dict is not None:
                    out_list.append(out_dict)
                product_accession = line.split()[0][1:]
                out_dict = {'product_accession': product_accession, 'protein_seq': ''}
            else:
                out_dict['protein_seq'] += line.strip()
        out_list.append(out_dict)
    out_df = pd.DataFrame(out_list)
    return out_df

                
def get_ncbi_seq_info(ncbi_feature_table, ncbi_cds_from_genomic, ncbi_protein_fasta):
    seq_info_df = pd.read_table(ncbi_feature_table)
    seq_info_df['attributes'] = seq_info_df['attributes'].astype(str)
    seq_info_df = (seq_info_df[(seq_info_df['# feature'] == 'CDS') & 
                                ~seq_info_df['attributes'].str.contains('pseudo', na=False)]
                                .reset_index(drop=True))
    seq_info_df['start'] = seq_info_df['start'].astype(int)
    seq_info_df['end'] = seq_info_df['end'].astype(int)
    cds_from_genomic_df = parse_ncbi_cds_from_genomic(ncbi_cds_from_genomic)
    protein_fasta_df = parse_ncbi_protein_fasta(ncbi_protein_fasta)
    seq_info_df = (seq_info_df
                   .merge(cds_from_genomic_df, on=['genomic_accession', 'start', 'product_accession', 'strand'], how='inner')
                   .merge(protein_fasta_df, on='product_accession', how='inner'))
    seq_info_df['protein_context_id'] = (seq_info_df['product_accession'] + '|' +
                                         seq_info_df['genomic_accession'] + '|' + 
                                         seq_info_df['start'].astype(str) + '|' + 
                                         seq_info_df['strand'])
    seq_info_df = seq_info_df[['protein_context_id', 'product_accession', 'name', 'symbol',
                               'genomic_accession', 'start', 'end', 'strand', 
                               'dna_seq', 'protein_seq']]  
    return seq_info_df


def parse_prokka_gff(gff_f):
    gff_list = list()
    with open(gff_f) as f:
        for line in f:
            if line.startswith('##FASTA'):
                break
            elif not line.startswith('#'):
                gff_list.append(line.strip().split('\t'))
    gff_df = pd.DataFrame(gff_list,
                          columns=['genomic_accession', 'source', 'type', 'start', 
                                   'end', 'score', 'strand', 'phase', 'attributes'])
    attributes_list = list()
    for attr in gff_df['attributes']:
        attributes_list.append(dict([x.split('=') for x in attr.split(';')]))
    attributes_df = pd.DataFrame(attributes_list)
    gff_df = pd.concat([gff_df, attributes_df], axis=1).drop('attributes', axis=1)
    return gff_df


def parse_prokka_ffn(ffn_f):
    with open(ffn_f) as f:
        out_dict = None
        out_list = list()
        for line in f:
            if line.startswith('>'):
                if out_dict is not None:
                    out_list.append(out_dict)
                out_dict = {'dna_seq': ''}
                out_dict['ID'] = line.split()[0][1:]
            else:
                out_dict['dna_seq'] += line.strip()
        out_list.append(out_dict)
    out_df = pd.DataFrame(out_list)
    return out_df


def parse_prokka_faa(faa_f):
    with open(faa_f) as f:
        out_dict = None
        out_list = list()
        for line in f:
            if line.startswith('>'):
                if out_dict is not None:
                    out_list.append(out_dict)
                out_dict = {'ID': line.split()[0][1:], 'protein_seq': ''}
            else:
                out_dict['protein_seq'] += line.strip()
        out_list.append(out_dict)
    out_df = pd.DataFrame(out_list)
    return out_df


def get_prokka_seq_info(prokka_gff, prokka_ffn, prokka_faa):
    seq_info_df = parse_prokka_gff(prokka_gff)
    seq_info_df = (seq_info_df[seq_info_df['type'] == 'CDS']
                   .reset_index(drop=True))
    seq_info_df['start'] = seq_info_df['start'].astype(int)
    seq_info_df['end'] = seq_info_df['end'].astype(int)
    ffn_df = parse_prokka_ffn(prokka_ffn)
    faa_df = parse_prokka_faa(prokka_faa)
    seq_info_df = (seq_info_df.merge(ffn_df, on='ID', how='inner')
                   .merge(faa_df, on='ID', how='inner'))
    seq_info_df = seq_info_df.rename(columns={'ID': 'product_accession', 
                                              'product': 'name',
                                              'gene': 'symbol'})
    seq_info_df['protein_context_id'] = (seq_info_df['product_accession'] + '|' +
                                         seq_info_df['genomic_accession'] + '|' +
                                         seq_info_df['start'].astype(str) + '|' +
                                         seq_info_df['strand'])
    seq_info_df = seq_info_df[['protein_context_id', 'product_accession', 'name', 'symbol',
                               'genomic_accession', 'start', 'end', 'strand',
                               'dna_seq', 'protein_seq']]
    return seq_info_df


def get_esm2_encodings(seq_info_df, device, 
                       toks_per_batch=4096, truncation_seq_len=1022, 
                       repr_layer=30, model_location='esm2_t30_150M_UR50D.pt'):
    # load model
    model_path = str(Path(__file__).parent / model_location)
    model, alphabet = pretrained.load_model_and_alphabet(model_path)
    device = torch.device(device)
    model = model.to(device)
    # prepare data
    unique_seq_df = seq_info_df[['protein_seq']].drop_duplicates()
    dataset = FastaBatchedDataset(unique_seq_df['protein_seq'], unique_seq_df['protein_seq'])
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              collate_fn=alphabet.get_batch_converter(truncation_seq_len), 
                                              batch_sampler=batches)
    n_feats = 640
    feat_names = ['ft' + str(i + 1) for i in range(n_feats)]
    esm2_encoding_list = list()
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader), 
                                                    total=len(batches), position=0):
            toks = toks.to(device, non_blocking=True)
            out = model(toks, repr_layers=[repr_layer], return_contacts=False)
            representations = out['representations'][repr_layer]
            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_len, len(strs[i]))
                mean_rep = representations[i, 1:truncate_len+1].mean(0).cpu().numpy()
                esm2_encoding_list.append(pd.Series(mean_rep, index=feat_names, name=label))
    esm2_encoding_df = pd.DataFrame(esm2_encoding_list)
    return esm2_encoding_df


def test_get_esm2_encodings(seq_fasta, device, 
                       toks_per_batch=4096, truncation_seq_len=1022, 
                       repr_layer=30, model_location='esm2_t30_150M_UR50D.pt'):
    # load model
    model_path = str(Path(__file__).parent / model_location)
    model, alphabet = pretrained.load_model_and_alphabet(model_path)
    device = torch.device(device)
    model = model.to(device)
    # prepare data
    dataset = FastaBatchedDataset.from_file(seq_fasta)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              collate_fn=alphabet.get_batch_converter(truncation_seq_len), 
                                              batch_sampler=batches)
    n_feats = 640
    feat_names = ['ft' + str(i + 1) for i in range(n_feats)]
    esm2_encoding_list = list()
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader), 
                                                    total=len(batches), position=0):
            toks = toks.to(device, non_blocking=True)
            out = model(toks, repr_layers=[repr_layer], return_contacts=False)
            representations = out['representations'][repr_layer]
            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_len, len(strs[i]))
                mean_rep = representations[i, 1:truncate_len+1].mean(0).cpu().numpy()
                esm2_encoding_list.append(pd.Series(mean_rep, index=feat_names, name=label))
            break
    esm2_encoding_df = pd.DataFrame(esm2_encoding_list)
    return esm2_encoding_df


def get_dna_features(seq_info_df):
    out_df = seq_info_df[['protein_context_id', 'dna_seq', 'start', 'end']].copy()
    # Note: length was originally calculated from the cds_from_genomic fasta, so length of sequences with ribosomal slippage are shorter in training data
    out_df['len'] = out_df['end'] - out_df['start']
    out_df['seq_len'] = out_df['dna_seq'].str.len()
    out_df['gc_frac'] = out_df['dna_seq'].str.count('G|C')/out_df['seq_len'] 
    nts = ['A', 'C', 'T', 'G']
    di_nts = []
    for n1 in nts:
        for n2 in nts:
            di_nts.append(n1 + n2)
    motifs = nts + di_nts
    frac_cols = ['gc_frac']
    for motif in motifs:
        col_name = motif + '_frac'
        frac_cols.append(col_name)
        out_df[col_name] = out_df['dna_seq'].str.count(motif)/out_df['seq_len']
    out_cols = ['protein_context_id', 'len']
    for feat in frac_cols:
        scaled_col = 'scaled_' + feat
        out_cols.append(scaled_col)
        out_df[scaled_col] = (out_df[feat] - out_df[feat].mean())/out_df[feat].std()
    out_df = out_df[out_cols]
    return out_df


def get_gene_neighbors(seq_info_df, n_neighbors=2):
    sorted_seq_info = seq_info_df.sort_values(['genomic_accession', 'start']).reset_index(drop=True)
    gene_neighbors_list = list()
    for i, center_row in sorted_seq_info.iterrows():
        center_id = center_row['protein_context_id']
        center_genomic_accession = center_row['genomic_accession']
        center_strand = center_row['strand']
        gene_neighbor_df = sorted_seq_info.iloc[max(i - n_neighbors, 0):(i + n_neighbors + 1), :]
        gene_neighbor_df = gene_neighbor_df[gene_neighbor_df['genomic_accession'] == center_genomic_accession]
        gene_neighbor_out = (gene_neighbor_df[['product_accession', 'protein_context_id']].reset_index()
                                .rename(columns={'index': 'relative_position'}))
        gene_neighbor_out['relative_position'] = gene_neighbor_out['relative_position'] - i
        if center_strand == '-':
            gene_neighbor_out['relative_position'] = -gene_neighbor_out['relative_position']
        gene_neighbor_out['center_id'] = center_id
        gene_neighbors_list.append(gene_neighbor_out)
    gene_neighbors_df = pd.concat(gene_neighbors_list)
    return gene_neighbors_df


def get_protein_dist(center_seq_id, context_df):
    out_dict = {'center_id': center_seq_id}
    context_df = (context_df.sort_values('start', ascending=True)
                  [['protein_context_id', 'start', 'end', 'relative_position']]
                  .dropna()
                  .reset_index(drop=True))
    if len(context_df):
        prev_end = context_df.loc[0, 'end']
        prev_pos = context_df.loc[0, 'relative_position']
        for _, row in context_df.iloc[1:, :].iterrows():
            curr_start = row['start']
            curr_pos = row['relative_position']
            if abs(curr_pos - prev_pos) != 1: # missing value
                break
            sep = curr_start - prev_end
            pos_key = ':'.join([str(x) for x in sorted([curr_pos, prev_pos])])
            out_dict['dist_' + pos_key] = sep
            prev_end =  row['end']
            prev_pos = row['relative_position']
    return out_dict


def get_neighbor_features(gene_neighbor_df, seq_info_df):
    gene_neighbor_info = gene_neighbor_df.merge(seq_info_df, on=['protein_context_id', 'product_accession'], how='inner')
    co_directional_list = list()
    dist_list = list()
    for center_id, center_group in gene_neighbor_info.groupby('center_id'):
        center_group = center_group.sort_values('start')
        center_row = center_group[center_group['protein_context_id'] == center_id].squeeze()
        center_strand = center_row['strand']
        center_group['co_directional'] = (center_group['strand'] == center_strand).astype(int)
        co_directional_list.append(center_group)
        dist_dict = get_protein_dist(center_id, center_group)
        dist_list.append(dist_dict)
    co_directional_df = pd.concat(co_directional_list)
    wide_co_directional_df = co_directional_df.pivot(index='center_id', columns='relative_position', values=['co_directional'])
    wide_co_directional_df.columns = ['_'.join([str(y) for y in x]) for x in wide_co_directional_df.columns.to_flat_index()]
    dist_df = pd.DataFrame(dist_list).set_index('center_id')
    merged_neighbor_features = wide_co_directional_df.merge(dist_df, left_index=True, right_index=True, how='outer')
    return merged_neighbor_features


def fill_missing_features(feature_df):
    fill_values = {'len': 869, 'dist': 70, 'co_directional': 2}
    for feature_base, fill_value in fill_values.items():
        fill_cols = [x for x in feature_df.columns if feature_base in x]
        feature_df[fill_cols] = feature_df[fill_cols].fillna(fill_value)
    feature_df = feature_df.fillna(0)
    return feature_df


def reoder_feature_columns(feature_df):
    columns = pd.read_csv(Path(__file__).parent / 'x_columns.csv')['0']
    feature_df = feature_df[columns]
    return feature_df


def load_model():
    with resources.files('defense_predictor').joinpath('beaker_v3.pkl').open('rb') as f:
        return load(f)


def predict(data):
    model = load_model()
    probs = model.predict_proba(data)[:, 1]
    output_df = pd.DataFrame(index=data.index)
    output_df['defense_probability'] = probs
    output_df['defense_log_odds'] = np.log(probs / (1 - probs))
    return output_df


def run_defense_predictor(ncbi_feature_table=None,  ncbi_cds_from_genomic=None, ncbi_protein_fasta=None, 
                          prokka_gff=None, prokka_ffn=None, prokka_faa=None, 
                          device=None):
    for f in ['beaker_v3.pkl', 'esm2_t30_150M_UR50D.pt', 'esm2_t30_150M_UR50D-contact-regression.pt']:
        if not Path(__file__).parent.joinpath(f).exists():
            raise FileNotFoundError(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  
        print("Reading data")
        if (ncbi_feature_table is not None) or (ncbi_cds_from_genomic is not None) or (ncbi_protein_fasta is not None):
            for f in [ncbi_feature_table, ncbi_cds_from_genomic, ncbi_protein_fasta]:
                if f is None:
                    raise ValueError('ncbi_feature_table, ncbi_cds_from_genomic, and ncbi_protein_fasta are required if input_type is ncbi')
            seq_info_df = get_ncbi_seq_info(ncbi_feature_table, ncbi_cds_from_genomic, ncbi_protein_fasta)
        elif (prokka_gff is not None) or (prokka_ffn is not None) or (prokka_faa is not None):
            for f in [prokka_gff, prokka_ffn, prokka_faa]:
                if f is None:
                    raise ValueError('prokka_gff, prokka_ffn, and prokka_faa are required if input_type is prokka')
            seq_info_df = get_prokka_seq_info(prokka_gff, prokka_ffn, prokka_faa)
        print("Getting ESM2 encodings")
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        esm2_encodings = get_esm2_encodings(seq_info_df, device)
        print("Calculating remaining features")
        dna_feature_df = get_dna_features(seq_info_df)
        self_feature_df = (seq_info_df[['protein_context_id', 'protein_seq']]
                        .merge(esm2_encodings, left_on='protein_seq', right_index=True)
                        .drop(columns='protein_seq')
                        .merge(dna_feature_df, how='inner', on='protein_context_id'))
        gene_neighbor_df = get_gene_neighbors(seq_info_df)
        gene_neighbor_feature_df = (gene_neighbor_df.merge(self_feature_df, on='protein_context_id', how='inner'))
        feature_df = gene_neighbor_feature_df.pivot(index='center_id',
                                                    columns='relative_position',
                                                    values=self_feature_df.columns[1:])
        feature_df.columns = ['_'.join([str(y) for y in x]) for x in feature_df.columns.to_flat_index()]
        neighbor_features = get_neighbor_features(gene_neighbor_df, seq_info_df)
        feature_df = feature_df.merge(neighbor_features, how='inner', left_index=True, right_index=True)
        feature_df = fill_missing_features(feature_df)
        feature_df = reoder_feature_columns(feature_df)
        print('Making predictions')
        prediction_df = predict(feature_df)
        out_df = (seq_info_df.merge(prediction_df, left_on='protein_context_id', right_index=True))
    return out_df
    

def main():
    parser = argparse.ArgumentParser(description='Run defense predictor')
    parser.add_argument('--ncbi_feature_table', type=str, help='Path to NCBI feature table')
    parser.add_argument('--ncbi_cds_from_genomic', type=str, help='Path to NCBI CDS from genomic file')
    parser.add_argument('--ncbi_protein_fasta', type=str, help='Path to NCBI protein FASTA file')
    parser.add_argument('--prokka_gff', type=str, help='Path to Prokka GFF file')
    parser.add_argument('--prokka_ffn', type=str, help='Path to Prokka FFN file')
    parser.add_argument('--prokka_faa', type=str, help='Path to Prokka FAA file')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to run the predictor on')
    parser.add_argument('--output', type=str, help='Filepath for csv output file')
    args = parser.parse_args()
    out_df = run_defense_predictor(ncbi_feature_table=args.ncbi_feature_table, 
                          ncbi_cds_from_genomic=args.ncbi_cds_from_genomic, 
                          ncbi_protein_fasta=args.ncbi_protein_fasta, 
                          prokka_gff=args.prokka_gff, 
                          prokka_ffn=args.prokka_ffn, 
                          prokka_faa=args.prokka_faa, 
                          device=args.device)
    if args.output is None:
        output = 'defense_predictions' + datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
    else:
        output = args.output
    out_df.to_csv(output, index=False)


if __name__ == '__main__':
    main()