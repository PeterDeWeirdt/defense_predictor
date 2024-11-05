from importlib import resources
from joblib import load
import pandas as pd
import numpy as np
import re


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
    out_df['end'] = out_df['location'].str.extract('[\.\.|\>]([0-9]+)').astype(int)
    out_df['genomic_accession'] = out_df['id'].str.extract('lcl\|(.+)_cds')
    out_df = out_df[['genomic_accession', 'start', 'end', 'strand', 'dna_seq']]
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
                   .merge(cds_from_genomic_df, on=['genomic_accession', 'start', 'end', 'strand'], how='left') # Missing CDS' with ribosomal slippage
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


def get_esm2_encodings(seq_info_df):
    pass


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


def run(input_type, 
        ncbi_feature_table=None,  ncbi_cds_from_genomic=None, ncbi_protein_fasta=None, 
        prokka_gff=None, prokka_ffn=None, prokka_faa=None):
    if (input_type == 'ncbi'):
        for f in [ncbi_feature_table, ncbi_cds_from_genomic, ncbi_protein_fasta]:
            if f is None:
                raise ValueError('ncbi_feature_table, ncbi_cds_from_genomic, and ncbi_protein_fasta are required if input_type is ncbi')
        seq_info_df = get_ncbi_seq_info(ncbi_feature_table, ncbi_cds_from_genomic, ncbi_protein_fasta)
    elif (input_type == 'prokka'):
        for f in [prokka_gff, prokka_ffn, prokka_faa]:
            if f is None:
                raise ValueError('prokka_gff, prokka_ffn, and prokka_faa are required if input_type is prokka')
        seq_info_df = get_prokka_seq_info(prokka_gff, prokka_ffn, prokka_faa)
    esm2_encodings = get_esm2_encodings(seq_info_df)
