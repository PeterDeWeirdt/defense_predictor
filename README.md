# DefensePredictor: A Machine Learning Model to Discover Novel Prokaryotic Immune Systems

Python package to run DefensePredictor, a machine-learning model that leverages embeddings from a protein language model, [ESM2](https://github.com/facebookresearch/esm), to classify proteins as anti-phage defensive. 

For additional details, read the paper [here](https://www.science.org/doi/10.1126/science.adv7924).

### Installation

In a fresh [conda](https://anaconda.org/anaconda/conda) or other virutal environment, run:

```bash
pip install defense_predictor
defense_predictor_download
```

The first command downloads the python package from PyPI and the second downloads the model weights. Once model weights are downloaded you do not need to run this command again. 

### Requirements 

Requires `python >= 3.10`

### Usage

`defense_predictor` can be run as python code

```python
import defense_predictor as dfp

ncbi_feature_table = 'GCF_003333385.1_ASM333338v1_feature_table.txt'
ncbi_cds_from_genomic = 'GCF_003333385.1_ASM333338v1_cds_from_genomic.fna'
ncbi_protein_fasta = 'GCF_003333385.1_ASM333338v1_protein.faa'
output_df, feature_matrix = dfp.defense_predictor(ft_file=ncbi_feature_table, fna_file=ncbi_cds_from_genomic, faa_file=ncbi_protein_fasta)
output_df.head()                                    
```

Or from the command line

```bash
defense_predictor \
     --ncbi_feature_table GCF_003333385.1_ASM333338v1_feature_table.txt \
     --ncbi_cds_from_genomic GCF_003333385.1_ASM333338v1_cds_from_genomic.fna \ 
     --ncbi_protein_fasta GCF_003333385.1_ASM333338v1_protein.faa \
     --output GCF_003333385_defense_predictor_output.csv
```

Alternatively, `defense_predictor` can take a single [GFF3](https://useast.ensembl.org/info/website/upload/gff3.html?) file with embedded genomic FASTA:

```python
output_df, feature_matrix = dfp.defense_predictor(gff='annot_with_genomic_fasta.gff')
```

```bash
defense_predictor \
     --gff annot_with_genomic_fasta.gff \
     --output defense_predictor_output.csv
```

When given a GFF, `defense_predictor` translates proteins from the embedded genomic sequence using the bacterial codon table (`transl_table=11`) and uses each CDS's `locus_tag` as its identifier.

<br>

`defense_predictor` outputs the predicted log-odds of defense for each input protein in the columns `mean_log_odds`. We reccomend using a stringent log-odds cutoff of `4` to call a protein predicted defensive.

To see an example you can run the `defense_predictor_example.ipynb` in colab: <a href="https://colab.research.google.com/github/PeterDeWeirdt/defense_predictor/blob/main/defense_predictor_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 

We reccomend running `defense_predictor` on a computer with a cuda-enabled GPU, to maximize computational efficiency. 

### Inputs

The NCBI input files can be downloaded from the [ftp webpage](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/) for any gemone of interest, which is linked on its [assembly page](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/).

For an unannotated nucleotide assembly, run NCBI's [Prokaryotic Genome Annotation Pipeline (PGAP)](https://github.com/ncbi/pgap) 
or [prokka](https://github.com/tseemann/prokka) and pass its `*.gff` output directly via `--gff`.
