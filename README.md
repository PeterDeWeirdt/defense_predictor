# DefensePredictor: A Machine Learning Model to Discover Novel Prokaryotic Immune Systems

Python package to run DefensePredictor, a machine-learning model that leverages embeddings from a protein language model, [ESM2](https://github.com/facebookresearch/esm), to classify proteins as anti-phage defensive.

### Installation

In a fresh [conda](https://anaconda.org/anaconda/conda) or other virutal environment, run:

```bash
pip install defense_predictor
defense_predictor_download
```

The first command downloads the python package and the second command downloads the model weights. Once model weights are downloaded you do not need to run this command again. 

### Requirements 

Requires `python >= 3.10`

### Usage

`defense_predictor` can be run as python code

```python
import defense_predictor as dfp

ncbi_feature_table = 'GCF_003333385.1_ASM333338v1_feature_table.txt'
ncbi_cds_from_genomic = 'GCF_003333385.1_ASM333338v1_cds_from_genomic.fna'
ncbi_protein_fasta = 'GCF_003333385.1_ASM333338v1_protein.faa'
output_df = dfp.run_defense_predictor(ncbi_feature_table=ncbi_feature_table,
                                      ncbi_cds_from_genomic=ncbi_cds_from_genomic,
                                      ncbi_protein_fasta=ncbi_protein_fasta)
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

<br>

`defense_predictor` outputs the predicted probability and log-odds of defense for each input protein. We reccomend using a stringent log-odds cutoff of `7.2` to call a protein predicted defensive.

To see an example you can run the `defense_predictor_example.ipynb` in colab: <a href="https://colab.research.google.com/github/PeterDeWeirdt/defense_predictor/blob/main/defense_predictor_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> 

We reccomend running `defense_predictor` on a computer with a cuda-enabled GPU, to maximize computational efficiency. 

### Inputs

Input files can be downloaded from the [ftp webpage](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/) for any gemone of interest, which is linked on its [assembly page](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/). Input files can be generated from an unannotated nuceotide assembly using NCBI's [Prokaryotic Genome Annotation Pipeline](https://github.com/ncbi/pgap). 

Alternatively, `defense_predictor` accepts inputs generated from [prokka](https://github.com/tseemann/prokka) using the arguments `prokka_gff`, `prokka_ffn`, and `prokka_faa`.
