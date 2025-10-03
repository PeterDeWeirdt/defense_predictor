import requests
import shutil
import os
import gzip


def download_file_weights(file, url):
    out_path = os.path.join(os.path.dirname(__file__), file)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(out_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    except:
        print(f"Could not download {file}, you can try manually downloading {file} from {url} and placing it in this directory {out_path}")


def download_esm2_weights():
    try:
        url = 'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt'
        path = os.path.join(os.path.dirname(__file__), 'esm2_t30_150M_UR50D.pt')
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    except:
        file = 'esm2_t30_150M_UR50D.pt'
        url = 'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt'
        out_path = os.path.dirname(__file__)
        print(f"Could not download {file}, you can try manually downloading {file} from {url} and placing it in this directory {out_path}")
            

def download_weights():
    download_dict = {'beaker_fold_0.pkl': 'https://zenodo.org/records/17258204/files/beaker_fold_0.pkl?download=1',
                     'beaker_fold_1.pkl': 'https://zenodo.org/records/17258204/files/beaker_fold_1.pkl?download=1',
                     'beaker_fold_2.pkl': 'https://zenodo.org/records/17258204/files/beaker_fold_2.pkl?download=1',
                     'beaker_fold_3.pkl': 'https://zenodo.org/records/17258204/files/beaker_fold_3.pkl?download=1',
                     'beaker_fold_4.pkl': 'https://zenodo.org/records/17258204/files/beaker_fold_4.pkl?download=1',
                     'esm2_t30_150M_UR50D-contact-regression.pt': 'https://zenodo.org/records/17258204/files/esm2_t30_150M_UR50D-contact-regression.pt?download=1'}
    required_files = list(download_dict.keys())
    base_dir = os.path.dirname(__file__)
    base_dir_files = os.listdir(base_dir)
    missing_weights = [x for x in required_files if x not in base_dir_files]
    if len(missing_weights) == 0:
        print("All model weights already downloaded")
    else:
        print('Downloading model weights...')
        for k in missing_weights:
            print(f"Downloading {k}")
            download_file_weights(k, download_dict[k])
    if 'esm2_t30_150M_UR50D.pt' not in base_dir_files:
        print('Downloading ESM2 weights')
        download_esm2_weights()
    else:
        print('ESM2 weights already downloaded')     


if __name__ == "__main__":
    download_weights()
    
