import requests
import shutil
import os
import gzip


def download_model_weights():
    url = "https://zenodo.org/records/16937977/files/beaker_v3.pkl.gz?download=1"
    gz_path = os.path.join(os.path.dirname(__file__), "beaker_v3.pkl.gz")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(gz_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    with gzip.open(gz_path, 'rb') as gz_file:
        with open(os.path.join(os.path.dirname(__file__), "beaker_v3.pkl"), 'wb') as out_file:
            shutil.copyfileobj(gz_file, out_file)
    os.remove(gz_path)


def download_esm2_contact_weights():
    url = 'https://zenodo.org/records/16937977/files/esm2_t30_150M_UR50D-contact-regression.pt.gz?download=1'
    gz_path = os.path.join(os.path.dirname(__file__), 'esm2_t30_150M_UR50D-contact-regression.pt.gz')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(gz_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    with gzip.open(gz_path, 'rb') as gz_file:
        with open(os.path.join(os.path.dirname(__file__), 'esm2_t30_150M_UR50D-contact-regression.pt'), 'wb') as out_file:
            shutil.copyfileobj(gz_file, out_file)
    os.remove(gz_path)


def download_esm2_weights():
    url = 'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt'
    path = os.path.join(os.path.dirname(__file__), 'esm2_t30_150M_UR50D.pt')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def download_weights():
    base_dir = os.path.dirname(__file__)
    downloaded_weights = os.listdir(base_dir)
    if "beaker_v3.pkl" not in downloaded_weights:
        print("Downloading model weights")
        download_model_weights()
    if "esm2_t30_150M_UR50D-contact-regression.pt" not in downloaded_weights:
        download_esm2_contact_weights()
    if "esm2_t30_150M_UR50D.pt" not in downloaded_weights:
        print("Downloading ESM2 weights")
        download_esm2_weights()


if __name__ == "__main__":
    download_weights()
    
