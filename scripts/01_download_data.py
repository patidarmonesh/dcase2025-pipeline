import os
import argparse
import yaml
import urllib.request
import zipfile
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Download DCASE 2025 Task 2 data.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_and_extract(url, dest_path, extract_to):
    if not os.path.exists(dest_path):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest_path)
    else:
        print(f"File {dest_path} already exists, skipping download.")
    print(f"Extracting {dest_path} ...")
    with zipfile.ZipFile(dest_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(dest_path)

def main():
    args = parse_args()
    config = load_config(args.config)
    raw_data_dir = config['paths']['raw_data']
    os.makedirs(raw_data_dir, exist_ok=True)

    machine_types = config['dataset']['machine_types']
    # You may want to update these URLs for your dataset
    base_url = "https://zenodo.org/records/15097779/files/dev_{}.zip"

    for machine in tqdm(machine_types, desc="Machines"):
        zip_url = base_url.format(machine)
        zip_path = os.path.join(raw_data_dir, f"dev_{machine}.zip")
        download_and_extract(zip_url, zip_path, raw_data_dir)

    print("✅ All files downloaded and extracted.")

if __name__ == "__main__":
    main()

