import os
import argparse
import yaml
import numpy as np
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Compute squared L2 AE reconstruction error (anomaly score) for each clip.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    input_base = os.path.join(processed_root, "pca_128")
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']

    for machine in tqdm(machine_types, desc="Machines"):
        for split in splits:
            root = os.path.join(input_base, machine, split)
            z_pca_path = os.path.join(root, "z_pca.npy")
            z_hat_path = os.path.join(root, "z_hat.npy")
            if not (os.path.isfile(z_pca_path) and os.path.isfile(z_hat_path)):
                continue
            z_pca = np.load(z_pca_path)
            z_hat = np.load(z_hat_path)
            if z_pca.shape != z_hat.shape:
                print(f"[SKIP] Shape mismatch in {root}: {z_pca.shape} vs {z_hat.shape}")
                continue
            s_AE = np.sum((z_pca - z_hat) ** 2, axis=1)
            np.save(os.path.join(root, "s_AE.npy"), s_AE)
            print(f"[DONE] {root} → s_AE.npy saved, shape: {s_AE.shape}")

if __name__ == "__main__":
    main()
