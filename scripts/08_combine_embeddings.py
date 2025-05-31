import os
import argparse
import yaml
import pickle
import numpy as np
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Combine all embeddings into a single MPEF embedding.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']

    for machine in tqdm(machine_types, desc="Machines"):
        for split in splits:
            base_dir = os.path.join(processed_root, machine, split)
            paths = {
                "panns":    os.path.join(base_dir, "panns_embeddings.pickle"),
                "wav2vec":  os.path.join(base_dir, "wav2vec2_embeddings.pickle"),
                "beats":    os.path.join(base_dir, "beats_embeddings.pickle"),
                "clap":     os.path.join(base_dir, "clap_embeddings.pickle"),
            }
            if not all(os.path.isfile(p) for p in paths.values()):
                print(f"[SKIP] Missing embeddings for {machine}/{split}")
                continue
            arrs = []
            for key, p in paths.items():
                with open(p, "rb") as f:
                    emb = pickle.load(f)
                arrs.append(emb)
            n_clips = arrs[0].shape[0]
            if any(a.shape[0] != n_clips for a in arrs[1:]):
                print(f"[ERROR] Mismatch in clip count for {machine}/{split}")
                continue
            mpef_emb = np.concatenate(arrs, axis=1)
            out_path = os.path.join(base_dir, "mpef_embeddings.pickle")
            with open(out_path, "wb") as f:
                pickle.dump(mpef_emb, f)
            print(f"[SAVED] {machine}/{split} → mpef_embeddings: {mpef_emb.shape}")

if __name__ == "__main__":
    main()
