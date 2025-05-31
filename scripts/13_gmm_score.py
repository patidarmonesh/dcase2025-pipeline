import os
import argparse
import yaml
import numpy as np
import pickle
from tqdm.auto import tqdm
from sklearn.mixture import GaussianMixture

def parse_args():
    parser = argparse.ArgumentParser(description="Fit GMM on z_q and compute GMM anomaly scores for all splits.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    checkpoint_dir = config['paths']['checkpoints']
    input_base = os.path.join(processed_root, "pca_128")
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']

    # STEP 1: Gather all train z_q embeddings
    zq_train_all = []
    for machine in machine_types:
        train_path = os.path.join(input_base, machine, "train", "z_q.npy")
        if not os.path.exists(train_path):
            print(f"[SKIP] {machine}/train/z_q.npy not found")
            continue
        z_q = np.load(train_path)
        if z_q.size == 0:
            print(f"[WARN] {machine}/train/z_q.npy is empty")
            continue
        zq_train_all.append(z_q)
        print(f"[LOADED] {machine}/train: {z_q.shape}")
    if not zq_train_all:
        raise ValueError("No training data found for GMM fitting.")

    zq_all = np.concatenate(zq_train_all, axis=0)
    print(f"Total training data shape: {zq_all.shape}")

    # STEP 2: Fit GMM
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
    gmm.fit(zq_all)
    print("✅ GMM fitted.")

    # Save GMM checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "branch_a_gmm.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(gmm, f)
    print(f"✅ GMM saved to {checkpoint_path}")

    # STEP 3: Apply to all splits and save s_GMM
    for machine in tqdm(machine_types, desc="Machines"):
        for split in splits:
            zq_file = os.path.join(input_base, machine, split, "z_q.npy")
            if not os.path.exists(zq_file):
                continue
            z_q = np.load(zq_file)
            if z_q.size == 0:
                continue
            s_gmm = -gmm.score_samples(z_q)
            out_path = os.path.join(input_base, machine, split, "s_GMM.npy")
            np.save(out_path, s_gmm)
            print(f"[SAVED] {machine}/{split} → s_GMM.npy shape: {s_gmm.shape}")

    print("✅ All GMM anomaly scores computed and saved!")

if __name__ == "__main__":
    main()
