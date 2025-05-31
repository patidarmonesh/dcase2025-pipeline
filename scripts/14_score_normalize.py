import os
import argparse
import yaml
import numpy as np
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Normalize AE and GMM scores and compute final anomaly scores.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    output_base = os.path.join(processed_root, "final_scores")
    input_base = os.path.join(processed_root, "pca_128")
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']

    # STEP 1: Compute normalization parameters from train data
    train_s_AE, train_s_GMM = [], []
    for machine in machine_types:
        train_dir = os.path.join(input_base, machine, "train")
        ae_path = os.path.join(train_dir, "s_AE.npy")
        gmm_path = os.path.join(train_dir, "s_GMM.npy")
        if os.path.exists(ae_path):
            train_s_AE.append(np.load(ae_path))
        if os.path.exists(gmm_path):
            train_s_GMM.append(np.load(gmm_path))

    if not train_s_AE or not train_s_GMM:
        print("No AE or GMM training scores found!")
        return

    global_min_AE = np.min(np.concatenate(train_s_AE))
    global_max_AE = np.max(np.concatenate(train_s_AE))
    global_min_GMM = np.min(np.concatenate(train_s_GMM))
    global_max_GMM = np.max(np.concatenate(train_s_GMM))

    print(f"AE:  min={global_min_AE:.4f}, max={global_max_AE:.4f}")
    print(f"GMM: min={global_min_GMM:.4f}, max={global_max_GMM:.4f}")

    # STEP 2: Normalize and average all splits
    for machine in tqdm(machine_types, desc="Machines"):
        for split in splits:
            current_dir = os.path.join(input_base, machine, split)
            output_dir = os.path.join(output_base, machine, split)
            os.makedirs(output_dir, exist_ok=True)
            ae_path = os.path.join(current_dir, "s_AE.npy")
            gmm_path = os.path.join(current_dir, "s_GMM.npy")
            if not (os.path.exists(ae_path) and os.path.exists(gmm_path)):
                continue
            s_AE = np.load(ae_path)
            s_GMM = np.load(gmm_path)
            if s_AE.shape != s_GMM.shape:
                print(f"[SKIP] Shape mismatch in {current_dir}: {s_AE.shape} vs {s_GMM.shape}")
                continue
            # Min-max normalization
            norm_AE = (s_AE - global_min_AE) / (global_max_AE - global_min_AE + 1e-8)
            norm_GMM = (s_GMM - global_min_GMM) / (global_max_GMM - global_min_GMM + 1e-8)
            norm_AE = np.clip(norm_AE, 0, 1)
            norm_GMM = np.clip(norm_GMM, 0, 1)
            s_tilde_A = 0.5 * (norm_AE + norm_GMM)
            np.save(os.path.join(output_dir, "norm_AE.npy"), norm_AE)
            np.save(os.path.join(output_dir, "norm_GMM.npy"), norm_GMM)
            np.save(os.path.join(output_dir, "s_tilde_A.npy"), s_tilde_A)
            print(f"[SAVED] {output_dir}/s_tilde_A.npy, shape: {s_tilde_A.shape}")

    print("✅ Normalization and averaging complete!")

if __name__ == "__main__":
    main()
