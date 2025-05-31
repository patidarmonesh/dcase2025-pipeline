import os
import argparse
import yaml
import pickle
import numpy as np
from tqdm.auto import tqdm
from sklearn.mixture import GaussianMixture

def parse_args():
    parser = argparse.ArgumentParser(description="Fit GMM on branch B fused embeddings and compute normalized anomaly scores.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    output_base = os.path.join(processed_root, "final_scores_branch_b")
    input_base = os.path.join(processed_root, "branch_b_fused")
    checkpoint_dir = config['paths']['checkpoints']
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']

    # STEP 1: Gather all train z_B embeddings
    zB_train_all = []
    for machine in machine_types:
        train_path = os.path.join(input_base, machine, "train", "z_B.pickle")
        if not os.path.exists(train_path):
            print(f"[SKIP] {machine}/train/z_B.pickle not found")
            continue
        with open(train_path, "rb") as f:
            z_B = pickle.load(f)
        if not z_B:
            print(f"[WARN] {machine}/train/z_B.pickle is empty")
            continue
        embeddings = np.stack(list(z_B.values()))
        zB_train_all.append(embeddings)
        print(f"[LOADED] {machine}/train: {embeddings.shape}")

    if not zB_train_all:
        raise ValueError("No training data found for GMM fitting.")

    zB_all = np.concatenate(zB_train_all, axis=0)
    print(f"Total training data shape: {zB_all.shape}")

    # STEP 2: Fit GMM
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
    gmm.fit(zB_all)
    print("✅ GMM fitted.")

    # Save GMM checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "branch_b_gmm.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(gmm, f)
    print(f"✅ GMM saved to {checkpoint_path}")

    # STEP 3: Compute train scores for normalization
    train_scores = -gmm.score_samples(zB_all)
    min_score, max_score = train_scores.min(), train_scores.max()
    print(f"Score normalization range: min={min_score:.4f}, max={max_score:.4f}")

    # STEP 4: Score and normalize all splits
    for machine in machine_types:
        for split in splits:
            zB_file = os.path.join(input_base, machine, split, "z_B.pickle")
            if not os.path.exists(zB_file):
                continue
            with open(zB_file, "rb") as f:
                z_B = pickle.load(f)
            if not z_B:
                continue
            embs = np.stack(list(z_B.values()))
            raw_scores = -gmm.score_samples(embs)
            norm_scores = (raw_scores - min_score) / (max_score - min_score + 1e-8)
            norm_scores = np.clip(norm_scores, 0, 1)
            # Save as dict {clip_id: score}
            sB_dict = {clip_id: float(score) for clip_id, score in zip(z_B.keys(), norm_scores)}
            out_dir = os.path.join(output_base, machine, split)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "s_tilde_B.pickle"), "wb") as f_out:
                pickle.dump(sB_dict, f_out)
            print(f"[{machine}/{split}] Saved {len(sB_dict)} normalized scores.")

    print("✅ All Branch B anomaly scores computed and saved!")

if __name__ == "__main__":
    main()
