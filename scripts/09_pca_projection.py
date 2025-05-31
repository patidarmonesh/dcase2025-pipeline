import os
import argparse
import yaml
import pickle
import numpy as np
from tqdm.auto import tqdm
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser(description="Apply PCA to MPEF embeddings and save projected vectors.")
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
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']

    # 1. Collect all mpef_embeddings
    embeddings_list = []
    file_paths = []
    for machine in tqdm(machine_types, desc="Machines"):
        for split in splits:
            path = os.path.join(processed_root, machine, split, "mpef_embeddings.pickle")
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                vals = list(data.values())
                embeddings = np.stack(vals)
            else:
                embeddings = np.array(data)
            embeddings_list.append(embeddings)
            file_paths.append(path)

    if not embeddings_list:
        print("No mpef_embeddings found!")
        return

    # 2. Concatenate and fit PCA
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    n_samples, n_features = all_embeddings.shape
    n_components = min(128, n_samples, n_features)
    mean_vector = np.mean(all_embeddings, axis=0)
    centered = all_embeddings - mean_vector
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=False)
    pca.fit(centered)
    components = pca.components_

    # 3. Save PCA parameters
    os.makedirs(checkpoint_dir, exist_ok=True)
    pca_params_path = os.path.join(checkpoint_dir, "pca_params.pkl")
    with open(pca_params_path, "wb") as f:
        pickle.dump({"mean": mean_vector, "components": components}, f)

    # 4. Project each file's embeddings and save 128-D vectors
    output_base = os.path.join(processed_root, "pca_128")
    for path, embeddings in zip(file_paths, embeddings_list):
        emb_array = np.array(embeddings)
        emb_centered = emb_array - mean_vector
        z_pca = np.dot(emb_centered, components.T)
        rel_path = os.path.relpath(path, processed_root)
        save_dir = os.path.join(output_base, os.path.dirname(rel_path))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "z_pca.npy")
        np.save(save_path, z_pca)
        print(f"[SAVED] {save_path}: {z_pca.shape}")

    print("✅ PCA projection complete. Parameters and 128-D vectors saved.")

if __name__ == "__main__":
    main()
