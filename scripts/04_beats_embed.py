import os
import sys
import argparse
import yaml
import pickle
import torch
import torchaudio
import numpy as np
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Extract BEATs embeddings for all audio segments.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_beats_repo(checkpoint_path):
    # Clone BEATs repo if not present
    if not os.path.isdir("unilm"):
        os.system("git clone https://github.com/microsoft/unilm.git")
    sys.path.append(os.path.abspath("unilm/beats"))
    # Download checkpoint if not present
    if not os.path.exists(checkpoint_path):
        import urllib.request
        url = "https://zenodo.org/records/15097779/files/BEATs_iter3_plus_AS2M.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        urllib.request.urlretrieve(url, checkpoint_path)

def extract_beats_embedding(model, device, wav, sr):
    import torchaudio
    if sr != 16000:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
        wav = wav_tensor.squeeze(0).cpu().numpy()
    wav_tensor = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        hidden_states, _ = model.extract_features(wav_tensor, None)
        emb = hidden_states.mean(dim=1).cpu().numpy().squeeze(0)
    return emb

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    checkpoint_path = os.path.join(config['paths']['checkpoints'], "BEATs_iter3_plus_AS2M.pt")
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']

    setup_beats_repo(checkpoint_path)
    from BEATs import BEATs, BEATsConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for machine in tqdm(machine_types, desc="Machines"):
        for split in splits:
            seg_dir = os.path.join(processed_root, machine, split, "raw_segments")
            if not os.path.isdir(seg_dir):
                continue
            save_dir = os.path.join(processed_root, machine, split)
            os.makedirs(save_dir, exist_ok=True)
            embeddings = []
            for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}", leave=False):
                if not fname.lower().endswith(".wav"):
                    continue
                wav_path = os.path.join(seg_dir, fname)
                import soundfile as sf
                wav, sr = sf.read(wav_path)
                if len(wav) < 16000:
                    continue
                emb = extract_beats_embedding(model, device, wav, sr)
                embeddings.append(emb)
            if embeddings:
                arr = np.stack(embeddings, axis=0)
                out_path = os.path.join(save_dir, "beats_embeddings.pickle")
                with open(out_path, "wb") as f:
                    pickle.dump(arr, f)
                print(f"[SAVED] {out_path}: {arr.shape}")

if __name__ == "__main__":
    main()
