import os
import argparse
import yaml
import pickle
import torch
import torchaudio
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Extract PANNs (CNN14) embeddings for all audio segments.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_panns_embedding(at, wav, sr):
    # Resample if needed (handles 1s/16kHz -> 1s/32kHz)
    if sr != 32000:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 32000)
        wav = wav_tensor.squeeze(0).cpu().numpy()
    audio = wav[None, :]
    _, embedding = at.inference(audio)
    return embedding.squeeze(0)  # (2048,)

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from panns_inference import AudioTagging
    at = AudioTagging(checkpoint_path=None, device=device)  # Uses default Cnn14 checkpoint

    for machine in tqdm(machine_types, desc="Machines"):
        for split in splits:
            seg_dir = os.path.join(processed_root, machine, split, "raw_segments")
            if not os.path.isdir(seg_dir):
                continue
            save_dir = os.path.join(processed_root, machine, split)
            os.makedirs(save_dir, exist_ok=True)
            embeddings = []
            for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}", leave=False):
                if not fname.lower().endswith('.wav'):
                    continue
                wav_path = os.path.join(seg_dir, fname)
                wav, sr = sf.read(wav_path)
                emb = extract_panns_embedding(at, wav, sr)
                embeddings.append(emb)
            if embeddings:
                arr = np.stack(embeddings, axis=0)
                out_path = os.path.join(save_dir, "panns_embeddings.pickle")
                with open(out_path, "wb") as f:
                    pickle.dump(arr, f)
                print(f"[SAVED] {out_path}: {arr.shape}")

if __name__ == "__main__":
    main()

