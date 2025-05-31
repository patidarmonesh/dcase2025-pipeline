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
    parser = argparse.ArgumentParser(description="Extract Wav2Vec2 embeddings for all audio segments.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_wav2vec2_embedding(model, feature_extractor, wav, sr, device):
    if sr != 16000:
        wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), orig_freq=sr, new_freq=16000).squeeze(0).numpy()
    inputs = feature_extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

    for machine in tqdm(machine_types, desc="Machines"):
        for split in splits:
            seg_dir = os.path.join(processed_root, machine, split, "raw_segments")
            if not os.path.isdir(seg_dir):
                continue
            save_dir = os.path.join(processed_root, machine, split)
            os.makedirs(save_dir, exist_ok=True)
            embeddings = []
            for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}", leave=False):
                if not fname.endswith(".wav"):
                    continue
                wav_path = os.path.join(seg_dir, fname)
                wav, sr = sf.read(wav_path)
                if len(wav) < 16000:
                    continue
                emb = extract_wav2vec2_embedding(model, feature_extractor, wav, sr, device)
                embeddings.append(emb)
            if embeddings:
                arr = np.stack(embeddings, axis=0)
                out_path = os.path.join(save_dir, "wav2vec2_embeddings.pickle")
                with open(out_path, "wb") as f:
                    pickle.dump(arr, f)
                print(f"[SAVED] {out_path}: {arr.shape}")

if __name__ == "__main__":
    main()
