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
    parser = argparse.ArgumentParser(description="Extract CLAP embeddings for all audio segments.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_clap_embedding(model, wav, sr, device):
    # Resample to 48kHz mono float32
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 48000:
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)
        wav_tensor = torchaudio.functional.resample(wav_tensor, orig_freq=sr, new_freq=48000)
        wav = wav_tensor.squeeze(0).cpu().numpy()
    wav = wav.astype(np.float32)
    # Write to a temp file (required by laion-clap)
    tmp_path = "clap_temp.wav"
    sf.write(tmp_path, wav, 48000, subtype="PCM_16")
    embed = model.get_audio_embedding_from_filelist(x=[tmp_path], use_tensor=False)
    os.remove(tmp_path)
    return embed[0]

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from laion_clap import CLAP_Module
    model = CLAP_Module(enable_fusion=False, device=device)
    model.load_ckpt()

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
                wav, sr = sf.read(wav_path)
                if len(wav) < 16000:
                    continue
                emb = extract_clap_embedding(model, wav, sr, device)
                embeddings.append(emb)
            if embeddings:
                arr = np.stack(embeddings, axis=0)
                out_path = os.path.join(save_dir, "clap_embeddings.pickle")
                with open(out_path, "wb") as f:
                    pickle.dump(arr, f)
                print(f"[SAVED] {out_path}: {arr.shape}")

if __name__ == "__main__":
    main()
