import os
import argparse
import yaml
import librosa
import numpy as np
import pickle
from pathlib import Path
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Extract 6 Mel spectrogram patches per audio segment.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_audio_clip(y, sr, frame_length, n_frames, n_mels, target_time_steps):
    clip_length = 1.0
    hop_length = (clip_length - frame_length) / (n_frames - 1)
    starts = np.linspace(0, clip_length - frame_length, num=n_frames)
    mel_patches = []
    for start in starts:
        end = start + frame_length
        sample_start = int(start * sr)
        sample_end = int(end * sr)
        frame = y[sample_start:sample_end]
        S = librosa.feature.melspectrogram(
            y=frame,
            sr=sr,
            n_fft=512,
            hop_length=57,
            n_mels=n_mels,
            fmin=20,
            fmax=8000
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        if S_db.shape[1] < target_time_steps:
            pad_width = target_time_steps - S_db.shape[1]
            S_db = np.pad(S_db, ((0,0), (0,pad_width)), mode='constant')
        else:
            S_db = S_db[:, :target_time_steps]
        mel_patches.append(S_db.T)
    return np.array(mel_patches)

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    output_base = os.path.join(processed_root, "mel_patches")
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']
    sr = config['audio']['sample_rate']
    frame_length = config['audio']['frame_length']
    n_frames = config['audio']['n_frames']
    n_mels = config['audio']['n_mels']
    target_time_steps = config['audio']['target_time_steps']

    for machine in tqdm(machine_types, desc="Machines"):
        for split in splits:
            audio_dir = Path(processed_root) / machine / split / "raw_segments"
            if not audio_dir.exists():
                continue
            output_dir = Path(output_base) / machine / split
            output_dir.mkdir(parents=True, exist_ok=True)
            all_patches = {}
            for audio_file in tqdm(list(audio_dir.glob("*.wav")), desc=f"{machine}/{split}", leave=False):
                y, _sr = librosa.load(audio_file, sr=sr)
                patches = process_audio_clip(y, sr, frame_length, n_frames, n_mels, target_time_steps)
                all_patches[audio_file.stem] = patches.astype(np.float32)
            with open(output_dir / "mel_patches.pickle", "wb") as f:
                pickle.dump(all_patches, f)
    print("✅ All Mel patches processed and saved!")

if __name__ == "__main__":
    main()
