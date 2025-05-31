import os
import argparse
import yaml
import librosa
import soundfile as sf
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Segment and resample audio files.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def segment_and_resample(raw_root, processed_root, sr, segment_duration, machine_types, splits):
    seg_len = int(sr * segment_duration)
    for machine in tqdm(machine_types, desc="Machines"):
        machine_raw = os.path.join(raw_root, machine)
        if not os.path.isdir(machine_raw):
            continue
        for split in splits:
            split_in = os.path.join(machine_raw, split)
            if not os.path.isdir(split_in):
                continue
            split_out = os.path.join(processed_root, machine, split, 'raw_segments')
            os.makedirs(split_out, exist_ok=True)
            for fname in tqdm(os.listdir(split_in), desc=f"{machine}/{split}", leave=False):
                if not fname.lower().endswith('.wav'):
                    continue
                path = os.path.join(split_in, fname)
                audio, orig_sr = librosa.load(path, sr=None)
                # Resample if needed
                if orig_sr != sr:
                    audio = librosa.resample(audio, orig_sr, sr)
                # Split into fixed-length segments
                total = len(audio)
                n_segs = total // seg_len
                for idx in range(n_segs):
                    start = idx * seg_len
                    end = start + seg_len
                    segment = audio[start:end]
                    out_fname = f"{os.path.splitext(fname)[0]}_seg{idx:02d}.wav"
                    out_path = os.path.join(split_out, out_fname)
                    sf.write(out_path, segment, sr)
    print("✅ All files segmented and resampled.")

def main():
    args = parse_args()
    config = load_config(args.config)
    raw_root = config['paths']['raw_data']
    processed_root = config['paths']['processed_data']
    sr = config['audio']['sample_rate']
    segment_duration = config['audio']['segment_duration']
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']
    segment_and_resample(raw_root, processed_root, sr, segment_duration, machine_types, splits)

if __name__ == "__main__":
    main()
