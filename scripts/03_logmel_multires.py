import os
import argparse
import yaml
import torch
import torchaudio
import pickle
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Compute multi-resolution log-Mel spectrograms.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_multi_res_mel(seg_root, out_root, sr, machine_types, splits):
    configs = [
        {'name': '64ms', 'n_fft': 1024, 'hop': 512},
        {'name': '256ms', 'n_fft': 4096, 'hop': 2048},
        {'name': '1000ms', 'n_fft': 16000, 'hop': 8000}
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Prepare transforms
    transforms = {}
    for cfg in configs:
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=cfg['n_fft'],
            hop_length=cfg['hop'],
            n_mels=128,
            power=2.0
        ).to(device)
        db_transform = torchaudio.transforms.AmplitudeToDB(
            stype='power',
            top_db=80.0
        ).to(device)
        transforms[cfg['name']] = (mel_spec, db_transform)

    for machine in tqdm(machine_types, desc="Machines"):
        for split in splits:
            seg_dir = os.path.join(seg_root, machine, split, 'raw_segments')
            if not os.path.isdir(seg_dir):
                continue
            save_dir = os.path.join(out_root, machine, split)
            os.makedirs(save_dir, exist_ok=True)
            output_data = {}
            for fname in tqdm(os.listdir(seg_dir), desc=f"{machine}/{split}", leave=False):
                if not fname.endswith('.wav'):
                    continue
                wav_path = os.path.join(seg_dir, fname)
                waveform, orig_sr = torchaudio.load(wav_path)
                if orig_sr != sr:
                    waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
                waveform = waveform.to(device)
                clip_mels = {}
                for res_name, (mel_spec, db_transform) in transforms.items():
                    mels = mel_spec(waveform)
                    log_mels = db_transform(mels)
                    arr = log_mels.squeeze(0).transpose(0, 1).cpu().numpy()
                    clip_mels[res_name] = arr
                clip_id = os.path.splitext(fname)[0]
                output_data[clip_id] = clip_mels
            # Save all spectrograms for this split
            if output_data:
                out_path = os.path.join(save_dir, 'mels_multires.pickle')
                with open(out_path, 'wb') as f:
                    pickle.dump(output_data, f)
                print(f"Saved {len(output_data)} clips to {out_path}")

def main():
    args = parse_args()
    config = load_config(args.config)
    seg_root = config['paths']['processed_data']
    out_root = config['paths']['processed_data']
    sr = config['audio']['sample_rate']
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']
    compute_multi_res_mel(seg_root, out_root, sr, machine_types, splits)

if __name__ == "__main__":
    main()
