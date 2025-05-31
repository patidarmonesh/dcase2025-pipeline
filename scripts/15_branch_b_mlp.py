import os
import argparse
import yaml
import pickle
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate MLP embeddings from multi-resolution Mel spectrograms.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class MultiResMLPHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_64ms = self._build_mlp(4096)
        self.head_256ms = self._build_mlp(1024)
        self.head_1000ms = self._build_mlp(384)
        
    def _build_mlp(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
    def forward(self, x_64ms, x_256ms, x_1000ms):
        h64 = self.head_64ms(x_64ms)
        h256 = self.head_256ms(x_256ms)
        h1000 = self.head_1000ms(x_1000ms)
        return h64, h256, h1000

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    output_base = os.path.join(processed_root, "branch_b_embeddings")
    checkpoint_dir = config['paths']['checkpoints']
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiResMLPHeads().to(device)
    model.eval()

    for machine in tqdm(machine_types, desc="Machines"):
        for split in splits:
            mel_file = os.path.join(processed_root, machine, split, "mels_multires.pickle")
            if not os.path.exists(mel_file):
                continue
            with open(mel_file, "rb") as f:
                mel_data = pickle.load(f)
            embeddings = {}
            for clip_id, specs in tqdm(mel_data.items(), desc=f"{machine}/{split}", leave=False):
                x_64ms = torch.tensor(specs["64ms"].flatten(), dtype=torch.float32).unsqueeze(0).to(device)
                x_256ms = torch.tensor(specs["256ms"].flatten(), dtype=torch.float32).unsqueeze(0).to(device)
                x_1000ms = torch.tensor(specs["1000ms"].flatten(), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    h64, h256, h1000 = model(x_64ms, x_256ms, x_1000ms)
                embeddings[clip_id] = {
                    "h64": h64.cpu().numpy().squeeze(0),
                    "h256": h256.cpu().numpy().squeeze(0),
                    "h1000": h1000.cpu().numpy().squeeze(0)
                }
            output_dir = os.path.join(output_base, machine, split)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "mlp_embeddings.pickle"), "wb") as f:
                pickle.dump(embeddings, f)

    # Save model checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "branch_b_mlp_heads.pth"))

if __name__ == "__main__":
    main()
