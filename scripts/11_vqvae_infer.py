import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="VQ-VAE inference: reconstruct and quantize PCA embeddings.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- VQ-VAE Model (same as in training) ---
class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=128, code_dim=32, beta=0.25):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))
        self.beta = beta

    def forward(self, z):
        dist = torch.cdist(z, self.codebook)
        indices = torch.argmin(dist, dim=1)
        z_q = self.codebook[indices]
        codebook_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss
        z_q_st = z + (z_q - z).detach()
        return z_q_st, vq_loss, indices, z_q

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 32)
        )
        self.quantizer = VectorQuantizer()
        self.decoder = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        z_latent = self.encoder(x)
        z_q, vq_loss, _, _ = self.quantizer(z_latent)
        z_hat = self.decoder(z_q)
        rec_loss = F.mse_loss(z_hat, x)
        total_loss = rec_loss + vq_loss
        return z_hat, z_q, total_loss

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    checkpoint_dir = config['paths']['checkpoints']
    input_base = os.path.join(processed_root, "pca_128")
    model_path = os.path.join(checkpoint_dir, "branch_a_vqvae_model.pth")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VQVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Loop over PCA dirs and process
    for root, _, files in os.walk(input_base):
        for file in files:
            if file == "z_pca.npy":
                z_path = os.path.join(root, file)
                z_pca = torch.tensor(np.load(z_path), dtype=torch.float32).to(device)
                with torch.no_grad():
                    z_hat, z_q, _ = model(z_pca)
                # Save
                np.save(os.path.join(root, "z_hat.npy"), z_hat.cpu().numpy())
                np.save(os.path.join(root, "z_q.npy"), z_q.cpu().numpy())
                print(f"[DONE] {z_path}")

if __name__ == "__main__":
    main()
