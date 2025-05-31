import os
import argparse
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE on PCA-projected embeddings.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- VQ-VAE Model ---
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

    # Load all z_pca.npy files
    all_zpca = []
    for root, _, files in os.walk(input_base):
        for file in files:
            if file == "z_pca.npy":
                arr = np.load(os.path.join(root, file))
                all_zpca.append(arr)
    if not all_zpca:
        print("No z_pca.npy files found!")
        return
    z_pca = torch.tensor(np.concatenate(all_zpca, axis=0), dtype=torch.float32)

    # Train VQ-VAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = DataLoader(TensorDataset(z_pca), batch_size=64, shuffle=True)
    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        total_loss = 0
        for batch, in dataset:
            optimizer.zero_grad()
            z_hat, z_q, loss = model(batch.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    # Save outputs
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "branch_a_vqvae_model.pth"))
    with open(os.path.join(checkpoint_dir, "branch_a_codebook.pkl"), "wb") as f:
        pickle.dump(model.quantizer.codebook.detach().cpu().numpy(), f)

    print("✅ VQ-VAE model + codebook saved.")

if __name__ == "__main__":
    main()
