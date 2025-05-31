import os
import argparse
import yaml
import pickle
import torch
import torch.nn as nn
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Fuse multi-resolution MLP embeddings using attention.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config YAML')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.W = nn.Linear(128, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=True)
        
    def forward(self, h64, h256, h1000):
        h_all = torch.stack([h64, h256, h1000], dim=0)
        u = self.v(torch.tanh(self.W(h_all)))  # (3, 1)
        alpha = torch.softmax(u, dim=0)        # (3, 1)
        z_B = torch.sum(alpha * h_all, dim=0)  # (128,)
        return z_B

def main():
    args = parse_args()
    config = load_config(args.config)
    processed_root = config['paths']['processed_data']
    input_base = os.path.join(processed_root, "branch_b_embeddings")
    output_base = os.path.join(processed_root, "branch_b_fused")
    checkpoint_dir = config['paths']['checkpoints']
    machine_types = config['dataset']['machine_types']
    splits = config['dataset']['splits']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentionFusion().to(device)
    model.eval()

    for machine in tqdm(machine_types, desc="Machines"):
        for split in splits:
            emb_file = os.path.join(input_base, machine, split, "mlp_embeddings.pickle")
            if not os.path.exists(emb_file):
                continue
            with open(emb_file, "rb") as f:
                emb_data = pickle.load(f)
            fused_embeddings = {}
            for clip_id, embs in tqdm(emb_data.items(), desc=f"{machine}/{split}", leave=False):
                h64 = torch.tensor(embs["h64"], dtype=torch.float32).to(device)
                h256 = torch.tensor(embs["h256"], dtype=torch.float32).to(device)
                h1000 = torch.tensor(embs["h1000"], dtype=torch.float32).to(device)
                with torch.no_grad():
                    z_B = model(h64, h256, h1000)
                fused_embeddings[clip_id] = z_B.cpu().numpy()
            output_dir = os.path.join(output_base, machine, split)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "z_B.pickle"), "wb") as f:
                pickle.dump(fused_embeddings, f)

    # Save model checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "branch_b_attention.pth"))

if __name__ == "__main__":
    main()
