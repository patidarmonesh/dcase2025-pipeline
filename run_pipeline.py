import subprocess
import sys
import yaml
from pathlib import Path

# Load config (optional, but can be used to pass config to scripts)
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# List of scripts to run in order
SCRIPTS = [
    "scripts/01_download_data.py",
    "scripts/02_segment_resample.py",
    "scripts/03_logmel_multires.py",
    "scripts/04_beats_embed.py",
    "scripts/05_clap_embed.py",
    "scripts/06_panns_embed.py",
    "scripts/07_wav2vec2_embed.py",
    "scripts/08_combine_embeddings.py",
    "scripts/09_pca_projection.py",
    "scripts/10_vqvae_train.py",
    "scripts/11_vqvae_infer.py",
    "scripts/12_anomaly_score_ae.py",
    "scripts/13_gmm_score.py",
    "scripts/14_score_normalize.py",
    "scripts/15_branch_b_mlp.py",
    "scripts/16_branch_b_attention.py",
    "scripts/17_branch_b_gmm.py",
    "scripts/18_patch_mel.py",
    "scripts/19_branch_c_mfn.py"
    # Add more scripts here as needed
]

def main():
    for script in SCRIPTS:
        print(f"\n🚀 Running {script}")
        result = subprocess.run([sys.executable, script], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error in {script}:")
            print(result.stderr)
            sys.exit(1)
        else:
            print(result.stdout)

if __name__ == "__main__":
    main()
