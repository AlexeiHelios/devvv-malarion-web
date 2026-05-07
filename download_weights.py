import os
from huggingface_hub import hf_hub_download

REPO_ID = "devranasinghe/malarion-weights"
WEIGHTS_DIR = "weights"

files = [
    "best_malarion_v1.pt",
    "bv_resnet18_best.pth",
    "bv_resnet18_hn_best.pth",
    "c3_yolo_cbam_best.pt",
]

os.makedirs(WEIGHTS_DIR, exist_ok=True)

for f in files:
    print(f"Downloading {f}...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=f,
        local_dir=WEIGHTS_DIR,
    )
    print(f"✓ Done: {f}")