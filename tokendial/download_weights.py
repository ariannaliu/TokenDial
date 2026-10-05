"""
Download the model weights TokenDial *training* needs:

  - InternVideo2-Stage2 (appearance loss)  ->  weights/internvideo2/
  - DINOv2 ViT-L/14      (motion loss)      ->  torch.hub cache

Run once before training:

    python -m tokendial.download_weights

Inference and V2V do NOT need this — the base Wan2.1-T2V-1.3B weights download
automatically on first run.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERNVIDEO2_DIR = os.path.join(REPO_ROOT, "weights", "internvideo2")
INTERNVIDEO2_CKPT = os.path.join(INTERNVIDEO2_DIR, "InternVideo2-stage2_1b-224p-f4.pt")


def download_internvideo2():
    from huggingface_hub import snapshot_download
    os.makedirs(INTERNVIDEO2_DIR, exist_ok=True)
    print("Downloading OpenGVLab/InternVideo2-Stage2_1B-224p-f4 ...")
    snapshot_download(repo_id="OpenGVLab/InternVideo2-Stage2_1B-224p-f4", local_dir=INTERNVIDEO2_DIR)
    print(f"  -> {INTERNVIDEO2_CKPT}")
    if not os.path.isfile(INTERNVIDEO2_CKPT):
        print("  [warn] expected checkpoint file not found; check the repo contents.")


def download_dinov2():
    import torch
    print("Fetching DINOv2 ViT-L/14 via torch.hub ...")
    # Avoid the "forked repo" validation error on some torch versions.
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    print("  -> cached under $TORCH_HOME (default ~/.cache/torch/hub)")


if __name__ == "__main__":
    download_internvideo2()
    download_dinov2()
    print("\nDone. `weights/` is git-ignored; the training config points at the path above.")
