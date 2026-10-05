"""
TokenDial inference (Wan2.1-T2V-1.3B).

Two modes:
  --mode appearance   (default) style/appearance slider.
                      If --target_words is given, a cross-attention mask is built
                      in a first pass so the edit is applied only to the relevant
                      region (localized editing). Without --target_words the token
                      is injected uniformly.
  --mode motion       motion/speed slider. The rgb token is injected with the first
                      latent frame skipped (identity anchor); no attention mask.

The primary control knob is --rgb_cfg_scales: 0 = base model (no effect),
positive = stronger effect, negative = reversed effect.

Examples:
  # appearance, localized to the dog via an attention mask
  python -m tokendial.inference \
      --ckpt checkpoints/dog_fluffy.safetensors \
      --prompt "A dog walking on the grass." --target_words dog \
      --rgb_cfg_scales "0,3"

  # motion (negative = faster, positive = slower); write negative scales with "="
  python -m tokendial.inference --mode motion \
      --ckpt checkpoints/motion.safetensors \
      --prompt "A street musician sings into a vintage microphone, eyes closed." \
      --rgb_cfg_scales="-3,-1.5,0,1.5,3"
"""

import os
import argparse

import torch
import numpy as np
from PIL import Image

os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from diffsynth.utils.data import save_video
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def normalize_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min + eps)


def find_target_token_indices(pipe, prompt: str, target_words: list) -> list:
    """Find token positions in the prompt embedding for the given target words."""
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids[0]
    mask = mask[0].bool()
    ids_valid = ids[: int(mask.sum().item())]
    tokens = pipe.tokenizer.tokenizer.convert_ids_to_tokens(ids_valid.tolist())
    targets = [w.lower().strip() for w in target_words]

    def norm_token(tok: str) -> str:
        return tok.lower().replace("▁", "").replace("Ġ", "").replace("##", "").strip()

    hits = []
    for i, tok in enumerate(tokens):
        nt = norm_token(tok)
        if any(nt == w or w in nt for w in targets):
            hits.append(i)
    hits = sorted(set(hits))
    if not hits:
        raise ValueError(f"No target token found for {target_words}. Decoded tokens: {tokens}")
    return hits


def build_rgb_attention_mask_from_store(attn_store, selected_timesteps=None,
                                        selected_layers=None, quantile=0.75) -> torch.Tensor:
    """Aggregate collected cross-attention maps into an rgb mask of shape [1, S, 1]."""
    collected = []
    for (timestep_id, layer_id), maps in attn_store.items():
        if selected_timesteps is not None and timestep_id not in selected_timesteps:
            continue
        if selected_layers is not None and layer_id not in selected_layers:
            continue
        for m in maps:
            collected.append(m[0].float())
    if not collected:
        raise ValueError("No attention maps collected with the given step/layer filters.")

    merged = normalize_01(torch.stack(collected, dim=0).mean(dim=0))
    if quantile is not None:
        q = torch.quantile(merged, quantile)
        merged = torch.clamp((merged - q) / (1.0 - q + 1e-6), min=0.0, max=1.0)
    return merged.view(1, -1, 1)


def save_mask_visualization(rgb_attention_mask, latent_frames, token_h, token_w, vis_dir, upsample=8):
    os.makedirs(vis_dir, exist_ok=True)
    flat = rgb_attention_mask.detach().float().cpu().view(-1)
    maps = normalize_01(flat.view(latent_frames, token_h, token_w))
    for i in range(latent_frames):
        img = (maps[i].numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img, mode="L")
        if upsample > 1:
            pil = pil.resize((token_w * upsample, token_h * upsample), Image.NEAREST)
        pil.save(os.path.join(vis_dir, f"attnmask_latent_frame_{i:02d}.png"))


def load_pipeline(ckpt_path: str) -> WanVideoPipeline:
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        redirect_common_files=False,
    )
    state_dict = load_state_dict(ckpt_path)
    pipe.dit.load_state_dict(state_dict, strict=False)  # rgb_token loads on top of base DiT
    return pipe


def build_mask(pipe, args, rgb_cfg_scales):
    """First pass: collect cross-attention on target words and build the spatial mask."""
    target_token_indices = find_target_token_indices(pipe, args.prompt, args.target_words)
    record_timesteps = [int(x) for x in args.attn_timesteps.split(",")] if args.attn_timesteps else None
    collect_layers = [int(x) for x in args.attn_layers.split(",")] if args.attn_layers else None

    print(f"[mask] target_words={args.target_words} token_indices={target_token_indices} "
          f"layers={collect_layers} timesteps={record_timesteps}")

    attn_store = {}
    pipe.dit.use_rgb_token = False
    _ = pipe(
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        seed=args.seed, tiled=True,
        height=args.height, width=args.width, num_frames=args.num_frames,
        rgb_cfg_scale=0.0,
        attn_collect_mode=True,
        attn_record_timesteps=record_timesteps,
        attn_collect_layers=collect_layers,
        attn_target_token_indices=target_token_indices,
        attn_store=attn_store,
        rgb_attention_mask=None,
    )
    mask = build_rgb_attention_mask_from_store(
        attn_store, selected_timesteps=record_timesteps,
        selected_layers=collect_layers, quantile=args.mask_quantile,
    )
    mask = mask ** args.mask_gamma

    if args.save_mask_vis:
        latent_frames = (args.num_frames - 1) // 4 + 1
        token_h = (args.height // pipe.vae.upsampling_factor) // pipe.dit.patch_size[1]
        token_w = (args.width // pipe.vae.upsampling_factor) // pipe.dit.patch_size[2]
        save_mask_visualization(mask, latent_frames, token_h, token_w,
                                os.path.join(args.output_dir, "mask_vis"))
    print(f"[mask] built: shape={tuple(mask.shape)} min={mask.min():.3f} max={mask.max():.3f}")
    return mask


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rgb_cfg_scales = [float(x) for x in args.rgb_cfg_scales.split(",")]

    pipe = load_pipeline(args.ckpt)

    # Motion mode: skip the first latent frame (identity anchor); no attention mask.
    if args.mode == "motion":
        pipe.dit.rgb_skip_first_latent = True

    # Appearance mode with target words: build a localized attention mask first.
    rgb_attention_mask = None
    if args.mode == "appearance" and args.target_words:
        rgb_attention_mask = build_mask(pipe, args, rgb_cfg_scales)

    pipe.dit.use_rgb_token = True
    for scale in rgb_cfg_scales:
        print(f"\n{'=' * 60}\n[{args.mode}] rgb_cfg_scale={scale}\n{'=' * 60}")
        pipe.dit.rgb_token_weight = 1.0 if scale != 0.0 else 0.0
        video = pipe(
            prompt=args.prompt, negative_prompt=args.negative_prompt,
            rgb_cfg_scale=scale,
            seed=args.seed, tiled=True,
            height=args.height, width=args.width, num_frames=args.num_frames,
            rgb_attention_mask=rgb_attention_mask,
        )
        out = os.path.join(args.output_dir, f"cfg_{scale}.mp4")
        save_video(video, out, fps=15, quality=5)
        print(f"✓ Saved: {out}")

    print(f"\nDone. Videos saved to {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="TokenDial inference (Wan2.1-T2V-1.3B).")
    p.add_argument("--ckpt", type=str, required=True, help="Path to the TokenDial rgb_token checkpoint.")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    p.add_argument("--mode", type=str, default="appearance", choices=["appearance", "motion"])
    p.add_argument("--output_dir", type=str, default="outputs/inference")
    p.add_argument("--rgb_cfg_scales", type=str, default="0,1,2,3",
                   help="Comma-separated rgb_cfg_scales. 0=base model, +=stronger, -=reversed.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_frames", type=int, default=41)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    # Appearance attention-mask options (only used when --target_words is set)
    p.add_argument("--target_words", type=str, nargs="*", default=None,
                   help="Words to localize the edit to (appearance mode). Omit for uniform injection.")
    p.add_argument("--attn_layers", type=str, default="13,15,17,21",
                   help="Comma-separated DiT layers to collect cross-attention from (empty=all).")
    p.add_argument("--attn_timesteps", type=str, default=None,
                   help="Comma-separated scheduler timesteps to collect at (empty=all steps).")
    p.add_argument("--mask_quantile", type=float, default=0.75, help="Threshold quantile for the mask.")
    p.add_argument("--mask_gamma", type=float, default=0.6, help="Mask sharpness (<1 broader, >1 tighter).")
    p.add_argument("--save_mask_vis", action="store_true", help="Save per-frame mask visualizations.")
    return p.parse_args()


if __name__ == "__main__":
    main()
