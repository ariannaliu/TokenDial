"""
TokenDial Video-to-Video Inference via FlowEdit (Inversion-Free)

FlowEdit adapts the inversion-free editing method (Kulikov et al., ICCV 2025)
to video editing with Wan 2.1 T2V 1.3B + TokenDial.

Instead of adding noise to the source video and denoising (SDEdit), FlowEdit:
  1. Starts at the clean source latent z_edit = x_src
  2. At each timestep t, computes forward-noised versions:
       z_src = (1-σ_t)*x_src + σ_t*noise
       z_tar = z_edit + z_src - x_src
  3. Gets velocity predictions for both (V_src without rgb token, V_tar with rgb token)
  4. Propagates the edit ODE: z_edit += (σ_{t+1} - σ_t) * (V_tar - V_src)
  5. Optionally: final n_min steps use regular generative denoising (SDEdit-style)

Key insight for TokenDial:
  When src_prompt == tar_prompt, the text-based CFG difference cancels out in
  V_delta = V_tar - V_src, and the edit is PURELY driven by the rgb token:
      V_delta = rgb_cfg_scale * (V_with_rgb - V_without_rgb)
  This makes FlowEdit a natural fit for TokenDial: no inversion, no noise
  trade-off, and the rgb token directly steers the ODE trajectory.

Usage:
  python -m tokendial.v2v \
      --ckpt checkpoints/person_east_asian.safetensors \
      --input_video assets/demo_videos/man_real.mp4 \
      --prompt "<a description of the input video>" \
      --rgb_cfg_scales "0,1,2,3"
"""

import os
import argparse
import torch
from tqdm import tqdm

os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def get_wan_sigmas(num_steps: int, shift: float = 5.0) -> torch.Tensor:
    """
    Compute the shifted sigma schedule for Wan 2.1.
    Returns sigmas of shape [num_steps], decreasing from ~sigma_max toward 0.

    Matches FlowMatchScheduler.set_timesteps_wan with denoising_strength=1.0.
    """
    sigma_min = 0.0
    sigma_max = 1.0
    sigmas = torch.linspace(sigma_max, sigma_min, num_steps + 1)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return sigmas


def get_velocity_prediction(
    pipe: WanVideoPipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    cfg_scale: float = 5.0,
    context_null: torch.Tensor = None,
    use_rgb_token: bool = False,
    rgb_cfg_scale: float = 0.0,
    rgb_attention_mask: torch.Tensor = None,
    attn_collect_mode: bool = False,
    attn_record_timesteps: list = None,
    attn_collect_layers: list = None,
    attn_target_token_indices: list = None,
    attn_store: dict = None,
    **model_kwargs,
) -> torch.Tensor:
    """
    Get velocity prediction from the DiT with classifier-free guidance.

    When use_rgb_token=True and rgb_cfg_scale != 0:
      output = [null + cfg*(text_plain - null)] + rgb_cfg*(text_rgb - text_plain)
    This isolates the rgb token's contribution as a separate CFG term.

    When use_rgb_token=False or rgb_cfg_scale == 0:
      output = null + cfg*(text - null)    (standard CFG)
    """
    from diffsynth.pipelines.wan_video import model_fn_wan_video

    dit = pipe.dit
    models = {"dit": dit, "motion_controller": None, "vace": None, "vap": None, "animate_adapter": None}

    # Prepare base kwargs for model_fn
    base_kwargs = {
        "latents": latents,
        "timestep": timestep,
        "cfg_merge": False,
        "use_unified_sequence_parallel": False,
        "attn_collect_mode": attn_collect_mode,
        "attn_record_timesteps": attn_record_timesteps,
        "attn_collect_layers": attn_collect_layers,
        "attn_target_token_indices": attn_target_token_indices,
        "attn_store": attn_store,
        "rgb_attention_mask": rgb_attention_mask,
    }
    base_kwargs.update(model_kwargs)

    if rgb_cfg_scale != 0.0 and use_rgb_token:
        # Decomposed CFG: separate text prediction from rgb token effect.
        # This matches the pipeline's rgb_cfg_scale logic (wan_video.py:315-346).
        prev_use_rgb = bool(getattr(dit, "use_rgb_token", False))
        try:
            # (1) Text prediction WITHOUT rgb token
            setattr(dit, "use_rgb_token", False)
            noise_pred_text_plain = model_fn_wan_video(
                **models, **base_kwargs, context=context,
            )

            # (2) Null prediction for CFG
            if cfg_scale != 1.0 and context_null is not None:
                noise_pred_null = model_fn_wan_video(
                    **models, **{**base_kwargs, "attn_collect_mode": False}, context=context_null,
                )
                noise_pred = noise_pred_null + cfg_scale * (noise_pred_text_plain - noise_pred_null)
            else:
                noise_pred = noise_pred_text_plain

            # (3) Text prediction WITH rgb token
            setattr(dit, "use_rgb_token", True)
            noise_pred_text_rgb = model_fn_wan_video(
                **models, **{**base_kwargs, "attn_collect_mode": False}, context=context,
            )

            # Add rgb CFG term: the rgb token's isolated contribution
            noise_pred = noise_pred + rgb_cfg_scale * (noise_pred_text_rgb - noise_pred_text_plain)
        finally:
            setattr(dit, "use_rgb_token", prev_use_rgb)
    else:
        # Standard CFG without rgb token decomposition
        prev_use_rgb = bool(getattr(dit, "use_rgb_token", False))
        setattr(dit, "use_rgb_token", use_rgb_token)
        try:
            noise_pred_text = model_fn_wan_video(
                **models, **base_kwargs, context=context,
            )
            if cfg_scale != 1.0 and context_null is not None:
                noise_pred_null = model_fn_wan_video(
                    **models, **{**base_kwargs, "attn_collect_mode": False}, context=context_null,
                )
                noise_pred = noise_pred_null + cfg_scale * (noise_pred_text - noise_pred_null)
            else:
                noise_pred = noise_pred_text
        finally:
            setattr(dit, "use_rgb_token", prev_use_rgb)

    return noise_pred


@torch.no_grad()
def flowedit_wan_video(
    pipe: WanVideoPipeline,
    x_src: torch.Tensor,
    context_src: torch.Tensor,
    context_tar: torch.Tensor,
    context_null: torch.Tensor,
    # FlowEdit parameters
    T_steps: int = 50,
    n_avg: int = 1,
    n_min: int = 0,
    n_max: int = 40,
    src_cfg_scale: float = 5.0,
    tar_cfg_scale: float = 5.0,
    # TokenDial parameters
    use_rgb_token: bool = True,
    rgb_cfg_scale: float = 3.0,
    rgb_attention_mask: torch.Tensor = None,
    # Scheduler
    sigma_shift: float = 5.0,
    # Misc
    seed: int = 0,
    progress_bar_cmd=tqdm,
) -> torch.Tensor:
    """
    FlowEdit adapted for Wan 2.1 video model with TokenDial.

    The ODE propagates the velocity difference V_tar - V_src. When both prompts
    are the same, V_delta is purely the rgb token's contribution:
        V_delta = rgb_cfg_scale * (V_with_rgb - V_without_rgb)

    The sigma schedule goes from high (t≈1, noisy) to low (t≈0, clean).
    ODE step: z_edit += (sigma_next - sigma_current) * V_delta
    Since sigma_next < sigma_current, the step size is negative — we move
    "backward in time" along the flow trajectory, accumulating the edit.

    Args:
        pipe: WanVideoPipeline with loaded model
        x_src: Source video latent [B, C, T, H, W] (clean, VAE-encoded)
        context_src: Text embedding for source prompt
        context_tar: Text embedding for target prompt (same as src for pure TokenDial)
        context_null: Text embedding for negative/null prompt (for CFG)
        T_steps: Total number of timesteps in the schedule
        n_avg: Number of noise samples to average (higher = smoother, slower)
        n_min: Number of final steps using generative denoising (0 = pure ODE)
        n_max: Maximum ODE steps (skip first T_steps - n_max steps)
        src_cfg_scale: CFG scale for source velocity prediction
        tar_cfg_scale: CFG scale for target velocity prediction
        use_rgb_token: Whether to use TokenDial rgb token for target prediction
        rgb_cfg_scale: CFG scale for rgb token (0 = no rgb effect)
        rgb_attention_mask: Spatial mask for rgb token application [1, S, 1]
        sigma_shift: Wan sigma schedule shift parameter (default 5.0)
        seed: Random seed for reproducibility
        progress_bar_cmd: Progress bar function

    Returns:
        Edited latent [B, C, T, H, W]
    """
    device = x_src.device
    dtype = x_src.dtype

    # Compute sigma schedule (full range, matches Wan scheduler with denoising_strength=1.0)
    sigmas = get_wan_sigmas(T_steps, shift=sigma_shift).to(device)
    timesteps = sigmas * 1000  # Wan uses 0-1000 range for timestep conditioning

    # Initialize the edit ODE: z_edit starts at x_src (the clean source)
    zt_edit = x_src.clone()

    # Deterministic noise generator for reproducibility across n_avg samples
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    for i in tqdm(range(T_steps), desc="FlowEdit", disable=progress_bar_cmd is None):
        # Skip early steps (only run the last n_max steps)
        if T_steps - i > n_max:
            continue

        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1] if i + 1 < T_steps else torch.tensor(0.0, device=device)
        t_i = timesteps[i].unsqueeze(0).to(dtype=dtype, device=device)

        if T_steps - i > n_min:
            # ===== ODE phase: propagate via velocity difference =====
            V_delta_avg = torch.zeros_like(x_src)

            for _ in range(n_avg):
                # Sample forward noise
                fwd_noise = torch.randn(x_src.shape, generator=generator, device=device, dtype=dtype)

                # Forward process: create noisy source at current sigma
                # z_src(t) = (1 - σ_t) * x_src + σ_t * noise
                zt_src = (1 - sigma_i) * x_src + sigma_i * fwd_noise

                # Coupled target: maintains the accumulated edit offset in noisy space
                # z_tar = z_edit + (z_src - x_src) = z_edit + σ_t * (noise - x_src)
                zt_tar = zt_edit + zt_src - x_src

                # Source velocity: standard CFG, NO rgb token
                V_src = get_velocity_prediction(
                    pipe, zt_src, t_i,
                    context=context_src,
                    context_null=context_null,
                    cfg_scale=src_cfg_scale,
                    use_rgb_token=False,
                    rgb_cfg_scale=0.0,
                )

                # Target velocity: CFG + rgb token (the edit direction)
                V_tar = get_velocity_prediction(
                    pipe, zt_tar, t_i,
                    context=context_tar,
                    context_null=context_null,
                    cfg_scale=tar_cfg_scale,
                    use_rgb_token=use_rgb_token,
                    rgb_cfg_scale=rgb_cfg_scale,
                    rgb_attention_mask=rgb_attention_mask,
                )

                V_delta_avg += (1.0 / n_avg) * (V_tar - V_src)

            # Propagate the ODE: z_edit += (σ_next - σ_i) * V_delta
            # Note: σ_next < σ_i, so the step is negative — correct for flow matching
            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (sigma_next - sigma_i).float() * V_delta_avg.float()
            zt_edit = zt_edit.to(dtype)

        else:
            # ===== Generative phase (last n_min steps): regular denoising =====
            if i == T_steps - n_min:
                # Initialize: create noised version and apply accumulated edit offset
                fwd_noise = torch.randn(x_src.shape, generator=generator, device=device, dtype=dtype)
                xt_src = (1 - sigma_i) * x_src + sigma_i * fwd_noise
                xt_tar = zt_edit + xt_src - x_src

            # Denoise with target prompt + rgb token (standard generative sampling)
            V_tar = get_velocity_prediction(
                pipe, xt_tar, t_i,
                context=context_tar,
                context_null=context_null,
                cfg_scale=tar_cfg_scale,
                use_rgb_token=use_rgb_token,
                rgb_cfg_scale=rgb_cfg_scale,
                rgb_attention_mask=rgb_attention_mask,
            )

            # Euler step: x_{t+1} = x_t + v * (σ_next - σ_i)
            xt_tar = xt_tar.to(torch.float32)
            xt_tar = xt_tar + (sigma_next - sigma_i).float() * V_tar.float()
            xt_tar = xt_tar.to(dtype)

    return zt_edit if n_min == 0 else xt_tar


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TokenDial video-to-video editing via FlowEdit (inversion-free)."
    )
    parser.add_argument("--input_video", type=str,
                        default="assets/demo_videos/man_real.mp4",
                        help="Path to the source video to edit.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/person_older.safetensors",
                        help="Path to the trained TokenDial rgb_token checkpoint.")
    parser.add_argument("--prompt", type=str,
                        default="A man in a dark suit and tie stands on a city street at dusk, turning his head, blurred bokeh lights in the background.",
                        help="Prompt describing the source video. Used for BOTH source and target "
                             "so the edit is driven purely by the rgb token.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output_dir", type=str, default="outputs/v2v")
    parser.add_argument("--rgb_cfg_scales", type=str, default="0,1,2,3",
                        help="Comma-separated rgb_cfg_scales to sweep. 0 = reconstruction baseline; "
                             "higher = stronger edit; negative = reversed edit.")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Edit the first N frames of the input video; must be 4n+1 (e.g. 17, 33, 61, 81). "
                             "Default: the whole video, trimmed to 4n+1 frames.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    # FlowEdit parameters
    parser.add_argument("--T_steps", type=int, default=50, help="Total timesteps in the schedule.")
    parser.add_argument("--n_avg", type=int, default=1, help="Noise averages per ODE step (1=fast).")
    parser.add_argument("--n_min", type=int, default=0, help="Final generative steps (0=pure ODE).")
    parser.add_argument("--n_max", type=int, default=40, help="ODE steps to run (edit intensity).")
    parser.add_argument("--src_cfg_scale", type=float, default=5.0)
    parser.add_argument("--tar_cfg_scale", type=float, default=5.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    rgb_cfg_scales = [float(x) for x in args.rgb_cfg_scales.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load input video. Wan needs 4n+1 frames: by default edit the whole video, trimmed to 4n+1.
    if args.num_frames is not None and args.num_frames % 4 != 1:
        raise ValueError(f"--num_frames must be 4n+1 (e.g. 17, 33, 61, 81), got {args.num_frames}.")
    input_video = VideoData(args.input_video, height=args.height, width=args.width)
    total_frames = len(input_video)
    num_frames = (total_frames - 1) // 4 * 4 + 1 if args.num_frames is None else args.num_frames
    if total_frames < num_frames:
        raise ValueError(f"{args.input_video} has {total_frames} frames, fewer than --num_frames={num_frames}.")
    if num_frames > 81:
        print(f"[warn] Editing {num_frames} frames. Wan2.1 is trained on clips of up to 81 frames, so longer "
              f"inputs are slower, use more memory and may lose quality; consider --num_frames 81.")
    source_frames = [input_video[i] for i in range(num_frames)]
    print(f"Loaded input video: {args.input_video} (editing the first {num_frames} of {total_frames} frames)")

    # Load pipeline
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

    # Load TokenDial checkpoint (rgb_token only; strict=False loads it on top of the base DiT)
    state_dict = load_state_dict(args.ckpt)
    pipe.dit.load_state_dict(state_dict, strict=False)

    # Encode source video to latent
    print("Encoding source video to latent space...")
    pipe.load_models_to_device(["vae"])
    input_frames = pipe.preprocess_video(source_frames)
    x_src = pipe.vae.encode(
        input_frames, device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)
    ).to(dtype=pipe.torch_dtype, device=pipe.device)
    print(f"  x_src shape: {x_src.shape}")

    # Encode prompts (src == tar → pure rgb-token editing)
    print("Encoding prompts...")
    pipe.load_models_to_device(["text_encoder"])

    def encode_prompt(prompt_text):
        ids, mask = pipe.tokenizer(prompt_text, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = pipe.text_encoder(ids, mask)
        for _, v in enumerate(seq_lens):
            emb[:, v:] = 0
        return emb

    context_src = encode_prompt(args.prompt)
    context_tar = context_src
    context_null = encode_prompt(args.negative_prompt)

    # Run FlowEdit with different rgb_cfg_scales
    pipe.load_models_to_device(["dit"])

    for rgb_scale in rgb_cfg_scales:
        print(f"\n{'=' * 60}")
        print(f"FlowEdit V2V: rgb_cfg_scale={rgb_scale}")
        print(f"  T_steps={args.T_steps}, n_min={args.n_min}, n_max={args.n_max}, n_avg={args.n_avg}")
        print(f"{'=' * 60}")

        use_rgb = rgb_scale != 0.0
        pipe.dit.rgb_token_weight = 1.0 if use_rgb else 0.0

        edited_latent = flowedit_wan_video(
            pipe=pipe,
            x_src=x_src,
            context_src=context_src,
            context_tar=context_tar,
            context_null=context_null,
            T_steps=args.T_steps,
            n_avg=args.n_avg,
            n_min=args.n_min,
            n_max=args.n_max,
            src_cfg_scale=args.src_cfg_scale,
            tar_cfg_scale=args.tar_cfg_scale,
            use_rgb_token=use_rgb,
            rgb_cfg_scale=rgb_scale,
            rgb_attention_mask=None,  # uniform (no spatial mask)
            sigma_shift=args.sigma_shift,
            seed=args.seed,
        )

        # Decode
        pipe.load_models_to_device(["vae"])
        video = pipe.vae.decode(edited_latent, device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
        video = pipe.vae_output_to_video(video)

        output_path = os.path.join(args.output_dir, f"flowedit_rgb{rgb_scale}.mp4")
        save_video(video, output_path, fps=15, quality=5)
        print(f"✓ Saved: {output_path}")

        pipe.load_models_to_device(["dit"])

    print(f"\nDone. Videos saved to {args.output_dir}")


if __name__ == "__main__":
    main()
