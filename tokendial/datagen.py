import argparse
import csv
import random
from pathlib import Path
import os
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

import torch

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


DEFAULT_NEGATIVE_PROMPT = (
    "oversaturated colors, overexposed, static, blurry details, subtitles, style, painting, "
    "frame freeze, gray tone, worst quality, low quality, jpeg artifacts, ugly, disfigured, "
    "deformed hands, deformed face, fused fingers, extra fingers, malformed limbs, cluttered background"
)


def read_prompts(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def assigned_indices(total: int, worker_id: int, num_workers: int) -> list[int]:
    return [i for i in range(total) if i % num_workers == worker_id]


def build_video_name(prefix: str, global_index: int, digits: int) -> str:
    return f"{prefix}_{global_index:0{digits}d}.mp4"


def choose_seed(global_index: int, strategy: str, seed_base: int) -> int | None:
    if strategy == "none":
        return None
    if strategy == "random":
        return random.randint(0, 2**31 - 1)
    if strategy == "deterministic":
        return seed_base + global_index
    raise ValueError(f"Unsupported seed strategy: {strategy}")


def build_pipeline(device: str, torch_dtype: str) -> WanVideoPipeline:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

    return WanVideoPipeline.from_pretrained(
        torch_dtype=dtype_map[torch_dtype],
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        redirect_common_files=False
    )


def merge_metadata_shards(
    metadata_root: Path,
    metadata_name: str,
    num_workers: int,
    strict: bool,
) -> None:
    rows: list[dict] = []
    missing: list[Path] = []

    for worker_id in range(num_workers):
        shard_path = metadata_root / f"metadata.worker{worker_id}.csv"
        if not shard_path.exists():
            missing.append(shard_path)
            continue
        with shard_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({"video": row["video"], "prompt": row["prompt"]})

    if missing and strict:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing metadata shards: {missing_str}")

    rows.sort(key=lambda r: r["video"])
    output_path = metadata_root / metadata_name
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged {len(rows)} rows -> {output_path}")
    if missing:
        print("Warning: some shard files were missing.")
        for p in missing:
            print(f"  - {p}")


def run_generation(args: argparse.Namespace) -> None:
    prompts_txt = Path(args.prompts_txt)
    output_train_dir = Path(args.output_train_dir)
    metadata_root = Path(args.metadata_root)
    metadata_root.mkdir(parents=True, exist_ok=True)
    output_train_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(prompts_txt)
    if len(prompts) == 0:
        raise ValueError(f"No prompts found in {prompts_txt}")

    shard_path = metadata_root / f"metadata.worker{args.worker_id}.csv"
    worker_indices = assigned_indices(len(prompts), args.worker_id, args.num_workers)

    pipe = build_pipeline(device=args.device, torch_dtype=args.torch_dtype)

    print(
        f"Worker {args.worker_id}/{args.num_workers} handles "
        f"{len(worker_indices)} prompts out of {len(prompts)} total."
    )

    with shard_path.open("w", encoding="utf-8", newline="") as shard_file:
        writer = csv.DictWriter(shard_file, fieldnames=["video", "prompt"])
        writer.writeheader()

        for rank, prompt_index in enumerate(worker_indices, start=1):
            prompt = prompts[prompt_index]
            global_index = args.start_index + prompt_index
            video_name = build_video_name(args.file_prefix, global_index, args.digits)
            save_path = output_train_dir / video_name

            writer.writerow({"video": video_name, "prompt": prompt})

            if save_path.exists() and (not args.overwrite_existing):
                print(f"[{rank}/{len(worker_indices)}] skip existing: {save_path}")
                continue

            seed = choose_seed(global_index, args.seed_strategy, args.seed_base)
            print(
                f"[{rank}/{len(worker_indices)}] generate: {video_name} | "
                f"global_index={global_index} | seed={seed}"
            )

            with torch.inference_mode():
                video = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    seed=seed,
                    rand_device=args.rand_device,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    sigma_shift=args.sigma_shift,
                    tiled=args.tiled,
                )
            save_video(video, str(save_path), fps=args.fps, quality=args.quality)

    print(f"Worker metadata written to: {shard_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Tokendial training videos from prompts.")
    parser.add_argument("--prompts_txt", type=str, help="Path to prompts txt (one prompt per line).")
    parser.add_argument("--output_train_dir", type=str, default="data/train")
    parser.add_argument("--metadata_root", type=str, default="data")
    parser.add_argument("--metadata_name", type=str, default="metadata.csv")

    parser.add_argument("--file_prefix", type=str, default="video")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--digits", type=int, default=6)
    parser.add_argument("--overwrite_existing", action="store_true")

    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--seed_strategy", choices=["none", "random", "deterministic"], default="none")
    parser.add_argument("--seed_base", type=int, default=0)
    parser.add_argument("--rand_device", type=str, default="cpu")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=61)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)

    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--merge_strict", action="store_true")

    args = parser.parse_args()

    if args.worker_id < 0 or args.num_workers <= 0 or args.worker_id >= args.num_workers:
        raise ValueError("Require 0 <= worker_id < num_workers and num_workers > 0.")
    if (not args.merge_only) and (not args.prompts_txt):
        raise ValueError("--prompts_txt is required unless --merge_only is set.")
    return args


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_metadata_shards(
            metadata_root=Path(args.metadata_root),
            metadata_name=args.metadata_name,
            num_workers=args.num_workers,
            strict=args.merge_strict,
        )
    else:
        run_generation(args)


if __name__ == "__main__":
    main()
