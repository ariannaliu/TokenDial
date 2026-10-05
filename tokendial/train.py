"""
TokenDial training entry (Wan2.1-T2V-1.3B).

Config-driven: all hyperparameters live in a YAML recipe (see configs/), and the
edit direction for appearance sliders is a separate JSON passed with --direction.

    accelerate launch -m tokendial.train --config configs/appearance.yaml \
        --direction configs/directions/person_older.json

Only the per-layer rgb_token is trained; the DiT stays frozen.
"""
import torch, os, argparse, accelerate, warnings, glob, yaml
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="tokendial:appearance",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        internvideo_loss_weight: float = 1.0,
        internvideo_config_path: str = "feature_losses/internvideo2/demo/internvideo2_stage2_config.py",
        internvideo_pretrained_path: str = "",
        internvideo_guidance_method: str = "text",
        internvideo_guidance_concept: str = "",
        internvideo_prompts_path: str = None,
        dino_speed_loss_weight: float = 1.0,
        dino_speed_model_name: str = "dinov2_vitl14",
        dino_speed_stride: int = 1,
        dino_speed_num_keyframes: int = 32,
        dino_speed_fast_ratio: float = 0.5,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is forcibly enabled to avoid OOM.")
            use_gradient_checkpointing = True

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, redirect_common_files=False)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        task = str(task)
        if task == "tokendial" or task == "tokendial:train":
            task = "tokendial:appearance"

        # InternVideo2 feature loss (appearance).
        self.pipe.internvideo_loss_weight = 0.0
        self.pipe.internvideo_loss_fn = None
        # DINOv2 speed loss (motion).
        self.pipe.dino_speed_loss_weight = 0.0
        self.pipe.dino_speed_loss_fn = None
        self.pipe.dino_speed_stride = int(dino_speed_stride)
        self.pipe.dino_speed_num_keyframes = int(dino_speed_num_keyframes)
        self.pipe.dino_speed_fast_ratio = float(dino_speed_fast_ratio)
        self.pipe.dino_firstframe_reg_weight = 0.0

        if task == "tokendial:appearance":
            self.pipe.internvideo_loss_weight = float(internvideo_loss_weight)
            if self.pipe.internvideo_loss_weight <= 0:
                raise ValueError("tokendial:appearance requires internvideo.loss_weight > 0.")

            _pretrained = str(internvideo_pretrained_path or "")
            if not _pretrained or not os.path.isfile(_pretrained):
                # Fall back to the download cache (see: python -m tokendial.download_weights).
                from huggingface_hub import snapshot_download
                _snap = snapshot_download(repo_id="OpenGVLab/InternVideo2-Stage2_1B-224p-f4")
                _pretrained = os.path.join(_snap, "InternVideo2-stage2_1b-224p-f4.pt")
            print(f"[InternVideo2] checkpoint: {_pretrained}")

            from feature_losses.internvideo2.feature_loss import InternVideo2FeatureLoss
            self.pipe.internvideo_loss_fn = InternVideo2FeatureLoss(
                config_path=internvideo_config_path,
                pretrained_path=_pretrained,
                guidance_method=internvideo_guidance_method,
                guidance_concept=internvideo_guidance_concept,
                device=str(device),
                prompts_path=internvideo_prompts_path or None,
            )

        elif task == "tokendial:motion":
            self.pipe.dino_speed_loss_weight = float(dino_speed_loss_weight)
            if self.pipe.dino_speed_loss_weight <= 0:
                raise ValueError("tokendial:motion requires dino.loss_weight > 0.")
            from feature_losses.dinov2_speed_loss import DINOv2IdentityLoss
            self.pipe.dino_speed_loss_fn = DINOv2IdentityLoss(
                model_name=str(dino_speed_model_name),
                mode="first_frame_cls_only",
                device=str(device),
            )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "tokendial:appearance": lambda pipe, inputs_shared, inputs_posi, inputs_nega: TokenDialLoss(pipe, **inputs_shared, **inputs_posi),
            "tokendial:motion": lambda pipe, inputs_shared, inputs_posi, inputs_nega: TokenDialSpeedLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


# ---- YAML config → argparse namespace ----------------------------------------

_INTERNVIDEO_MAP = {
    "loss_weight": "internvideo_loss_weight",
    "guidance_method": "internvideo_guidance_method",
    "guidance_concept": "internvideo_guidance_concept",
    "pretrained_path": "internvideo_pretrained_path",
    "config_path": "internvideo_config_path",
    "direction": "internvideo_prompts_path",
}
_DINO_MAP = {
    "loss_weight": "dino_speed_loss_weight",
    "model_name": "dino_speed_model_name",
    "num_keyframes": "dino_speed_num_keyframes",
    "fast_ratio": "dino_speed_fast_ratio",
    "stride": "dino_speed_stride",
    "firstframe_reg_weight": "dino_firstframe_reg_weight",
}


def flatten_config(cfg: dict) -> dict:
    flat = {}
    for k, v in cfg.items():
        if k == "internvideo" and isinstance(v, dict):
            for kk, vv in v.items():
                flat[_INTERNVIDEO_MAP.get(kk, "internvideo_" + kk)] = vv
        elif k == "dino" and isinstance(v, dict):
            for kk, vv in v.items():
                flat[_DINO_MAP.get(kk, "dino_" + kk)] = vv
        else:
            flat[k] = v
    return flat


def build_parser():
    parser = argparse.ArgumentParser(description="TokenDial training (config-driven).")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML training recipe (see configs/).")
    parser.add_argument("--direction", type=str, default=None, help="Path to a guidance-direction JSON (appearance); overrides the config default.")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--rgb_token_weight", type=float, default=1.0)
    parser.add_argument("--internvideo_loss_weight", type=float, default=1.0)
    parser.add_argument("--internvideo_config_path", type=str, default="feature_losses/internvideo2/demo/internvideo2_stage2_config.py")
    parser.add_argument("--internvideo_pretrained_path", type=str, default="")
    parser.add_argument("--internvideo_guidance_method", type=str, default="text")
    parser.add_argument("--internvideo_guidance_concept", type=str, default="")
    parser.add_argument("--internvideo_prompts_path", type=str, default=None)
    parser.add_argument("--dino_speed_loss_weight", type=float, default=1.0)
    parser.add_argument("--dino_speed_model_name", type=str, default="dinov2_vitl14")
    parser.add_argument("--dino_speed_stride", type=int, default=1)
    parser.add_argument("--dino_speed_num_keyframes", type=int, default=32)
    parser.add_argument("--dino_speed_fast_ratio", type=float, default=0.5)
    parser.add_argument("--dino_firstframe_reg_weight", type=float, default=0.0)
    # TokenDial invariants (kept fixed; the DiT is frozen and only rgb_token trains).
    parser.set_defaults(
        use_rgb_token=True,
        trainable_models="dit",
        remove_prefix_in_ckpt="pipe.dit.",
        use_gradient_checkpointing=True,
    )
    return parser


def parse_args():
    # 1) read --config early
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, required=True)
    pre.add_argument("--direction", type=str, default=None)
    known, _ = pre.parse_known_args()

    with open(known.config) as f:
        cfg = yaml.safe_load(f) or {}
    flat = flatten_config(cfg)
    if known.direction:
        flat["internvideo_prompts_path"] = known.direction

    # 2) YAML values become defaults; explicit CLI flags still override them.
    parser = build_parser()
    # argparse checks `required` before applying set_defaults, so relax fields the
    # YAML is expected to supply (dataset_base_path is required=True upstream).
    for action in parser._actions:
        if action.dest == "dataset_base_path":
            action.required = False
    parser.set_defaults(**flat)
    args = parser.parse_args()
    if not args.dataset_base_path:
        parser.error("dataset_base_path must be set in the --config YAML (or on the CLI).")
    return args


def main():
    args = parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        internvideo_loss_weight=args.internvideo_loss_weight,
        internvideo_config_path=args.internvideo_config_path,
        internvideo_pretrained_path=args.internvideo_pretrained_path,
        internvideo_guidance_method=args.internvideo_guidance_method,
        internvideo_guidance_concept=args.internvideo_guidance_concept,
        internvideo_prompts_path=args.internvideo_prompts_path,
        dino_speed_loss_weight=args.dino_speed_loss_weight,
        dino_speed_model_name=args.dino_speed_model_name,
        dino_speed_stride=args.dino_speed_stride,
        dino_speed_num_keyframes=args.dino_speed_num_keyframes,
        dino_speed_fast_ratio=args.dino_speed_fast_ratio,
    )
    model.pipe.dino_firstframe_reg_weight = args.dino_firstframe_reg_weight

    # RGB-token training mode: freeze everything except rgb_token; resume if present.
    if not hasattr(model.pipe.dit, "rgb_token"):
        raise AttributeError("rgb_token not found in pipe.dit.")
    model.pipe.dit.use_rgb_token = True
    model.pipe.dit.rgb_token_weight = args.rgb_token_weight
    if "motion" in str(args.task):
        model.pipe.dit.rgb_skip_first_latent = True  # keep the first frame as identity anchor
    model.pipe.dit.requires_grad_(False)
    model.pipe.dit.rgb_token.requires_grad_(True)

    if args.output_path is not None and os.path.isdir(args.output_path):
        ckpts = glob.glob(os.path.join(args.output_path, "*.safetensors"))
        if len(ckpts) > 0:
            latest_ckpt = max(ckpts, key=os.path.getmtime)
            state = load_state_dict(latest_ckpt)
            if "rgb_token" in state:
                with torch.no_grad():
                    model.pipe.dit.rgb_token.copy_(state["rgb_token"].to(device=model.pipe.dit.rgb_token.device, dtype=model.pipe.dit.rgb_token.dtype))
                print(f"[rgb_token] resumed from {latest_ckpt}")

    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
