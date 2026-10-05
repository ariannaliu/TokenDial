from .base_pipeline import BasePipeline
import torch

def TokenDialLoss(pipe: BasePipeline, **inputs):
    """
    TokenDial appearance loss.

    Runs two velocity predictions under the same (x_t, t):
    - v_on:  rgb token enabled  (gradients flow into rgb_token)
    - v_off: rgb token disabled (no grad; the reference "off" branch)

    Both are converted to x0 predictions and decoded to pixels, then an
    InternVideo2 feature loss steers the token-on video toward the target
    concept direction relative to the token-off video.
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    x0 = inputs["input_latents"]
    noise = torch.randn_like(x0)

    # Use the same noisy latent for both branches.
    inputs = dict(inputs)
    inputs["latents"] = pipe.scheduler.add_noise(x0, noise, timestep)

    # Optional: first-frame conditioning (keep behavior aligned with FlowMatchSFTLoss).
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    # Forward with/without token.
    dit = getattr(pipe, "dit", None)
    if dit is None:
        raise AttributeError("TokenDialLoss expects pipe.dit to exist.")
    prev_flag = bool(getattr(dit, "use_rgb_token", False))

    try:
        dit.use_rgb_token = True
        v_on = pipe.model_fn(**models, **inputs, timestep=timestep)

        with torch.no_grad():
            dit.use_rgb_token = False
            v_off = pipe.model_fn(**models, **inputs, timestep=timestep)
    finally:
        dit.use_rgb_token = prev_flag

    x_t = inputs["latents"]
    pred_x0 = pipe.scheduler.step(v_on, timestep=timestep, sample=x_t, to_final=True)
    pred_x0_wotoken = pipe.scheduler.step(v_off, timestep=timestep, sample=x_t, to_final=True)

    # If first frame is conditioned, avoid training loss on that frame.
    if "first_frame_latents" in inputs:
        pred_x0 = pred_x0[:, :, 1:]
        pred_x0_wotoken = pred_x0_wotoken[:, :, 1:]
        x0 = x0[:, :, 1:]

    # Decode latents to pixel space.
    # IMPORTANT: do NOT use pipe.vae.decode(...) since it offloads to CPU.
    vae = getattr(pipe, "vae", None)
    if vae is None or not hasattr(vae, "model") or not hasattr(vae, "scale"):
        raise AttributeError("TokenDialLoss expects pipe.vae.model and pipe.vae.scale for decoding.")

    with torch.no_grad():
        pred_pixels_wotoken = vae.model.decode(
            pred_x0_wotoken.to(dtype=pipe.torch_dtype),
            vae.scale,
        ).clamp(-1, 1)

    x0_pixels = torch.utils.checkpoint.checkpoint(
        lambda z: vae.model.decode(z, vae.scale),
        pred_x0.to(dtype=pipe.torch_dtype),
        use_reentrant=False,
    ).clamp(-1, 1)

    # InternVideo2 feature loss objective (token-on vs token-off).
    intern_fn = getattr(pipe, "internvideo_loss_fn", None)
    intern_w = float(getattr(pipe, "internvideo_loss_weight", 1.0))
    if intern_fn is None:
        raise AttributeError(
            "TokenDialLoss requires InternVideo2 loss initialized on pipe. "
            "Expected attributes: pipe.internvideo_loss_fn and pipe.internvideo_loss_weight."
        )

    intern_loss = intern_fn(pred_pixel=x0_pixels, target_pixel=pred_pixels_wotoken)
    # InternVideo2 loss may return per-element values; reduce to scalar for training.
    if isinstance(intern_loss, torch.Tensor) and intern_loss.ndim > 0:
        intern_loss = intern_loss.mean()

    loss = (intern_w * intern_loss)
    return loss


def TokenDialSpeedLoss(pipe: BasePipeline, **inputs):
    """
    TokenDial motion/speed loss using DINOv2 feature-flow self-constraint.

    This computes two x0 predictions under the same (x_t, t):
    - pred_x0: with rgb_token enabled (grad flows)
    - pred_x0_wotoken: with rgb_token disabled (no grad)

    Then decodes pred_x0 to pixels (with grad) and applies DINOv2 motion loss.
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    x0 = inputs["input_latents"]
    noise = torch.randn_like(x0)

    # Use the same noisy latent for both branches.
    inputs = dict(inputs)
    inputs["latents"] = pipe.scheduler.add_noise(x0, noise, timestep)

    # Optional: first-frame conditioning (keep behavior aligned with FlowMatchSFTLoss).
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    # Forward with/without token.
    dit = getattr(pipe, "dit", None)
    if dit is None:
        raise AttributeError("TokenDialSpeedLoss expects pipe.dit to exist.")
    prev_flag = bool(getattr(dit, "use_rgb_token", False))

    try:
        dit.use_rgb_token = True
        v_on = pipe.model_fn(**models, **inputs, timestep=timestep, rgb_skip_first_latent=True)

        with torch.no_grad():
            dit.use_rgb_token = False
            v_off = pipe.model_fn(**models, **inputs, timestep=timestep)
    finally:
        dit.use_rgb_token = prev_flag

    x_t = inputs["latents"]
    pred_x0 = pipe.scheduler.step(v_on, timestep=timestep, sample=x_t, to_final=True)
    pred_x0_wotoken = pipe.scheduler.step(v_off, timestep=timestep, sample=x_t, to_final=True)

    # If first frame is conditioned, avoid training loss on that frame.
    if "first_frame_latents" in inputs:
        pred_x0 = pred_x0[:, :, 1:]
        pred_x0_wotoken = pred_x0_wotoken[:, :, 1:]

    # Decode latents to pixel space.
    # IMPORTANT: do NOT use pipe.vae.decode(...) since it offloads to CPU.
    vae = getattr(pipe, "vae", None)
    if vae is None or not hasattr(vae, "model") or not hasattr(vae, "scale"):
        raise AttributeError("TokenDialSpeedLoss expects pipe.vae.model and pipe.vae.scale for decoding.")

    with torch.no_grad():
        pred_pixels_wotoken = vae.model.decode(
            pred_x0_wotoken.to(dtype=pipe.torch_dtype),
            vae.scale,
        ).clamp(-1, 1)

    pred_pixels = torch.utils.checkpoint.checkpoint(
        lambda z: vae.model.decode(z, vae.scale),
        pred_x0.to(dtype=pipe.torch_dtype),
        use_reentrant=False,
    ).clamp(-1, 1)

    dino_fn = getattr(pipe, "dino_speed_loss_fn", None)
    dino_w = float(getattr(pipe, "dino_speed_loss_weight", 0.0))
    if dino_fn is None:
        raise AttributeError(
            "TokenDialSpeedLoss requires DINO speed loss initialized on pipe. "
            "Expected attributes: pipe.dino_speed_loss_fn and pipe.dino_speed_loss_weight."
        )

    stride = int(getattr(pipe, "dino_speed_stride", 1))
    num_keyframes = int(getattr(pipe, "dino_speed_num_keyframes", 32))
    fast_ratio = float(getattr(pipe, "dino_speed_fast_ratio", 0.5))

    # Self-constraint only uses x0_pixels for device; pass a tensor on the right device.
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        motion_loss = dino_fn.calculate_fast_motion_loss_scale(
            pred_pixels,
            stride=stride,
            num_keyframes=num_keyframes,
            fast_ratio=fast_ratio,
        )
    if isinstance(motion_loss, torch.Tensor) and motion_loss.ndim > 0:
        motion_loss = motion_loss.mean()
    
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        dino_firstframe_reg_loss = dino_fn(pred_pixels,pred_pixels_wotoken)
    
    if isinstance(dino_firstframe_reg_loss, torch.Tensor) and dino_firstframe_reg_loss.ndim > 0:
        dino_firstframe_reg_loss = dino_firstframe_reg_loss.mean()

    # loss = dino_w * motion_loss + dino_firstframe_reg_loss
    firstframe_reg_weight = float(getattr(pipe, "dino_firstframe_reg_weight", 1.0))
    loss = dino_w * motion_loss + firstframe_reg_weight * dino_firstframe_reg_loss
    return loss

def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss

