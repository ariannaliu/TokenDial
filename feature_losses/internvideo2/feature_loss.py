import sys
import os
from turtle import pos
import torch
import torch.nn as nn
import numpy as np
import cv2

from .demo_config import (Config, eval_dict_leaf)
from .demo.utils import (retrieve_text, video_text_similarity,
                      _frame_from_video,
                      setup_internvideo2, frames2tensor)

class InternVideo2FeatureLoss(nn.Module):
    """
    A perceptual loss module using InternVideo2 visual features.
    This module takes predicted and target videos (in pixel space),
    computes their visual features using a pretrained InternVideo2 model,
    and returns the loss between these features.
    """
    def __init__(self, config_path: str, pretrained_path: str, guidance_method: str = 'text' ,guidance_concept: str = '', device: str = 'cuda', prompts_path: str = None):
        """
        Initializes the InternVideo2 feature loss module.
        Args:
            config_path (str): Path to the InternVideo2 config file.
            pretrained_path (str): Path to the InternVideo2 pretrained model checkpoint.
            guidance_method (str): The edit-direction guidance for the loss, either "text" or "video".
            guidance_concept (str): The concept for loss calculation, either "person" or "snowflake" or "campfire" or "smoke" or "ink".
            device (str): The device to run the model on.
            prompts_path (str): Path to the prompts file.
        """
        super().__init__()
        self.device = torch.device(device)
        self.prompts_path = prompts_path

        print("Initializing InternVideo2 for perceptual loss...")
        # 1. Setup InternVideo2 model
        config = Config.from_file(config_path)
        config = eval_dict_leaf(config)
        config.pretrained_path = pretrained_path
        # All weights (vision + text + fusion) come from the full stage-2 checkpoint
        # above; don't load a separate vision-backbone checkpoint (which would
        # otherwise depend on a hardcoded local path).
        config.model.vision_encoder.pretrained = None
        config.device = device

        config.use_bf16 = True
        self.intern_model, _ = setup_internvideo2(config)
        self.intern_model.eval()
        for param in self.intern_model.parameters():
            param.requires_grad = False

        self.intern_model.to(self.device)
        # self.intern_model = self.intern_model.to(torch.bfloat16)

        # Store some config for processing frames
        self.num_frames = config.model.vision_encoder.num_frames
        self.target_size = (config.model.vision_encoder.img_size, config.model.vision_encoder.img_size)


        self.loss_fn = nn.CosineSimilarity(dim=1, eps=1e-6)

        # set up guidance
        self.guidance_method = guidance_method
        self.guidance_concept = guidance_concept
        self.guidance_vector = None
        self.guidance_concept_vector = None
        self._build_guidance()
    
    def _build_guidance(self):
        """
        Sets up the edit direction for the loss. Supports 'text' and 'video',
        both driven by the direction JSON passed via prompts_path
        (--direction / internvideo.direction in the config).
        """

        # Concept anchor vector (used only by the optional regularization term,
        # which is disabled by default).
        if self.guidance_concept not in ['person', 'snowflake', 'campfire', 'smoke', 'ink']:
            self.guidance_concept_vector = torch.zeros((1, 512), device=self.device)
        else:
            guidance_concept_prompt = [self.guidance_concept]
            guidance_concept_feat = [self.intern_model.get_txt_feat(t) for t in guidance_concept_prompt]
            guidance_concept_feat = torch.cat(guidance_concept_feat, dim=0)
            guidance_concept_feat = guidance_concept_feat.mean(dim=0, keepdim=True)
            self.guidance_concept_vector = guidance_concept_feat.to(self.device)

        if self.guidance_method not in ['text', 'video']:
            raise ValueError(
                f"Unsupported guidance method: {self.guidance_method} (expected 'text' or 'video')."
            )

        # Both methods read the edit direction from a JSON.
        if not self.prompts_path:
            raise ValueError(
                f"'{self.guidance_method}' guidance requires a direction JSON. "
                "Pass --direction <path.json> (or set internvideo.direction in the config)."
            )
        import json
        with open(self.prompts_path, 'r') as f:
            data = json.load(f)

        if self.guidance_method == 'text':
            direction_prompts = data.get("direction_prompts", [])
            neg_direction_prompts = data.get("neg_direction_prompts", [])
            base_prompts = data.get("base_prompts", ["a video"])
            # Optional: remove components of guidance along a subspace spanned by anchor-pair
            # directions. Each pair [a, b] defines an anchor direction feat(a) - feat(b).
            orthogonal_anchor_pairs = data.get("orthogonal_anchor_pairs", [])
            orthogonalize_strength = float(data.get("orthogonalize_strength", 1.0))
            orthogonalize_rcond = float(data.get("orthogonalize_rcond", 1e-3))
            if not direction_prompts or not neg_direction_prompts:
                raise ValueError(
                    "text guidance JSON must contain 'direction_prompts' and 'neg_direction_prompts'."
                )

            with torch.no_grad():
                direction_feat = [self.intern_model.get_txt_feat(t) for t in direction_prompts]
                direction_feat = torch.cat(direction_feat, dim=0)               # (n,512)
                direction_feat = direction_feat.mean(dim=0, keepdim=True)       # (1,512)
                base_feat = [self.intern_model.get_txt_feat(t) for t in base_prompts]
                base_feat = torch.cat(base_feat, dim=0)
                base_feat = base_feat.mean(dim=0, keepdim=True)
                neg_direction_feat = [self.intern_model.get_txt_feat(t) for t in neg_direction_prompts]
                neg_direction_feat = torch.cat(neg_direction_feat, dim=0)
                neg_direction_feat = neg_direction_feat.mean(dim=0, keepdim=True)

            g = (direction_feat - neg_direction_feat).to(self.device)
            if orthogonal_anchor_pairs:
                g = self._project_orthogonal_to_anchor_pairs(
                    g=g,
                    anchor_pairs=orthogonal_anchor_pairs,
                    strength=orthogonalize_strength,
                    rcond=orthogonalize_rcond,
                )
            self.guidance_vector = g

        elif self.guidance_method == 'video':
            # Reference videos that define the edit direction, read from the JSON.
            # Expected keys: "video_dir", "pos_videos" (list), "neg_videos" (list).
            video_dir = data.get("video_dir", "")
            pos_direction_videos = data.get("pos_videos", [])
            neg_direction_videos = data.get("neg_videos", [])
            if not video_dir or not pos_direction_videos or not neg_direction_videos:
                raise ValueError(
                    "video guidance JSON must contain 'video_dir', 'pos_videos', and 'neg_videos'."
                )

            with torch.no_grad():
                pos_features = []
                for pos_videos in pos_direction_videos:
                    video_path = os.path.join(video_dir, pos_videos)
                    video_frames = [x for x in _frame_from_video(cv2.VideoCapture(video_path))]
                    frames_tensor = frames2tensor(video_frames, fnum=self.num_frames, target_size=self.target_size).to("cuda").to(torch.bfloat16)
                    vid_feat = self.intern_model.get_vid_feat(frames_tensor)
                    pos_features.append(vid_feat)
                pos_feat = torch.cat(pos_features, dim=0)  # (n, 512)
                pos_feat = pos_feat.mean(dim=0, keepdim=True)  # (1, 512)

                neg_features = []
                for neg_videos in neg_direction_videos:
                    video_path = os.path.join(video_dir, neg_videos)
                    video_frames = [x for x in _frame_from_video(cv2.VideoCapture(video_path))]
                    frames_tensor = frames2tensor(video_frames, fnum=self.num_frames, target_size=self.target_size).to("cuda").to(torch.bfloat16)
                    vid_feat = self.intern_model.get_vid_feat(frames_tensor)
                    neg_features.append(vid_feat)
                neg_feat = torch.cat(neg_features, dim=0)  # (n, 512)
                neg_feat = neg_feat.mean(dim=0, keepdim=True)  # (1, 512)

            self.guidance_vector = (pos_feat - neg_feat).to(self.device)
    
    def _project_orthogonal_to_anchor_pairs(
        self,
        g: torch.Tensor,                    # (1, D)
        anchor_pairs,                       # list[[str, str], ...]
        strength: float = 1.0,              # alpha in g' = g - alpha * P(g)
        rcond: float = 1e-3,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Build anchor directions from text prompt pairs: a_i = feat(a) - feat(b),
        form subspace U = span{a_i}, and remove (softly) projection of g onto U:
            g' = g - strength * P_U(g)

        This helps prevent guidance entanglement (e.g., "heavier" accidentally also "older")
        by explicitly removing undesired directions.
        """
        if anchor_pairs is None or len(anchor_pairs) == 0:
            return g

        strength = float(strength)
        if strength <= 0.0:
            return g

        with torch.no_grad():
            anchor_vecs = []
            for pair in anchor_pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                pa, pb = pair[0], pair[1]
                fa = self.intern_model.get_txt_feat(pa)  # (1, D)
                fb = self.intern_model.get_txt_feat(pb)  # (1, D)
                anchor_vecs.append(fa - fb)              # (1, D)

            if len(anchor_vecs) == 0:
                return g

            A = torch.cat(anchor_vecs, dim=0)  # (K, D)

        # Hard safety: if any NaN/Inf exists, skip orthogonalization
        if not torch.isfinite(A).all():
            return g

        # Normalize each anchor direction for numerical stability; span is unchanged.
        A = A / (A.norm(dim=1, keepdim=True) + eps)
        if float(A.abs().sum().item()) < eps:
            return g

        # SVD gives an orthonormal basis of span(A) in feature space R^D.
        Af = A.float()
        _, S, Vh = torch.linalg.svd(Af, full_matrices=False)
        if S.numel() == 0:
            return g

        tol = float(rcond) * float(S.max().item())
        r = int((S > tol).sum().item())
        if r == 0:
            return g

        V = Vh[:r, :].T  # (D, r), columns orthonormal

        gf = g.float()
        proj = (gf @ V) @ V.T
        g_new = gf - strength * proj

        # Avoid killing the guidance direction (would later normalize to unstable values)
        if float(g_new.norm().item()) < 1e-6:
            return g

        return g_new.to(dtype=g.dtype, device=g.device)
    

    def _preprocess_video(self, video_tensor: torch.Tensor, stride=None, still=None) -> torch.Tensor:
        """
        Preprocesses a batch of video tensors to the format expected by InternVideo2.
        Args:
            video_tensor (torch.Tensor): A tensor of shape [B, C, T, H, W] with pixel values in [-1, 1].
        Returns:
            torch.Tensor: A tensor of shape [B, T_iv, C, H_iv, W_iv] ready for InternVideo2.
        """
        if video_tensor.shape[2] < self.num_frames:
             # If the input video has fewer frames than required, we can't sample.
             # simply bypass by returning a tensor that will result in zero loss.
             return None

        # 1. Denormalize from [-1, 1] to [0, 1]
        # video_tensor = (video_tensor.clamp(-1, 1) + 1) / 2.0
        one = torch.tensor(1.0, dtype=video_tensor.dtype, device=video_tensor.device)
        two = torch.tensor(2.0, dtype=video_tensor.dtype, device=video_tensor.device)
        video_tensor = (video_tensor.clamp(-1, 1) + one) / two

        b, c, t, h, w = video_tensor.shape

        # 2. Uniformly sample frames along the time dimension
        if still:
            # All keyframes are the first frame, representing a "still" video.
            indices = torch.zeros(self.num_frames, device=video_tensor.device, dtype=torch.long)
        else:
            if not stride:
                indices = torch.linspace(0, t - 1, self.num_frames, device=video_tensor.device, dtype=torch.long)
            else:
                if (self.num_frames - 1) * stride + 1 > t:
                    print(f"Warning: video with {t} frames is too short for {self.num_frames} frames with stride {stride}. ")
                    return None

                # Select keyframes from the beginning
                indices = torch.linspace(0, (self.num_frames - 1) * stride, self.num_frames, device=video_tensor.device, dtype=torch.long)

                # # # Select keyframes from the center
                # start_idx = (t - ((self.num_frames - 1) * stride + 1)) // 2
                # indices = torch.linspace(start_idx, start_idx + (self.num_frames - 1) * stride, self.num_frames, device=video_tensor.device, dtype=torch.long)

        video_tensor = torch.index_select(video_tensor, 2, indices)

        # 3. Resize frames to the target size
        # To use interpolate, we need to treat frames as a batch of images.
        # [B, C, T, H, W] -> [B*T, C, H, W]
        video_tensor_reshaped = video_tensor.permute(0, 2, 1, 3, 4).reshape(b * self.num_frames, c, h, w)


        resized_video = torch.nn.functional.interpolate(
            video_tensor_reshaped,
            size=self.target_size,
            mode='bicubic',
            align_corners=False
        ).to(video_tensor.dtype)

        # 4. Normalize with same as internvideo2 utils.py
        mean = torch.tensor([0.485, 0.456, 0.406], device=video_tensor.device, dtype=resized_video.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=video_tensor.device, dtype=resized_video.dtype).view(1, 3, 1, 1)
        normalized_video = (resized_video - mean) / std

        # 5. Reshape back to what InternVideo2 expects for its forward pass.
        # [B*T, C, H, W] -> [B, T, C, H, W]
        final_video = normalized_video.view(b, self.num_frames, c, self.target_size[0], self.target_size[1])

        return final_video

    def forward(self, pred_pixel: torch.Tensor, target_pixel: torch.Tensor) -> torch.Tensor:
        """
        Computes the perceptual loss.
        Args:
            pred_pixel (torch.Tensor): The predicted video tensor from the VAE decoder, shape [B, C, T, H, W], range [-1, 1].
            target_pixel (torch.Tensor): The original video tensor without lora, shape [B, C, T, H, W], range [-1, 1].
        Returns:
            torch.Tensor: The calculated feature loss.
        """

        # Preprocess both predicted and target videos
        # assert pred_pixel.requires_grad, "pred_pixel lost grad tracking! (before checkpoint)"
        pred_pixel_processed = self._preprocess_video(pred_pixel)

        # Handle cases where preprocessing might fail (e.g., not enough frames)
        if pred_pixel_processed is None:
            return torch.tensor(0.0, device=pred_pixel.device, requires_grad=True)
        
        with torch.no_grad(): # Target doesn't need gradients
            target_pixel_processed = self._preprocess_video(target_pixel)
            if target_pixel_processed is None:
                return torch.tensor(0.0, device=pred_pixel.device, requires_grad=True)

        # Extract visual features
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # assert pred_pixel_processed.requires_grad, "pred_pixel_processed lost grad tracking! (before checkpoint)"
            pred_feat = torch.utils.checkpoint.checkpoint(lambda x: self.intern_model.get_vid_feat(x), pred_pixel_processed, use_reentrant=False)
        
        with torch.no_grad():
            target_feat = self.intern_model.get_vid_feat(target_pixel_processed)
        
        # Editing direction guidance
        editing_direction = pred_feat - target_feat

        # Compute the loss. Minimize negative cosine similarity, both shape is (1, 512)
        # normalize first
        editing_direction = torch.nn.functional.normalize(editing_direction, p=2, dim=1, eps=1e-6)
        guidance_vector = torch.nn.functional.normalize(self.guidance_vector.to(editing_direction.device), p=2, dim=1, eps=1e-6)

        one = torch.tensor(1.0, device=editing_direction.device, dtype=editing_direction.dtype)
        loss_per_element = -self.loss_fn(editing_direction, guidance_vector) + one

        regularization_loss_weight = 0.0
        regularization_loss_per_element = -self.loss_fn(pred_feat - self.guidance_concept_vector, target_feat - self.guidance_concept_vector) + one

        # Average over all dimensions except the batch dimension
        if loss_per_element.ndim > 1:
            loss = loss_per_element.mean(dim=tuple(range(1, loss_per_element.ndim))) + regularization_loss_weight * regularization_loss_per_element.mean(dim=tuple(range(1, regularization_loss_per_element.ndim)))
        else:
            loss = loss_per_element + regularization_loss_weight * regularization_loss_per_element

        return loss