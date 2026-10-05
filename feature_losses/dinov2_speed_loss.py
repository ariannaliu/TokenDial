"""
DINOv2 Identity Loss
Preserves visual identity (person appearance, clothing, background) while allowing spatial/temporal changes.
Optimized for first-frame strict matching with patch-level and CLS token constraints.
"""

import torch
import torch.nn as nn
import os
import numpy as np
import cv2


class DINOv2IdentityLoss(nn.Module):
    """
    Identity-preserving loss using DINOv2 features.
    Supports multiple modes for different use cases:
    - 'first_frame_strict': Patch-level + CLS token for first frame (strongest, best for exact match)
    - 'first_frame_cls_only': Only CLS token for first frame (simpler, still strong)
    - 'temporal_pooling': Average features over time with statistics (position-invariant)
    """

    def __init__(
        self,
        model_name: str = 'dinov2_vitl14',
        mode: str = 'first_frame_strict',
        device: str = 'cuda',
    ):
        """
        Args:
            model_name: DINOv2 model variant ('dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
            mode: Loss computation mode
                - 'first_frame_strict': Patch-level + CLS for first frame (strongest)
                - 'first_frame_cls_only': Only CLS token for first frame
                - 'temporal_pooling': Statistics aggregation across all frames (position-invariant)
            device: Device to run the model on
        """
        super().__init__()

        self.mode = mode
        self.device = torch.device(device)

        # Load DINOv2 model
        print(f"Loading DINOv2 model: {model_name} for identity loss...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dino_model.eval()
        self.dino_model.to(self.device)

        # Freeze all parameters
        for param in self.dino_model.parameters():
            param.requires_grad = False

        print(f"DINOv2 Identity Loss initialized with mode='{mode}', model='{model_name}'")

    def preprocess_images(self, images):
        """
        Preprocess images for DINOv2.
        Args:
            images: [B, C, H, W] or [B*F, C, H, W], range [-1, 1]
        Returns:
            Preprocessed images [B, C, 224, 224] or [B*F, C, 224, 224], range [0, 1]
        """
        # Resize to 224x224
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize from [-1, 1] to [0, 1]
        one = torch.tensor(1.0, dtype=images.dtype, device=images.device)
        two = torch.tensor(2.0, dtype=images.dtype, device=images.device)
        images = (images.clamp(-1, 1) + one) / two

        return images


    def compute_feature_flow_lucas_kanade(self, feats, window_size=3, epsilon=1e-6):
        """
        Compute optical flow using Lucas-Kanade method on DINOv2 feature space.

        The multi-channel LK method solves:
            sum_d [Ix_d * u + Iy_d * v + It_d] = 0 (over all feature dimensions d)

        This gives us a least-squares solution for (u, v) using the structure tensor.

        Args:
            feats: [B, T, N, D] Normalized DINOv2 patch features
            window_size: Size of local window for spatial coherence (odd number)
            epsilon: Small value for numerical stability

        Returns:
            flow: [B, T-1, N, 2] Flow vectors (dx, dy) in grid units
        """
        B, T, N, D = feats.shape
        H_grid = int(N**0.5)
        W_grid = N // H_grid
        device = feats.device
        dtype = feats.dtype

        # Reshape to spatial grid: [B, T, H, W, D]
        feats_spatial = feats.view(B, T, H_grid, W_grid, D)

        # Pad for gradient computation (replicate padding to handle boundaries)
        # Reshape to 4D for padding (PyTorch replicate padding works best with 4D)
        # [B, T, H, W, D] -> [B*T, D, H, W]
        feats_4d = feats_spatial.permute(0, 1, 4, 2, 3).reshape(B * T, D, H_grid, W_grid)
        feats_4d_padded = torch.nn.functional.pad(feats_4d, (1, 1, 1, 1), mode='replicate')
        # Reshape back: [B*T, D, H+2, W+2] -> [B, T, H+2, W+2, D]
        feats_padded = feats_4d_padded.view(B, T, D, H_grid + 2, W_grid + 2).permute(0, 1, 3, 4, 2)

        # Compute spatial gradients using central differences
        # Ix: gradient in x (width) direction
        Ix = (feats_padded[:, :, 1:-1, 2:, :] - feats_padded[:, :, 1:-1, :-2, :]) / 2.0  # [B, T, H, W, D]
        # Iy: gradient in y (height) direction
        Iy = (feats_padded[:, :, 2:, 1:-1, :] - feats_padded[:, :, :-2, 1:-1, :]) / 2.0  # [B, T, H, W, D]

        # Temporal gradient: difference between consecutive frames
        It = feats_spatial[:, 1:] - feats_spatial[:, :-1]  # [B, T-1, H, W, D]

        # Use gradients from frame t (could also average with t+1 for better accuracy)
        Ix = Ix[:, :-1]  # [B, T-1, H, W, D]
        Iy = Iy[:, :-1]  # [B, T-1, H, W, D]

        # Build structure tensor (sum over feature dimension D for multi-channel LK)
        # For D-dimensional features, we solve the overdetermined system via normal equations
        Ixx = (Ix * Ix).sum(dim=-1)  # [B, T-1, H, W]
        Iyy = (Iy * Iy).sum(dim=-1)  # [B, T-1, H, W]
        Ixy = (Ix * Iy).sum(dim=-1)  # [B, T-1, H, W]
        Ixt = (Ix * It).sum(dim=-1)  # [B, T-1, H, W]
        Iyt = (Iy * It).sum(dim=-1)  # [B, T-1, H, W]

        # Apply local window averaging for spatial coherence constraint
        # This is the "window" in Lucas-Kanade that assumes neighbors have similar motion
        pad = window_size // 2
        kernel = torch.ones(1, 1, window_size, window_size, device=device, dtype=dtype)
        kernel = kernel / (window_size * window_size)

        def apply_window(x):
            """Apply averaging window via convolution."""
            x_flat = x.reshape(-1, 1, H_grid, W_grid)
            x_conv = torch.nn.functional.conv2d(x_flat, kernel, padding=pad)
            return x_conv.view(B, T-1, H_grid, W_grid)

        Ixx_w = apply_window(Ixx)
        Iyy_w = apply_window(Iyy)
        Ixy_w = apply_window(Ixy)
        Ixt_w = apply_window(Ixt)
        Iyt_w = apply_window(Iyt)

        # Solve 2x2 linear system using Cramer's rule:
        # [[Ixx, Ixy], [Ixy, Iyy]] * [u, v]^T = -[Ixt, Iyt]^T
        det = Ixx_w * Iyy_w - Ixy_w * Ixy_w + epsilon

        # Flow components
        u = -(Iyy_w * Ixt_w - Ixy_w * Iyt_w) / det  # x-direction flow
        v = -(Ixx_w * Iyt_w - Ixy_w * Ixt_w) / det  # y-direction flow

        # Stack flow: [B, T-1, H, W, 2]
        flow_spatial = torch.stack([u, v], dim=-1)

        # Reshape to match original format: [B, T-1, N, 2]
        flow = flow_spatial.view(B, T-1, N, 2)

        return flow


    def calculate_fast_motion_loss_scale(self, pred_pixels, stride=1, num_keyframes=16, fast_ratio=0.5):
        B, C, F, H, W = pred_pixels.shape

        if F < num_keyframes:
             return torch.zeros(B, device=pred_pixels.device, requires_grad=True)


        with torch.no_grad():
            t = 0
            idx_strided = torch.arange(t, t + num_keyframes * stride, stride, device=pred_pixels.device)
            # gt_video = x0_pixels[:, :, idx_strided, :, :] # [B, C, K, H, W]
            # gt_video = pred_pixels_wotoken[:, :, idx_strided, :, :] # [B, C, K, H, W]
            gt_video = pred_pixels.detach()[:, :, idx_strided, :, :] # [B, C, K, H, W]

            # Reshape [B, C, K, H, W] -> [B, K, C, H, W] -> [B*K, C, H, W]
            gt_flat = gt_video.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            gt_flat_preprocessed = self.preprocess_images(gt_flat)

            gt_feat_dict = self.dino_model.forward_features(gt_flat_preprocessed)
            gt_feat = gt_feat_dict['x_norm_patchtokens'] # [B*K, N, D]

            _, N, D = gt_feat.shape
            gt_feat = gt_feat.view(B, num_keyframes, N, D)
            flow_gt = self.compute_feature_flow_lucas_kanade(gt_feat)     #[B, K-1, N, 2]

            flow_gt_ref = flow_gt * fast_ratio


        # Target Flow: From pred_pixels (Prediction)
        pred_video = pred_pixels[:, :, idx_strided, :, :] # [B, C, K, H, W]

        # Reshape [B, C, K, H, W] -> [B, K, C, H, W] -> [B*K, C, H, W]
        pred_flat = pred_video.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        pred_flat_preprocessed = self.preprocess_images(pred_flat)

        pred_feat_dict = self.dino_model.forward_features(pred_flat_preprocessed)
        pred_feat = pred_feat_dict['x_norm_patchtokens']
        pred_feat = pred_feat.view(B, num_keyframes, N, D)

        flow_pred = self.compute_feature_flow_lucas_kanade(pred_feat)    #[B, K-1, N, 2]

        # 3. Loss
        loss = torch.nn.functional.mse_loss(flow_pred, flow_gt_ref.detach(), reduction='none').mean(dim=(1, 2, 3))
        return loss


    def forward(self, pred_pixels, pred_pixels_wotoken):
        """
        Compute identity-preserving loss.

        Args:
            pred_pixels: [B, C, F, H, W] predicted video with token modification
            pred_pixels_wotoken: [B, C, F, H, W] original video without token

        Returns:
            loss: [B] tensor - per-batch loss values
        """
        B, C, F, H, W = pred_pixels.shape

        if self.mode == 'first_frame_strict':
            # First frame: strictest constraint with patch-level + CLS
            first_frame_pred = pred_pixels[:, :, 0, :, :]  # [B, C, H, W]
            first_frame_target = pred_pixels_wotoken[:, :, 0, :, :]  # [B, C, H, W]

            # Preprocess
            first_frame_pred = self.preprocess_images(first_frame_pred)
            first_frame_target = self.preprocess_images(first_frame_target)

            # Extract features (target without grad)
            with torch.no_grad():
                feat_dict_target = self.dino_model.forward_features(first_frame_target)
                patch_tokens_target = feat_dict_target['x_norm_patchtokens']  # [B, N_patches, D]
                cls_target = feat_dict_target['x_norm_clstoken']  # [B, D]

            # Extract features (pred with grad)
            feat_dict_pred = self.dino_model.forward_features(first_frame_pred)
            patch_tokens_pred = feat_dict_pred['x_norm_patchtokens']  # [B, N_patches, D]
            cls_pred = feat_dict_pred['x_norm_clstoken']  # [B, D]

            # Patch-level loss: preserves spatial details, each patch location should match
            patch_loss = 1 - torch.nn.functional.cosine_similarity(
                patch_tokens_pred,
                patch_tokens_target,
                dim=-1
            )  # [B, N_patches]
            patch_loss = patch_loss.mean(dim=1)  # [B] - average over all patches
            patch_loss = patch_loss/2.0

            # CLS loss: global consistency
            cls_loss = 1 - torch.nn.functional.cosine_similarity(cls_pred, cls_target, dim=1)  # [B]
            cls_loss = cls_loss/2.0

            # Combined loss: patch-level more important for preserving details
            loss = 0.8 * patch_loss + 0.2 * cls_loss

            return loss

        elif self.mode == 'first_frame_cls_only':
            # First frame: simpler CLS-only constraint
            first_frame_pred = pred_pixels[:, :, 0, :, :]  # [B, C, H, W]
            first_frame_target = pred_pixels_wotoken[:, :, 0, :, :]  # [B, C, H, W]

            # Preprocess
            first_frame_pred = self.preprocess_images(first_frame_pred)
            first_frame_target = self.preprocess_images(first_frame_target)

            # Extract CLS features
            with torch.no_grad():
                feat_dict_target = self.dino_model.forward_features(first_frame_target)
                cls_target = feat_dict_target['x_norm_clstoken']  # [B, D]

            feat_dict_pred = self.dino_model.forward_features(first_frame_pred)
            cls_pred = feat_dict_pred['x_norm_clstoken']  # [B, D]

            # CLS loss only
            loss = 1 - torch.nn.functional.cosine_similarity(cls_pred, cls_target, dim=1)  # [B]

            return loss

        elif self.mode == 'temporal_pooling':
            # All frames: position-invariant with statistics aggregation
            # Use this for subsequent frames when object can be at different locations
            pred_reshaped = pred_pixels.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            target_reshaped = pred_pixels_wotoken.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)

            # Preprocess
            pred_reshaped = self.preprocess_images(pred_reshaped)
            target_reshaped = self.preprocess_images(target_reshaped)

            # Extract patch features
            with torch.no_grad():
                feat_dict_target = self.dino_model.forward_features(target_reshaped)
                patch_tokens_target = feat_dict_target['x_norm_patchtokens']  # [B*F, N_patches, D]

            feat_dict_pred = self.dino_model.forward_features(pred_reshaped)
            patch_tokens_pred = feat_dict_pred['x_norm_patchtokens']  # [B*F, N_patches, D]

            # Statistics aggregation: mean + std (position-invariant)
            feat_pred = torch.cat([
                patch_tokens_pred.mean(dim=1),  # [B*F, D] - average appearance
                patch_tokens_pred.std(dim=1)    # [B*F, D] - appearance distribution
            ], dim=-1)  # [B*F, 2*D]

            feat_target = torch.cat([
                patch_tokens_target.mean(dim=1),
                patch_tokens_target.std(dim=1)
            ], dim=-1)  # [B*F, 2*D]

            # Temporal pooling: average features over time
            feat_pred = feat_pred.view(B, F, -1).mean(dim=1)  # [B, 2*D]
            feat_target = feat_target.view(B, F, -1).mean(dim=1)  # [B, 2*D]

            # Cosine similarity loss
            loss = 1 - torch.nn.functional.cosine_similarity(feat_pred, feat_target, dim=1)  # [B]

            return loss

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose from: 'first_frame_strict', 'first_frame_cls_only', 'temporal_pooling'")


# Convenience function for common use cases
def create_dino_identity_loss(mode='first_frame_strict', model_name='dinov2_vitl14', device='cuda'):
    """
    Create a DINOv2 identity loss with common settings.

    Args:
        mode: 'first_frame_strict' (default, best for exact first frame match)
              'first_frame_cls_only' (simpler, CLS only)
              'temporal_pooling' (position-invariant for all frames)
        model_name: DINOv2 model variant
        device: Device to run on

    Returns:
        DINOv2IdentityLoss instance
    """
    return DINOv2IdentityLoss(
        model_name=model_name,
        mode=mode,
        device=device,
    )
