import torch
import torch.nn as nn
import torch.nn.functional as F

class SaliencyTargetedPerturbationModule(nn.Module):
    """
    Refined Perturbation & Token Reduction Module:
    - Uses enhanced query-aware cosine similarity with aggressive logit sharpening 
      to eliminate background noise and tightly ground target objects.
    """
    def __init__(self, embed_dim=768, projection_dim=512, target_ratio=0.5, tau=1.0, use_query=False, noise_std=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.target_ratio = target_ratio  
        self.tau = tau
        self.use_query = use_query
        self.noise_std = noise_std  # Standard deviation for Gaussian noise

    def forward(
        self, 
        V: torch.Tensor, 
        text_embeds_projected: torch.Tensor = None, 
        visual_projection: nn.Linear = None
    ):
        B, N, D = V.shape
        
        # 1. Separate CLS token from spatial patches
        cls_token = V[:, :1, :]          # Shape: [B, 1, D]
        spatial_tokens = V[:, 1:, :]     # Shape: [B, 49, D]
        num_spatial = N - 1

        # 2. Sharpened Query-Aware Saliency Computation
        if self.use_query and text_embeds_projected is not None and visual_projection is not None:
            spatial_projected = visual_projection(spatial_tokens)
            v_norm = F.normalize(spatial_projected, p=2, dim=-1)
            t_norm = F.normalize(text_embeds_projected, p=2, dim=-1).unsqueeze(1)
            
            # Compute cosine similarity and apply a much higher scaling factor (40.0) for sharp contrast
            raw_similarity = (v_norm * t_norm).sum(dim=-1) # [B, num_spatial]
            
            # Contrast normalization: subtract mean across patches to suppress background clutter
            sim_mean = raw_similarity.mean(dim=-1, keepdim=True)
            sim_std = raw_similarity.std(dim=-1, keepdim=True) + 1e-6
            saliency_logits = ((raw_similarity - sim_mean) / sim_std) * 40.0
        else:
            spatial_mean = spatial_tokens.mean(dim=1, keepdim=True)
            saliency_logits = torch.norm(spatial_tokens - spatial_mean, p=2, dim=-1)

        saliency_scores = F.softmax(saliency_logits / self.tau, dim=-1) # [B, num_spatial]
        
        # 3. Compute Binary Saliency Mask and Top-k Indices
        k = max(1, int(num_spatial * self.target_ratio))
        topk_vals, topk_indices = torch.topk(saliency_logits, k=k, dim=1)
        
        binary_mask = torch.zeros_like(saliency_logits)
        binary_mask.scatter_(1, topk_indices, 1.0) # [B, num_spatial]

        # 4. Simultaneously Compute Corrupted Stream (Saliency-Targeted Gaussian Noise)
        gaussian_noise = torch.randn_like(spatial_tokens) * self.noise_std
        saliency_weights_expanded = saliency_scores.unsqueeze(-1)
        spatial_corrupted = spatial_tokens + (gaussian_noise * saliency_weights_expanded)
        V_corrupted = torch.cat([cls_token, spatial_corrupted], dim=1) # [B, 50, 768]

        # 5. Simultaneously Compute Reduced Stream (True Token Reduction)
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        reduced_spatial = torch.gather(spatial_tokens, dim=1, index=gather_indices)
        V_reduced = torch.cat([cls_token, reduced_spatial], dim=1) # [B, k+1, 768]

        # Clean Intact Stream
        V_intact = torch.cat([cls_token, spatial_tokens], dim=1) # [B, 50, 768]

        return {
            "V_intact": V_intact,
            "V_corrupted": V_corrupted,
            "V_reduced": V_reduced,
            "saliency_scores": saliency_scores,
            "saliency_mask": binary_mask.unsqueeze(-1),
            "topk_indices": topk_indices
        }
