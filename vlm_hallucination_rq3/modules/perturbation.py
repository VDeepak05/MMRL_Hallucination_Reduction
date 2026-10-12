import torch
import torch.nn as nn

class VisualPerturbationAndAdaptiveSaliency(nn.Module):
    """
    RQ3 — Member 1: Visual Evidence Extraction & Counterfactual Generation Module
    Based on Visual Contrastive Decoding (Leng et al., CVPR 2024).
    """
    def __init__(self, top_k_ratio=0.3, temperature=1.0, noise_std=0.3):
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.tau = temperature
        self.sigma = noise_std

    def forward(self, V: torch.Tensor):
        """
        Args:
            V: Tensor of shape [B, N, D] from a frozen vision encoder (e.g., SigLIP/CLIP).
        Returns:
            dict containing intact features, corrupted features, saliency mask, and scores.
        """
        B, N, D = V.shape
        device = V.device

        # Step 1: Global Visual Context [B, 1, D]
        v_bar = V.mean(dim=1, keepdim=True)

        # Step 2: Patch-Level Saliency via Feature Divergence [B, N]
        distances = torch.norm(V - v_bar, p=2, dim=-1) # [B, N]
        saliency_scores = torch.softmax(distances / self.tau, dim=-1) # [B, N]

        # Step 3: Top-K Masking for Foreground Objects [B, N]
        k = max(1, int(N * self.top_k_ratio))
        _, topk_indices = torch.topk(saliency_scores, k=k, dim=1)
        
        saliency_mask = torch.zeros(B, N, device=device, dtype=torch.float32)
        saliency_mask.scatter_(1, topk_indices, 1.0) # [B, N]

        # Step 4: Targeted Counterfactual Noise Injection [B, N, D]
        var_v = torch.var(V, dim=(1, 2), keepdim=True) # [B, 1, 1]
        noise_scale = self.sigma * torch.sqrt(var_v + 1e-8)
        delta = torch.randn_like(V) * noise_scale

        # Apply noise only where saliency_mask == 1
        mask_expanded = saliency_mask.unsqueeze(-1) # [B, N, 1]
        v_corrupted = V + mask_expanded * delta

        # Output Contract (Handoff payload)
        return {
            "V_intact": V,                     # Original visual features [B, N, D]
            "V_corrupted": v_corrupted,        # Counterfactual features with targeted noise [B, N, D]
            "saliency_mask": saliency_mask,    # Binary mask of perturbed foreground patches [B, N]
            "saliency_scores": saliency_scores # Patch-importance distribution [B, N]
        }