import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveTokenSelector(nn.Module):
    """
    RQ3 Section 4.2: Adaptive Token Selection via Gumbel-Softmax & Straight-Through Estimator.
    Computes a question/context-aware binary mask with a sparsity/budget loss.
    """
    def __init__(self, embed_dim=768, temperature=1.0):
        super().__init__()
        # Lightweight gating head to score token importance (binary drop/keep)
        self.gate_fc = nn.Linear(embed_dim, 2)
        self.temperature = temperature

    def forward(self, V: torch.Tensor, hard=True):
        """
        Args:
            V: Visual tokens [B, N, D] from the frozen vision encoder.
            hard: If True, uses Straight-Through Estimator (STE) for discrete 0/1 masks.
        Returns:
            dict containing selected tokens, binary keep mask, and sparsity ratio.
        """
        B, N, D = V.shape
        
        # 1. Compute selection logits for each patch [B, N, 2]
        logits = self.gate_fc(V)
        
        # 2. Differentiable Gumbel-Softmax Sampling
        gumbel_soft = F.gumbel_softmax(logits, tau=self.temperature, hard=hard, dim=-1)
        
        # Index 1 corresponds to "keep", Index 0 corresponds to "drop"
        keep_mask = gumbel_soft[..., 1:2] # [B, N, 1]
        
        # 3. Apply mask to obtain reduced visual tokens
        V_selected = V * keep_mask
        
        # 4. Compute sparsity ratio (fraction of tokens retained)
        sparsity_ratio = keep_mask.mean()
        
        return {
            "V_selected": V_selected,
            "keep_mask": keep_mask.squeeze(-1), # [B, N]
            "sparsity_ratio": sparsity_ratio
        }

    def compute_budget_loss(self, keep_mask, target_budget=0.5, lambda_budget=1.0):
        """
        Sparsity/budget loss to ensure token count shrinks for simple inputs 
        and expands for detail-heavy inputs, matching ATP-LLaVA style budgeting.
        """
        mean_kept = keep_mask.mean()
        budget_loss = lambda_budget * torch.pow(mean_kept - target_budget, 2)
        return budget_loss
