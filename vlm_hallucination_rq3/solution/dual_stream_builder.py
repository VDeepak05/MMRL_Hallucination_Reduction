import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.perturbation_improved import SaliencyTargetedPerturbationModule
from transformers import CLIPModel, CLIPProcessor

class Member1DualStreamPipeline(nn.Module):
    """
    Member 1 Evidence Module Pipeline:
    - Computes clean (V_intact), corrupted (V_corrupted), reduced (V_reduced),
      saliency scores, and binary saliency masks simultaneously in every forward pass.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", use_query=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Explicitly freeze vision encoder parameters
        for param in self.clip.vision_model.parameters():
            param.requires_grad = False
        self.clip.vision_model.eval()

        embed_dim = self.clip.config.vision_config.hidden_size # 768
        projection_dim = self.clip.config.projection_dim        # 512
        
        self.token_pruner = SaliencyTargetedPerturbationModule(
            embed_dim=embed_dim, 
            projection_dim=projection_dim,
            use_query=use_query
        )

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        text_prompts: list = None
    ):
        self.clip.vision_model.eval()
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            patch_tokens = vision_outputs.last_hidden_state  # [B, 50, 768]

        text_embeds_projected = None
        if text_prompts is not None and len(text_prompts) > 0:
            text_inputs = self.processor(
                text=text_prompts, padding=True, return_tensors="pt"
            ).to(pixel_values.device)
            text_outputs = self.clip.text_model(**text_inputs)
            text_embeds_projected = self.clip.text_projection(text_outputs.pooler_output)

        prune_output = self.token_pruner(
            patch_tokens, 
            text_embeds_projected=text_embeds_projected,
            visual_projection=self.clip.visual_projection
        )

        return {
            "V_intact": prune_output["V_intact"],
            "V_corrupted": prune_output["V_corrupted"],
            "V_reduced": prune_output["V_reduced"],
            "saliency_scores": prune_output["saliency_scores"],
            "saliency_mask": prune_output["saliency_mask"],
            "topk_indices": prune_output["topk_indices"]
        }
