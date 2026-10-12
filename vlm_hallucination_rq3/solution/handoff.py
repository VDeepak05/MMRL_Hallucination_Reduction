import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution.dual_stream_builder import Member1DualStreamPipeline

class Member1HandoffInterface:
    """
    Official Handoff Interface from Member 1 (Visual Evidence & Perturbation)
    to Member 2 (Frozen VLM Decoder & Contrastive Calibration).
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.pipeline = Member1DualStreamPipeline(model_name=model_name)
        self.pipeline.eval()

    def generate_member2_payload(self, pixel_values, text_prompt=None):
        """
        Processes images and bundles everything Member 2 needs into a dictionary payload.
        
        Args:
            pixel_values: Tensor [B, C, H, W] of preprocessed images.
            text_prompt: Optional text strings or tokens if needed by Member 2.
        Returns:
            dict: The exact payload contract for Member 2.
        """
        with torch.no_grad():
            outputs = self.pipeline(pixel_values)

        # Standardized Handoff Contract matching the flowchart
        payload = {
            "dual_stream": outputs["dual_stream"],     # Shape: [B, 2, N, D] -> Index 0: V_intact, Index 1: V_corrupted
            "V_intact": outputs["V_intact"],           # Shape: [B, N, D]
            "V_corrupted": outputs["V_corrupted"],     # Shape: [B, N, D]
            "saliency_mask": outputs["saliency_mask"], # Shape: [B, N] (Useful for verification)
            "metadata": {
                "source_module": "Member1_VisualPerturbation",
                "status": "Ready for VLM Decoder",
                "batch_size": pixel_values.shape[0]
            }
        }
        return payload
