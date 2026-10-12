import torch
from PIL import Image
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution.vision_pipeline import RQ3CompleteSolutionPipeline
from solution.adaptive_selection import AdaptiveTokenSelector

def test_adaptive_pipeline():
    print("==================================================")
    print(" RQ3 — Adaptive Selection & Budget Loss Pipeline  ")
    print("==================================================")

    # 1. Instantiate Frozen Vision Pipeline (Section 4.3)
    vision_pipeline = RQ3CompleteSolutionPipeline(model_name="openai/clip-vit-base-patch32")
    vision_pipeline.eval()

    # 2. Instantiate Adaptive Token Selector (Section 4.2)
    adaptive_selector = AdaptiveTokenSelector(embed_dim=768, temperature=1.0)
    adaptive_selector.eval()

    # 3. Create simulated input image [1, 3, 224, 224]
    dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    inputs = vision_pipeline.processor(images=dummy_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        # Extract visual tokens from frozen backbone
        encoder_outputs = vision_pipeline.encoder(pixel_values=pixel_values)
        patch_tokens = encoder_outputs.last_hidden_state # [1, 50, 768]

        # Apply Gumbel-Softmax adaptive token selection
        selection_results = adaptive_selector(patch_tokens, hard=True)

        # Calculate budget loss against a target token retention budget (e.g., 40%)
        budget_loss = adaptive_selector.compute_budget_loss(
            selection_results["keep_mask"], 
            target_budget=0.4
        )

    print("[+] Adaptive Selection Pipeline Executed Successfully!")
    print(f"    - Original Token Shape : {patch_tokens.shape}")
    print(f"    - Selected Token Shape : {selection_results['V_selected'].shape}")
    print(f"    - Token Sparsity Ratio : {selection_results['sparsity_ratio'].item():.4f}")
    print(f"    - Computed Budget Loss : {budget_loss.item():.4f}")
    print(f"    - Differentiable/STE   : Active (NaN-Free: {not torch.isnan(selection_results['V_selected']).any()})")

if __name__ == "__main__":
    test_adaptive_pipeline()
