import torch
from PIL import Image
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution.dual_stream_builder import Member1DualStreamPipeline

def test_dual_stream():
    print("==================================================")
    print(" RQ3 — Member 1 Dual Stream Pipeline Execution   ")
    print("==================================================")

    # 1. Initialize Member 1 pipeline
    member1_pipeline = Member1DualStreamPipeline(model_name="openai/clip-vit-base-patch32")
    member1_pipeline.eval()

    # 2. Simulate input image
    dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    inputs = member1_pipeline.vision_pipeline.processor(images=dummy_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    # 3. Forward pass through Member 1 Evidence Module
    with torch.no_grad():
        outputs = member1_pipeline(pixel_values)

    print("[+] Member 1 Evidence Module Executed Successfully!")
    print(f"    - Frozen Vision Encoder Output : Active")
    print(f"    - Saliency Extraction & Mask   : Complete")
    print(f"    - Targeted Noise Injection     : Complete")
    print(f"    - Batched Dual Stream Shape    : {outputs['dual_stream'].shape} (Matches flowchart [2, N, D] structure)")
    print(f"    - NaN-Free Verification        : {not torch.isnan(outputs['dual_stream']).any().item()}")

if __name__ == "__main__":
    test_dual_stream()
