import torch
from PIL import Image
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution.vision_pipeline import RQ3CompleteSolutionPipeline

def run_solution_test():
    print("==================================================")
    print(" RQ3 — Running Isolated Solution Pipeline         ")
    print("==================================================")

    # 1. Instantiate solution pipeline
    solution_pipeline = RQ3CompleteSolutionPipeline(model_name="openai/clip-vit-base-patch32")
    solution_pipeline.eval()

    # 2. Create a simulated real input image (RGB 224x224)
    dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

    # 3. Preprocess using the encoder's processor
    inputs = solution_pipeline.processor(images=dummy_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    # 4. Execute forward pass
    with torch.no_grad():
        outputs = solution_pipeline(pixel_values)

    # 5. Verify outputs
    print("[+] Solution Pipeline Executed Successfully!")
    print(f"    - Input Pixel Shape      : {pixel_values.shape}")
    print(f"    - Output V_intact Shape  : {outputs['V_intact'].shape}")
    print(f"    - Output V_corrupted Shape : {outputs['V_corrupted'].shape}")
    print(f"    - Soft Saliency Mask Shape: {outputs['saliency_mask'].shape}")
    print(f"    - NaN-Free Verification  : {not torch.isnan(outputs['V_corrupted']).any().item()}")

if __name__ == "__main__":
    run_solution_test()
