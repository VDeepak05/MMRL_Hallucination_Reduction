import torch
from PIL import Image
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution.dual_stream_builder import Member1DualStreamPipeline

def test_with_real_image():
    image_path = "images/image3.jpg"
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at {image_path}. Please ensure image3.jpg is in the images/ folder.")
        return

    print(f"[*] Loading real test image from {image_path}...")
    image = Image.open(image_path).convert("RGB")

    print("[*] Initializing Member1DualStreamPipeline with query support...")
    pipeline = Member1DualStreamPipeline(use_query=True)
    pipeline.eval()

    # Preprocess image using the pipeline's built-in processor
    inputs = pipeline.processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    # Define a target test query
    query = ["The red car parked on the street"]
    print(f"[*] Running forward pass with query: '{query[0]}'")

    with torch.no_grad():
        outputs = pipeline(pixel_values=pixel_values, text_prompts=query, hard_reduction=True)

    v_intact = outputs["V_intact"]
    v_reduced = outputs["V_reduced"]
    saliency_scores = outputs["saliency_scores"]
    saliency_mask = outputs["saliency_mask"]

    print("\n[+] Real Image Pipeline Execution Successful!")
    print(f"    - Input Image Shape : {pixel_values.shape}")
    print(f"    - V_intact shape    : {v_intact.shape} (Expected: [1, 50, 768])")
    print(f"    - V_reduced shape   : {v_reduced.shape} (Expected: [1, 25, 768] for 50% ratio)")
    print(f"    - Saliency scores   : {saliency_scores.shape} (Min: {saliency_scores.min():.4f}, Max: {saliency_scores.max():.4f})")
    
    # Find peak saliency spatial token
    spatial_scores = saliency_scores[:, 1:] # Exclude CLS token dummy score
    peak_token_idx = spatial_scores.argmax(dim=-1).item()
    print(f"    - Peak Saliency Patch Index (Spatial 0-48): #{peak_token_idx}")
    print("\n[🎉] Real-world image verification completed successfully!")

if __name__ == "__main__":
    test_with_real_image()
