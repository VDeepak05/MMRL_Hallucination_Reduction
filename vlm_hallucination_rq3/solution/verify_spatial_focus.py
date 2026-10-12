import torch
from PIL import Image
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution.dual_stream_builder import Member1DualStreamPipeline

def verify_model_focus():
    image_path = "images/image3.jpg"
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at {image_path}. Please place image3.jpg in images/ folder.")
        return

    image = Image.open(image_path).convert("RGB")
    pipeline = Member1DualStreamPipeline(use_query=True)
    pipeline.eval()

    inputs = pipeline.processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    # Test with a specific query
    query = ["The red car parked on the street"]
    
    with torch.no_grad():
        # Pipeline now computes all streams simultaneously without hard_reduction flag
        outputs = pipeline(pixel_values=pixel_values, text_prompts=query)

    saliency_scores = outputs["saliency_scores"] # Shape: [1, 49] for spatial patches
    spatial_scores = saliency_scores[0] # Shape: [49]
    
    # Reshape into 7x7 grid to inspect spatial distribution
    grid_scores = spatial_scores.reshape(7, 7)
    
    print("\n[+] 7x7 Spatial Saliency Heatmap Matrix (Query Alignment):")
    print("-" * 55)
    for row in grid_scores:
        print(" | ".join([f"{val:.4f}" for val in row]))
    print("-" * 55)

    peak_idx = spatial_scores.argmax().item()
    peak_row = peak_idx // 7
    peak_col = peak_idx % 7
    max_score = spatial_scores[peak_idx].item()

    print(f"• Peak Saliency Token Index : Patch #{peak_idx} (0-indexed spatial)")
    print(f"• Mapped Grid Coordinate  : Row {peak_row + 1}, Column {peak_col + 1}")
    print(f"• Maximum Relevance Score : {max_score:.4f}")

    assert max_score > 0.01, "Warning: Maximum saliency score is too flat."
    print("\n[🎉] Spatial focus verification check passed successfully!")

if __name__ == "__main__":
    verify_model_focus()
