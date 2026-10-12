import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution.dual_stream_builder import Member1DualStreamPipeline

def generate_saliency_overlay():
    image_path = "images/image3.jpg"
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at {image_path}. Please place image3.jpg in images/ folder.")
        return

    # Load original image
    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    # Initialize pipeline
    pipeline = Member1DualStreamPipeline(use_query=True)
    pipeline.eval()

    inputs = pipeline.processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    # Target query to ground
    query = ["The red car parked on the street"]
    
    with torch.no_grad():
        outputs = pipeline(pixel_values=pixel_values, text_prompts=query)

    saliency_scores = outputs["saliency_scores"] # Shape: [1, 49]
    grid_scores = saliency_scores[0].reshape(7, 7).cpu().numpy()

    # Normalize grid scores between 0 and 1 for heatmap visualization
    g_min, g_max = grid_scores.min(), grid_scores.max()
    grid_normalized = (grid_scores - g_min) / (g_max - g_min + 1e-8)

    # Plot original image and overlay heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    
    # Overlay the 7x7 saliency map resized smoothly using bilinear interpolation
    ax.imshow(
        grid_normalized, 
        cmap='jet', 
        alpha=0.55, 
        extent=[0, W, H, 0], 
        interpolation='bilinear'
    )
    ax.axis('off')
    ax.set_title(f"Query Grounding: '{query[0]}'", fontsize=10, fontweight='bold', pad=10)

    # Save output overlay image
    output_path = "solution/saliency_overlay.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()

    print(f"\n[+] Visual verification overlay successfully generated!")
    print(f"• Saved to : {output_path}")
    print(f"• Action   : Open '{output_path}' to visually confirm if the red overlay highlights the car.")

if __name__ == "__main__":
    generate_saliency_overlay()
