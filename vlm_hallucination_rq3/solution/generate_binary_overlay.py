import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution.dual_stream_builder import Member1DualStreamPipeline

def generate_binary_mask_overlay():
    image_path = "images/image3.jpg"
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at {image_path}.")
        return

    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    pipeline = Member1DualStreamPipeline(use_query=True)
    pipeline.eval()

    inputs = pipeline.processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    query = ["The red car parked on the street"]
    
    with torch.no_grad():
        outputs = pipeline(pixel_values=pixel_values, text_prompts=query)

    # Get binary saliency mask shape [1, 49, 1]
    binary_mask = outputs["saliency_mask"][0].squeeze(-1).reshape(7, 7).cpu().numpy()

    # Plot original image alongside the binary token selection mask
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Overlay binary mask (1 for retained top-k evidence patches, 0 for dropped background)
    axes[1].imshow(image)
    axes[1].imshow(
        binary_mask, 
        cmap='gray', 
        alpha=0.6, 
        extent=[0, W, H, 0], 
        interpolation='nearest'
    )
    axes[1].set_title(f"Top-k Evidence Mask (Retained Patches)", fontsize=10, fontweight='bold')
    axes[1].axis('off')

    output_path = "solution/binary_mask_overlay.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()

    print(f"\n[+] Binary evidence mask overlay generated successfully!")
    print(f"• Saved to : {output_path}")

if __name__ == "__main__":
    generate_binary_mask_overlay()
