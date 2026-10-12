import torch
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
from transformers import CLIPImageProcessor

# Ensure root directory is in path so 'solution' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution.dual_stream_builder import Member1DualStreamPipeline

# ==========================================
# REVIEW CONFIGURATION:
# ==========================================
TARGET_IMAGE_NAME = "image3.jpg"  
CUSTOM_QUERY = "The red car parked on the street"  
# ==========================================

images_dir = "images"

def verify_and_highlight_patch():
    print("================================================================")
    print(" RQ3 — Saliency Patch Visual Highlighting & Verification       ")
    print("================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Member1DualStreamPipeline().to(device)
    pipeline.eval()
    
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 1. Load target image
    image_path = os.path.join(images_dir, TARGET_IMAGE_NAME)
    
    if os.path.exists(image_path):
        print(f"[+] Successfully loaded target image : {image_path}")
        img = Image.open(image_path).convert("RGB")
    else:
        print(f"[!] Warning: '{TARGET_IMAGE_NAME}' not found. Using fallback gradient image...")
        img = Image.new("RGB", (224, 224), color=(200, 100, 50))

    # Resize image to CLIP standard 224x224 for precise 7x7 patch mapping
    img_resized = img.resize((224, 224))
    pixel_tensor = processor(images=img_resized, return_tensors="pt")["pixel_values"].to(device)

    print(f"[+] Active Text Query        : \"{CUSTOM_QUERY}\"")

    # 2. Run Pipeline Forward Pass
    with torch.no_grad():
        outputs = pipeline(pixel_tensor, text_prompts=[CUSTOM_QUERY])

    saliency_scores = outputs["saliency_scores"] # Shape: [1, 50]
    
    # 3. Exclude CLS token and map to 7x7 spatial grid
    token_features = outputs["V_intact"]
    cls_token, spatial_grid = pipeline.tokens_to_spatial_grid(token_features)
    
    patch_saliency = saliency_scores[0, 1:] # Shape: [49]
    heatmap_7x7 = patch_saliency.reshape(7, 7).cpu().numpy()

    # 4. Locate the peak activation patch
    max_idx = torch.argmax(patch_saliency).item()
    grid_r, grid_c = divmod(max_idx, 7)
    
    # Calculate pixel bounding box for the 32x32 patch square (since 224 / 7 = 32)
    left = grid_c * 32
    top = grid_r * 32
    right = (grid_c + 1) * 32
    bottom = (grid_r + 1) * 32

    # 5. Draw a red bounding box around the winning square patch on the image
    draw_img = img_resized.copy()
    draw = ImageDraw.Draw(draw_img)
    # Draw a thick red outline rectangle
    draw.rectangle([left, top, right, bottom], outline="red", width=4)
    
    output_path = os.path.join(images_dir, "selected_patch_highlight.jpg")
    draw_img.save(output_path)

    print("\n[Verification Result]")
    print(f"• Peak Saliency Token Index : Patch #{max_idx + 1}")
    print(f"• Peak Spatial Coordinates  : Row {grid_r + 1}, Column {grid_c + 1} (on 7x7 grid)")
    print(f"• Pixel Bounding Box        : [Left: {left}, Top: {top}, Right: {right}, Bottom: {bottom}]")
    print(f"• Maximum Relevance Score   : {patch_saliency[max_idx].item():.4f}")
    print(f"[SUCCESS] Highlighted image saved to: {output_path}")

if __name__ == "__main__":
    verify_and_highlight_patch()
