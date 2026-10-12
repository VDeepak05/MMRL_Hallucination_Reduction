import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution.dual_stream_builder import Member1DualStreamPipeline
from PIL import Image

def evaluate_pipeline_metrics():
    image_path = "images/image3.jpg"
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at {image_path}.")
        return

    image = Image.open(image_path).convert("RGB")
    pipeline = Member1DualStreamPipeline(use_query=True)
    pipeline.eval()

    inputs = pipeline.processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    query = ["a red car"]
    
    with torch.no_grad():
        outputs = pipeline(pixel_values=pixel_values, text_prompts=query)

    saliency_scores = outputs["saliency_scores"][0] # Shape: [49]
    topk_indices = outputs["topk_indices"][0]       # Shape: [k]

    # Compute Concentration Metrics
    total_score_sum = saliency_scores.sum().item()
    topk_score_sum = saliency_scores[topk_indices].sum().item()
    concentration_ratio = (topk_score_sum / total_score_sum) * 100

    print("\n" + "="*50)
    print("📊 QUANTITATIVE GROUNDING METRICS REPORT")
    print("="*50)
    print(f"• Target Query                  : '{query[0]}'")
    print(f"• Total Spatial Patches         : {len(saliency_scores)}")
    print(f"• Retained Top-k Tokens (k)     : {len(topk_indices)}")
    print(f"• Saliency Concentration (Top-k): {concentration_ratio:.2f}% of total score")
    print(f"• Peak Patch Score              : {saliency_scores.max().item():.4f}")
    print(f"• Minimum Patch Score           : {saliency_scores.min().item():.4f}")
    print("="*50)
    print("[+] Metrics evaluation completed successfully!")

if __name__ == "__main__":
    evaluate_pipeline_metrics()
