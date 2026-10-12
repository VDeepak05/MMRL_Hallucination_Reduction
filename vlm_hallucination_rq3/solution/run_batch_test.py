import torch
import os
import sys
import time
from PIL import Image
from transformers import CLIPImageProcessor

# Ensure root directory is in path so 'solution' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution.dual_stream_builder import Member1DualStreamPipeline

def run_query_aware_batch_evaluation(image_dir="images"):
    print("================================================================")
    print(" RQ3 — Query-Aware Multi-Image Batch Pipeline Test (10 Samples) ")
    print("================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Initializing query-aware pipeline on device: {device}")
    
    pipeline = Member1DualStreamPipeline().to(device)
    pipeline.eval()
    
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Sample text queries corresponding to each image in the batch
    sample_queries = [
        "What is the main subject in this image?",
        "Identify key objects and regions.",
        "Locate visual details for verification.",
        "Detect potential hallucination targets.",
        "Analyze background vs foreground elements.",
        "Examine features relevant to the query.",
        "Isolate high-saliency visual regions.",
        "Perform query-guided patch retention.",
        "Measure cross-modal alignment.",
        "Finalize dual-stream evidence payload."
    ]

    # Gather images or fallback to synthetic tensors
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    image_paths = []
    if os.path.exists(image_dir):
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
    
    processed_tensors = []
    if len(image_paths) > 0:
        print(f"[+] Found {len(image_paths)} images in '{image_dir}/'. Loading up to 10...")
        for path in image_paths[:10]:
            try:
                img = Image.open(path).convert("RGB")
                pixel_tensor = processor(images=img, return_tensors="pt")["pixel_values"]
                processed_tensors.append(pixel_tensor)
            except Exception as e:
                print(f"[!] Skipping {path}: {e}")
    
    if len(processed_tensors) == 0:
        print(f"[!] No images found in '{image_dir}/'. Generating synthetic image batch [10, 3, 224, 224]...")
        batch_tensor = torch.randn(10, 3, 224, 224, device=device)
    else:
        batch_tensor = torch.cat(processed_tensors, dim=0).to(device)
        print(f"[+] Successfully loaded {batch_tensor.shape[0]} real images.")

    # Match queries to batch size
    active_queries = sample_queries[:batch_tensor.shape[0]]

    # Execute Forward Pass with text queries
    print("[+] Executing query-aware pipeline forward pass...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = pipeline(batch_tensor, text_prompts=active_queries)
        
    elapsed_time = (time.time() - start_time) * 1000
    dual_stream = outputs["dual_stream"]
    saliency_scores = outputs["saliency_scores"]
    batch_size = batch_tensor.shape[0]

    # Validation Checks
    is_nan_free = not torch.isnan(dual_stream).any().item()
    expected_shape = torch.Size([batch_size, 2, 50, 768])
    shape_matches = (dual_stream.shape == expected_shape)

    print("-" * 64)
    print(f"{'Metric / Check':<35} | {'Status / Value':<20}")
    print("-" * 64)
    print(f"{'Images & Queries Processed':<35} | {batch_size:<20}")
    print(f"{'Output Dual-Stream Shape':<35} | {str(tuple(dual_stream.shape)):<20}")
    print(f"{'Query-Aware Saliency Shape':<35} | {str(tuple(saliency_scores.shape)):<20}")
    print(f"{'Shape Compliance':<35} | {str(shape_matches):<20}")
    print(f"{'NaN-Free Stability':<35} | {str(is_nan_free):<20}")
    print(f"{'Total Batch Latency':<35} | {elapsed_time:.2f} ms")
    print(f"{'Per-Image Latency':<35} | {elapsed_time/batch_size:.2f} ms")
    print("-" * 64)

    if is_nan_free and shape_matches:
        print("[SUCCESS] Query-aware batch evaluation passed all checks!")
    else:
        print("[WARNING] Validation check failed.")

if __name__ == "__main__":
    run_query_aware_batch_evaluation()
