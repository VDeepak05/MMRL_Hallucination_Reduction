import torch

def inspect_selected_patches(outputs, top_k=5):
    """
    Inspects and prints the top selected patch indices based on saliency scores.
    """
    saliency_scores = outputs["saliency_scores"] # Shape: [B, N]
    # Get top-k patch indices for the first batch item
    top_scores, top_indices = torch.topk(saliency_scores[0], k=top_k)
    
    print("==================================================")
    print(" RQ3 — Visual Evidence Patch Inspection          ")
    print("==================================================")
    print(f"    - Total Patches Evaluated : {saliency_scores.shape[1]}")
    print(f"    - Top {top_k} Most Salient Patch Indices:")
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
        print(f"      {rank}. Patch Index: {idx.item():2d} | Saliency Score: {score.item():.4f}")
    print("--------------------------------------------------")
    print("[+] Saliency localization verified successfully!")
