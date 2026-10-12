import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution.dual_stream_builder import Member1DualStreamPipeline

def test_pipeline():
    print("[*] Initializing Member1DualStreamPipeline...")
    pipeline = Member1DualStreamPipeline(use_query=True)
    
    # 1. Verify Encoder Freeze Status
    print("[*] Checking encoder parameter freeze status...")
    for name, param in pipeline.clip.vision_model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} is not frozen!"
    print("[+] Success: All vision encoder parameters have requires_grad=False.")

    # 2. Create Dummy Batch
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_queries = ["A red car parked on the street", "A dog running in the park"]

    # 3. Test Simultaneous Forward Pass
    print("[*] Testing simultaneous forward pass (Intact, Corrupted, Reduced, Scores, Masks)...")
    outputs = pipeline(pixel_values=dummy_images, text_prompts=dummy_queries)
    
    v_intact = outputs["V_intact"]
    v_corrupted = outputs["V_corrupted"]
    v_reduced = outputs["V_reduced"]
    saliency_scores = outputs["saliency_scores"]
    saliency_mask = outputs["saliency_mask"]

    print(f"    - V_intact shape    : {v_intact.shape} (Expected: [2, 50, 768])")
    print(f"    - V_corrupted shape : {v_corrupted.shape} (Expected: [2, 50, 768])")
    print(f"    - V_reduced shape   : {v_reduced.shape} (Expected: [2, 25, 768] for ratio=0.5)")
    print(f"    - Saliency scores   : {saliency_scores.shape} (Expected: [2, 49])")
    print(f"    - Saliency mask     : {saliency_mask.shape} (Expected: [2, 49, 1])")

    assert v_intact.shape == (2, 50, 768), "Invalid V_intact shape!"
    assert v_corrupted.shape == (2, 50, 768), "Invalid V_corrupted shape!"
    assert v_reduced.shape == (2, 25, 768), "Invalid V_reduced shape!"
    assert not torch.isnan(v_reduced).any(), "NaN values detected in V_reduced!"
    assert not torch.isnan(v_corrupted).any(), "NaN values detected in V_corrupted!"
    
    print("[+] Success: All simultaneous stream contracts verified successfully.")
    print("\n[🎉] All pipeline checks passed successfully! Ready for review.")

if __name__ == "__main__":
    test_pipeline()
