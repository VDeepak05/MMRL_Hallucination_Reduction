import torch
import sys
import os

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.perturbation import VisualPerturbationAndAdaptiveSaliency

def test_perturbation_module():
    print("[-] Initializing Member 1 Visual Perturbation Module Test...")
    
    batch_size = 2
    num_patches = 196
    embed_dim = 1024

    # 1. Simulate visual patch tokens from a frozen vision encoder [B, N, D]
    dummy_patches = torch.randn(batch_size, num_patches, embed_dim)
    
    # 2. Instantiate module with standard configuration
    module = VisualPerturbationAndAdaptiveSaliency(
        top_k_ratio=0.3, 
        temperature=1.0, 
        noise_std=0.3
    )
    
    # 3. Forward pass
    outputs = module(dummy_patches)
    
    # 4. Assertions to verify output contract
    assert "V_intact" in outputs, "Missing V_intact in output dictionary"
    assert "V_corrupted" in outputs, "Missing V_corrupted in output dictionary"
    assert "saliency_mask" in outputs, "Missing saliency_mask in output dictionary"
    assert "saliency_scores" in outputs, "Missing saliency_scores in output dictionary"

    assert outputs["V_intact"].shape == (batch_size, num_patches, embed_dim)
    assert outputs["V_corrupted"].shape == (batch_size, num_patches, embed_dim)
    assert outputs["saliency_mask"].shape == (batch_size, num_patches)
    assert outputs["saliency_scores"].shape == (batch_size, num_patches)

    print("[+] All unit test assertions passed successfully!")
    print(f"    - V_intact shape: {outputs['V_intact'].shape}")
    print(f"    - V_corrupted shape: {outputs['V_corrupted'].shape}")
    print(f"    - Saliency Mask shape: {outputs['saliency_mask'].shape}")

if __name__ == "__main__":
    test_perturbation_module()