import torch
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.perturbation import VisualPerturbationAndAdaptiveSaliency as BaselineModule
from modules.perturbation_improved import VisualPerturbationSoftSaliency as ImprovedModule

def compare_models():
    print("==================================================")
    print(" RQ3 — Model Variant Comparison (Baseline vs Improved)")
    print("==================================================")

    batch_size = 2
    num_patches = 196
    embed_dim = 1024
    
    # Identical test input for fair comparison
    torch.manual_seed(42)
    dummy_input = torch.randn(batch_size, num_patches, embed_dim)

    # Instantiate both models
    baseline = BaselineModule(top_k_ratio=0.3, temperature=1.0, noise_std=0.3)
    improved = ImprovedModule(temperature=1.0, noise_std=0.3)

    baseline.eval()
    improved.eval()

    # --- 1. Evaluate Baseline ---
    start = time.perf_counter()
    out_base = baseline(dummy_input)
    time_base = (time.perf_counter() - start) * 1000.0
    
    sim_base = torch.nn.functional.cosine_similarity(
        out_base["V_intact"], out_base["V_corrupted"], dim=-1
    ).mean().item()

    # --- 2. Evaluate Improved ---
    start = time.perf_counter()
    out_impr = improved(dummy_input)
    time_impr = (time.perf_counter() - start) * 1000.0
    
    sim_impr = torch.nn.functional.cosine_similarity(
        out_impr["V_intact"], out_impr["V_corrupted"], dim=-1
    ).mean().item()

    # --- 3. Print Results Table ---
    print(f"\n{'Metric':<25} | {'Baseline (Hard Top-K)':<22} | {'Improved (Soft Saliency)':<24}")
    print("-" * 78)
    print(f"{'Execution Latency (ms)':<25} | {time_base:<22.3f} | {time_impr:<24.3f}")
    print(f"{'Mean Cosine Similarity':<25} | {sim_base:<22.4f} | {sim_impr:<24.4f}")
    print(f"{'NaN-Free Status':<25} | {str(not torch.isnan(out_base['V_corrupted']).any()):<22} | {str(not torch.isnan(out_impr['V_corrupted']).any()):<24}")
    
    print("\n[+] Evaluation Verdict:")
    if sim_impr > sim_base:
        print("    -> Improved model retains stronger background feature fidelity (higher similarity).")
    else:
        print("    -> Baseline model maintains tighter localized disruption.")

if __name__ == "__main__":
    compare_models()
