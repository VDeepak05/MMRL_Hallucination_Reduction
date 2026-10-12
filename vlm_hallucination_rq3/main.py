import torch
import yaml
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from modules.perturbation import VisualPerturbationAndAdaptiveSaliency
from utils.metrics import benchmark_latency, feature_dissimilarity_sweep

def load_config():
    config_path = "configs/model_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {
        "top_k_ratio": 0.3,
        "temperature": 1.0,
        "noise_std": 0.3,
        "batch_size": 2,
        "num_patches": 196,
        "embed_dim": 1024
    }

def main():
    print("==================================================")
    print(" RQ3 — Member 1: Visual Perturbation & Saliency   ")
    print("==================================================")
    
    config = load_config()
    
    # 1. Initialize Module
    module = VisualPerturbationAndAdaptiveSaliency(
        top_k_ratio=config["top_k_ratio"],
        temperature=config["temperature"],
        noise_std=config["noise_std"]
    )
    
    # 2. Create Dummy Input [B, N, D]
    dummy_input = torch.randn(
        config["batch_size"], 
        config["num_patches"], 
        config["embed_dim"]
    )
    
    # 3. Run Forward Pass
    outputs = module(dummy_input)
    print("[+] Forward pass executed successfully.")
    print(f"    - Output Keys: {list(outputs.keys())}")
    
    # 4. Run Benchmarks & Metrics
    print("\n--- Running Evaluation Metrics ---")
    benchmark_latency(module, dummy_input)
    feature_dissimilarity_sweep(module, dummy_input)
    
    print("\n[+] All Member 1 tasks completed successfully for tomorrow's review!")

if __name__ == "__main__":
    main()