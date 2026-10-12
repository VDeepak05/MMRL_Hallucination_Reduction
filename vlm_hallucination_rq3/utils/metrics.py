import torch
import time

def benchmark_latency(module, dummy_input, num_iterations=100):
    """
    Benchmarks execution overhead to ensure latency is below the 2.0 ms target per batch.
    """
    module.eval()
    device = dummy_input.device
    
    # Warmup
    for _ in range(10):
        _ = module(dummy_input)
        
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = module(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.perf_counter()
    
    avg_latency_ms = ((end_time - start_time) / num_iterations) * 1000.0
    print(f"[-] Execution Overhead Benchmark:")
    print(f"    - Average Latency: {avg_latency_ms:.3f} ms per batch (Target: < 2.0 ms)")
    return avg_latency_ms

def feature_dissimilarity_sweep(module, dummy_input, noise_range=[0.1, 0.3, 0.5, 0.7, 1.0]):
    """
    Evaluates cosine distance across different noise scaling parameters sigma 
    to confirm semantic degradation without NaN states.
    """
    print("[-] Running Feature Dissimilarity Sweep across sigma values...")
    results = {}
    
    for sigma in noise_range:
        module.sigma = sigma
        outputs = module(dummy_input)
        v_intact = outputs["V_intact"]
        v_corrupted = outputs["V_corrupted"]
        
        # Check for NaNs
        has_nans = torch.isnan(v_corrupted).any().item()
        
        # Compute mean cosine similarity between intact and corrupted patches
        cos_sim = torch.nn.functional.cosine_similarity(v_intact, v_corrupted, dim=-1).mean().item()
        
        results[sigma] = {"cosine_similarity": cos_sim, "has_nans": has_nans}
        print(f"    - Sigma: {sigma} | Mean Cosine Sim: {cos_sim:.4f} | NaN Free: {not has_nans}")
        
    return results