# ======================================================================================
# compute_state_stats_structured.py (Structured & Parallel)
#
# Description:
#   Calculates mean/std for the structured state dictionary across the entire
#   training dataset. It uses a parallel Welford's algorithm to compute these
#   stats efficiently and saves them as dictionaries for normalization.
#
# Author: Antonio Guillen-Perez
# ======================================================================================
import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import multiprocessing
from functools import reduce

from src.shared.utils import TrainingConfig

def initialize_stats_dict(config: TrainingConfig):
    """Initializes empty dictionaries for count, mean, and M2 with correct shapes."""
    mean_dict = {
        'ego': np.zeros(3, dtype=np.float64),
        'agents': np.zeros((config.num_closest_agents, 10), dtype=np.float64),
        'lanes': np.zeros((config.num_closest_map_points, 2), dtype=np.float64),
        'crosswalks': np.zeros((config.num_closest_crosswalk_points, 2), dtype=np.float64),
        'route': np.zeros((config.num_future_waypoints, 2), dtype=np.float64),
        'rules': np.zeros(9, dtype=np.float64)
    }
    M2_dict = {k: np.zeros_like(v) for k, v in mean_dict.items()}
    return 0, mean_dict, M2_dict

def process_stats_chunk(chunk_tuple: tuple) -> tuple:
    """
    Worker function. Processes a chunk of scenario files and returns the
    (count, mean_dict, M2_dict) tuple.
    """
    scenario_chunk, config = chunk_tuple
    count, mean_dict, M2_dict = initialize_stats_dict(config)

    for scenario_path in scenario_chunk:
        try:
            scenario_samples = torch.load(scenario_path, weights_only=False)
            for sample in scenario_samples:
                state_dict = sample['state']
                
                count += 1
                for key in mean_dict.keys():
                    state_val = state_dict[key].astype(np.float64)
                    delta = state_val - mean_dict[key]
                    mean_dict[key] += delta / count
                    delta2 = state_val - mean_dict[key]
                    M2_dict[key] += delta * delta2
        except Exception:
            continue
            
    return (count, mean_dict, M2_dict)

def combine_stats(stats_a: tuple, stats_b: tuple) -> tuple:
    """
    Combines two sets of Welford's algorithm statistics dictionaries.
    """
    count_a, mean_dict_a, M2_dict_a = stats_a
    count_b, mean_dict_b, M2_dict_b = stats_b
    
    if count_a == 0: return stats_b
    if count_b == 0: return stats_a

    count_combined = count_a + count_b
    
    # Initialize new dictionaries for the combined results
    _, mean_dict_c, M2_dict_c = initialize_stats_dict(TrainingConfig())

    for key in mean_dict_a.keys():
        delta = mean_dict_b[key] - mean_dict_a[key]
        mean_dict_c[key] = mean_dict_a[key] + delta * count_b / count_combined
        M2_dict_c[key] = M2_dict_a[key] + M2_dict_b[key] + np.power(delta, 2) * count_a * count_b / count_combined
    
    return (count_combined, mean_dict_c, M2_dict_c)

def main():
    config = TrainingConfig()
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"--- Computing Structured State Stats using {num_workers} cores ---")

    # Use the RL-specific scenario-per-file data
    data_files = glob.glob(os.path.join(config.cql_preprocess_dir, '*.pt'))
    if not data_files:
        raise FileNotFoundError(f"No .pt files found in {config.cql_preprocess_dir}.")
        
    print(f"Found {len(data_files)} scenario files to analyze...")
    
    file_chunks = np.array_split(data_files, num_workers * 2) # Fewer, larger chunks
    tasks = [(chunk, config) for chunk in file_chunks]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_stats_chunk, tasks), total=len(tasks), desc="Processing chunks"))
    
    print("Combining statistics from all workers...")
    final_count, final_mean_dict, final_M2_dict = reduce(combine_stats, results)

    if final_count < 2:
        raise ValueError("Not enough data to compute variance.")
        
    final_variance_dict = {k: v / (final_count - 1) for k, v in final_M2_dict.items()}
    final_std_dict = {k: np.sqrt(v) for k, v in final_variance_dict.items()}
    
    # Add epsilon for numerical stability
    for key in final_std_dict:
        final_std_dict[key][final_std_dict[key] < 1e-6] = 1e-6

    # --- Save the statistics dictionaries ---
    # Convert back to torch tensors for consistency
    mean_torch_dict = {k: torch.from_numpy(v).float() for k, v in final_mean_dict.items()}
    std_torch_dict = {k: torch.from_numpy(v).float() for k, v in final_std_dict.items()}
    
    stats_data = {'mean': mean_torch_dict, 'std': std_torch_dict}
    output_path = config.state_stats_structured_path
    torch.save(stats_data, output_path)
    
    print(f"\n--- State statistics computation complete ---")
    print(f"Saved mean and std dictionaries to: {output_path}")

if __name__ == "__main__":
    main()