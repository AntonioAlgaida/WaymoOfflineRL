# ======================================================================================
# compute_state_stats.py (Multiprocess-Optimized)
#
# Description:
#   Calculates the mean and standard deviation for each feature in the state vector
#   in parallel across multiple CPU cores. It uses a parallel version of
#   Welford's algorithm for numerically stable, one-pass computation.
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

# Import from our central config file
from src.shared.utils import TrainingConfig

def process_stats_chunk(file_chunk: list) -> tuple:
    """
    Worker function. Processes a list (chunk) of .pt files and returns the
    (count, mean, M2) tuple required for parallel Welford's algorithm.
    """
    count = 0
    mean = np.zeros(279, dtype=np.float64)
    M2 = np.zeros(279, dtype=np.float64)

    for file_path in file_chunk:
        try:
            data = torch.load(file_path, weights_only=False)
            state = data['state'].numpy().astype(np.float64)
            
            # Standard Welford's algorithm for this chunk
            count += 1
            delta = state - mean
            mean += delta / count
            delta2 = state - mean
            M2 += delta * delta2
        except Exception:
            continue
            
    return (count, mean, M2)

def combine_stats(stats_a: tuple, stats_b: tuple) -> tuple:
    """
    Combines two sets of Welford's algorithm statistics into one.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    count_a, mean_a, M2_a = stats_a
    count_b, mean_b, M2_b = stats_b
    
    # If one of the chunks was empty, return the other one
    if count_a == 0: return stats_b
    if count_b == 0: return stats_a

    count_combined = count_a + count_b
    delta = mean_b - mean_a
    
    mean_combined = mean_a + delta * count_b / count_combined
    M2_combined = M2_a + M2_b + delta**2 * count_a * count_b / count_combined
    
    return (count_combined, mean_combined, M2_combined)

def main():
    config = TrainingConfig()
    num_workers = multiprocessing.cpu_count() // 2
    print(f"--- Computing State Normalization Statistics using {num_workers} cores ---")

    data_files = glob.glob(os.path.join(config.bck_preprocess_train_dir, '*.pt'))
    if not data_files:
        raise FileNotFoundError(f"No .pt files found in {config.bck_preprocess_train_dir}.")
        
    print(f"Found {len(data_files)} samples to analyze...")
    
    # Split the list of files into chunks for each worker
    file_chunks = np.array_split(data_files, num_workers * 4) # More chunks for a smoother progress bar
    
    # --- Run the multiprocessing pool ---
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Each worker returns a (count, mean, M2) tuple
        results = list(tqdm(pool.imap_unordered(process_stats_chunk, file_chunks), total=len(tasks), desc="Processing chunks"))
    
    # --- The Reduction Step ---
    print("Combining statistics from all workers...")
    # Use functools.reduce to apply the combine function across the list of results
    final_count, final_mean, final_M2 = reduce(combine_stats, results)

    if final_count < 2:
        raise ValueError("Not enough data to compute variance.")
        
    final_variance = final_M2 / (final_count - 1)
    final_std = np.sqrt(final_variance)
    
    # Add a small epsilon to std to prevent division by zero for constant features
    final_std[final_std < 1e-6] = 1e-6

    # --- Save the statistics ---
    stats_path = os.path.join(os.path.dirname(config.cql_model_path), 'state_stats.pt')
    stats_data = {
        'mean': torch.from_numpy(final_mean).float(),
        'std': torch.from_numpy(final_std).float()
    }
    torch.save(stats_data, stats_path)
    
    print(f"\n--- State statistics computation complete ---")
    print(f"Saved mean and std to: {stats_path}")

if __name__ == "__main__":
    # Add `tasks` variable to the `main` function to make it work
    config = TrainingConfig()
    num_workers = multiprocessing.cpu_count() // 2
    data_files = glob.glob(os.path.join(config.bck_preprocess_train_dir, '*.pt'))
    file_chunks = np.array_split(data_files, num_workers * 4) # More chunks for a smoother progress bar
    tasks = file_chunks
    main()