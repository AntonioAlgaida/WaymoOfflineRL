# ======================================================================================
# compute_action_weights_stage_2_5.py (Multiprocess-Optimized)
#
# Description:
#   Analyzes the preprocessed kinematic training data in parallel to compute
#   action frequencies. Each worker process computes a histogram for a chunk
#   of the data, and the results are combined before calculating the final weights.
#
# Author: Antonio Guillen-Perez
# Date: 2025-07-31
# ======================================================================================
import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import multiprocessing
from src.shared.utils import TrainingConfig

def process_chunk(file_chunk_tuple):
    """
    Worker function. Processes a list (chunk) of .pt files and returns a
    local histogram of action frequencies.
    """
    file_chunk, accel_bins, steer_bins = file_chunk_tuple
    
    # Each worker gets its own local histogram
    local_histogram = np.zeros((len(accel_bins) - 1, len(steer_bins) - 1), dtype=np.int64)
    
    for file_path in file_chunk:
        try:
            data = torch.load(file_path, weights_only=False)
            action = data['action'].numpy()
            
            accel_bin = np.digitize(action[0], accel_bins) - 1
            steer_bin = np.digitize(action[1], steer_bins) - 1
            
            if 0 <= accel_bin < local_histogram.shape[0] and 0 <= steer_bin < local_histogram.shape[1]:
                local_histogram[accel_bin, steer_bin] += 1
        except Exception:
            print(f"Error processing file {file_path}. Skipping this file.")
            continue
            
    return local_histogram

def main():
    config = TrainingConfig()
    # Use a sensible number of workers, leaving some cores for system tasks
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"--- Computing Action Frequency Weights using {num_workers} cores ---")

    # --- 1. Define the Data-Driven, Non-Uniform Discretization Grid ---
    print("Defining non-uniform binning strategy...")
    ACCEL_BINS = np.concatenate([
        np.linspace(-10.0, -4.0, num=10),
        np.linspace(-4.0, -1.0, num=20),
        np.linspace(-1.0, 1.0, num=40),
        np.linspace(1.0, 4.0, num=20),
        np.linspace(4.0, 8.0, num=10),
    ])
    STEER_BINS = np.concatenate([
        np.linspace(-0.8, -0.3, num=15),
        np.linspace(-0.3, 0.3, num=70),
        np.linspace(0.3, 0.8, num=15),
    ])
    
    total_accel_bins = len(ACCEL_BINS) - 1
    total_steer_bins = len(STEER_BINS) - 1
    print(f"Using a non-uniform grid with {total_accel_bins} acceleration bins and {total_steer_bins} steering bins.")
    
    # --- 2. Get File List and Split into Chunks ---
    # Use the config path for the filtered kinematic training data
    data_files = glob.glob(os.path.join(config.bck_preprocess_train_dir, '*.pt'))
    if not data_files:
        raise FileNotFoundError(f"No .pt files found in {config.bck_preprocess_train_dir}. Please run the filtering preprocessor first.")
        
    print(f"Found {len(data_files)} samples to analyze...")
    
    # Split the list of files into chunks for each worker
    file_chunks = np.array_split(data_files, num_workers * 4) # More chunks for a smoother progress bar
    tasks = [(chunk, ACCEL_BINS, STEER_BINS) for chunk in file_chunks]

    # --- 3. Run Multiprocessing Pool and Get Results ---
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_chunk, tasks), total=len(tasks), desc="Processing chunks"))

    # --- 4. The Reduction Step: Combine Histograms ---
    print("Combining results from all workers...")
    # This is the correct way: create the final histogram by summing the results.
    final_histogram = np.sum(results, axis=0)
            
    # --- 5. Calculate Smoothed Inverse Frequency Weights ---
    print("Calculating inverse frequency weights...")
    epsilon = 1.0 
    action_weights = 1.0 / (final_histogram.astype(np.float32) + epsilon)
    action_weights /= np.mean(action_weights)

    # --- 6. Save the Weight Matrix ---
    # Use the centralized path from the config file
    output_path = os.path.join(os.path.dirname(config.bck_model_path), 'action_weights.pt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    weight_data = {
        'weights': torch.from_numpy(action_weights),
        'accel_bins': torch.from_numpy(ACCEL_BINS),
        'steer_bins': torch.from_numpy(STEER_BINS)
    }
    torch.save(weight_data, output_path)
    
    print(f"\n--- Weight computation complete ---")
    print(f"Action frequency weight matrix saved to: {output_path}")
    
if __name__ == "__main__":
    main()