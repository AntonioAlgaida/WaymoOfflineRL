# ======================================================================================
# compute_action_weights_stage_2_6.py (Multiprocess-Optimized)
#
# Description:
#   Analyzes the preprocessed kinematic training data in parallel to compute
#   action frequencies. Each worker process computes a histogram for a chunk
#   of the data, and the results are combined before calculating the final weights.
#
# Author: Antonio Guillen-Perez
# ======================================================================================
import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import multiprocessing
from src.shared.utils import TrainingConfig

def process_chunk(scenario_chunk_tuple): # Renamed for clarity
    """
    Worker function. Processes a list (chunk) of SCENARIO .pt files and returns
    a local histogram of action frequencies.
    """
    scenario_chunk, accel_bins, steer_bins = scenario_chunk_tuple
    
    local_histogram = np.zeros((len(accel_bins) - 1, len(steer_bins) - 1), dtype=np.int64)
    
    # --- THIS IS THE KEY CHANGE ---
    # The outer loop is now over scenario files.
    for scenario_path in scenario_chunk:
        try:
            # Load the entire list of samples for one scenario
            scenario_samples = torch.load(scenario_path, weights_only=False)
            
            # The inner loop is over the samples within that scenario
            for sample in scenario_samples:
                action = sample['action'].numpy()
                
                accel_bin = np.digitize(action[0], accel_bins) - 1
                steer_bin = np.digitize(action[1], steer_bins) - 1
                
                if 0 <= accel_bin < local_histogram.shape[0] and 0 <= steer_bin < local_histogram.shape[1]:
                    local_histogram[accel_bin, steer_bin] += 1
        except Exception:
            continue
            
    return local_histogram

def main():
    config = TrainingConfig()
    # Use a sensible number of workers, leaving some cores for system tasks
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"--- Computing Action Frequency Weights using {num_workers} cores ---")

    # --- 1. Define the Data-Driven, Non-Uniform Discretization Grid ---
    print("Defining final non-uniform binning strategy based on filtered data distribution...")

    # Bins for Acceleration (m/s^2), covering the filtered range [-10.0, 8.0]
    ACCEL_BINS = np.concatenate([
        np.linspace(-10.0, -5.0, num=10),      # 9 bins for hard braking
        np.linspace(-5.0, -2.0, num=15),      # 14 bins for moderate braking
        np.linspace(-2.0, 1.5, num=50),       # 49 small bins for the dense central region
        np.linspace(1.5, 4.0, num=15),        # 14 bins for moderate acceleration
        np.linspace(4.0, 8.0, num=10),        # 9 bins for hard acceleration
    ])

    # Bins for Steering Angle (radians), covering the filtered range [-0.8, 0.8]
    STEER_BINS = np.concatenate([
        np.linspace(-0.8, -0.3, num=15),      # 14 bins for significant left turns
        np.linspace(-0.3, -0.05, num=30),     # 29 bins for slight left turns
        np.linspace(-0.05, 0.05, num=20),     # 19 very small bins for the "dead zone" of driving straight
        np.linspace(0.05, 0.3, num=30),      # 29 bins for slight right turns
        np.linspace(0.3, 0.8, num=15),        # 14 bins for significant right turns
    ])

    total_accel_bins = len(ACCEL_BINS) - 1
    total_steer_bins = len(STEER_BINS) - 1
    print(f"Using a non-uniform grid with {total_accel_bins} acceleration bins and {total_steer_bins} steering bins.")
    
    # --- 2. Get File List and Split into Chunks ---
    # Use the config path for the filtered kinematic training data
    data_files = glob.glob(os.path.join(config.bcs_preprocess_train_dir, '*.pt'))
    if not data_files:
        raise FileNotFoundError(f"No .pt files found in {config.bcs_preprocess_train_dir}. Please run the filtering preprocessor first.")
        
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
    output_path = os.path.join(os.path.dirname(config.bcs_model_path), 'action_weights_stage_2_6.pt')
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