# ======================================================================================
# analyze_actions.py
#
# Description:
#   Performs a deep analysis of the expert action distribution. It runs in parallel
#   to collect all (acceleration, steering) actions from the preprocessed training
#   set and saves them. This data is then used to generate histograms to
#   inform our binning strategy for the weighted loss function.
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
from utils import TrainingConfig

def collect_actions_chunk(file_chunk):
    """
    Worker function. Processes a list (chunk) of .pt files and returns
    a list of all the action vectors found.
    """
    actions_list = []
    for file_path in file_chunk:
        try:
            data = torch.load(file_path, weights_only=False)
            actions_list.append(data['action'].numpy())
        except Exception:
            continue
    return np.array(actions_list)

def main():
    config = TrainingConfig()
    num_workers = multiprocessing.cpu_count() // 2
    print(f"--- Analyzing Action Distribution using {num_workers} cores ---")

    data_files = glob.glob(os.path.join(config.bck_preprocess_train_dir, '*.pt'))
    if not data_files:
        raise FileNotFoundError(f"No .pt files found in {config.bck_preprocess_train_dir}. Please run preprocessing first.")
        
    print(f"Found {len(data_files)} samples to analyze...")
    
    file_chunks = np.array_split(data_files, num_workers * 4) # More chunks for better progress bar
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(collect_actions_chunk, file_chunks), total=len(file_chunks), desc="Collecting actions"))

    print("Combining results from all workers...")
    all_actions = np.vstack(results) # Shape will be (num_total_samples, 2)
    
    output_path = os.path.join(os.path.dirname(config.bck_model_path), 'all_actions_distribution.npy')
    np.save(output_path, all_actions)
    
    print(f"\n--- Analysis Complete ---")
    print(f"Saved {all_actions.shape[0]} action vectors to: {output_path}")

if __name__ == "__main__":
    main()