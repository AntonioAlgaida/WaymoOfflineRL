# ======================================================================================
# analyze_actions.py (Stage 2.6 - For Structured, Scenario-per-File Data)
#
# Description:
#   Analyzes the expert action distribution from the "scenario-per-file" dataset.
#   It runs in parallel to collect all (acceleration, steering) actions and saves
#   them for visualization in a Jupyter notebook.
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

def collect_actions_from_scenarios(scenario_chunk: list):
    """
    Worker function. Processes a list (chunk) of scenario .pt files and returns
    a single NumPy array of all the action vectors found.
    """
    actions_list = []
    for scenario_path in scenario_chunk:
        try:
            # Load the list of samples for one scenario
            scenario_samples = torch.load(scenario_path, weights_only=False)
            # Extract the 'action' tensor from each sample in the list
            for sample in scenario_samples:
                actions_list.append(sample['action'].numpy())
        except Exception:
            # Silently skip corrupted or empty files
            continue
            
    # vstack is more efficient for combining arrays of the same shape
    # and handles the case where a worker finds no valid actions.
    return np.vstack(actions_list) if actions_list else np.array([]).reshape(0, 2)


def main():
    config = TrainingConfig()
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"--- Analyzing Action Distribution using {num_workers} cores (Scenario-per-File format) ---")

    # Point to the new structured "scenario-per-file" training data
    data_files = glob.glob(os.path.join(config.bcs_preprocess_train_dir, '*.pt'))
    if not data_files:
        raise FileNotFoundError(f"No scenario .pt files found in {config.bcs_preprocess_train_dir}. Please run the correct preprocessor first.")
        
    print(f"Found {len(data_files)} scenario files to analyze...")
    
    # Split the list of scenario files into chunks for each worker
    scenario_chunks = np.array_split(data_files, num_workers * 4) # More chunks for a smoother progress bar
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(collect_actions_from_scenarios, scenario_chunks), total=len(scenario_chunks), desc="Collecting actions from scenarios"))

    print("Combining results from all workers...")
    # Filter out any empty results from workers that found no valid actions
    valid_results = [res for res in results if res.size > 0]
    if not valid_results:
        raise ValueError("No valid actions were collected from any of the files.")
        
    all_actions = np.vstack(valid_results) # Shape will be (num_total_samples, 2)
    
    # Use the centralized path from the config for the output
    output_path = os.path.join(os.path.dirname(config.bcs_preprocess_train_dir), 'all_actions_distribution_stage_2_6.npy')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, all_actions)
    
    print(f"\n--- Analysis Complete ---")
    print(f"Saved {all_actions.shape[0]} action vectors to: {output_path}")
if __name__ == "__main__":
    main()