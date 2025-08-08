# ======================================================================================
# train.py (Stage 2.6 - Structured BC)
#
# Description:
#   Trains the Structured Behavioral Cloning (BC-S) agent. It uses a dataset
#   that pre-loads all chunked, structured data into RAM for maximum I/O performance
#   and trains the new entity-centric StructuredBCModel.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm
from functools import partial
import random

from src.shared.utils import TrainingConfig, WeightedMSELoss 
from src.stage_2_6_structured_bc.networks import StructuredBCModel # Import our new structured model

# def scenario_subsampling_collate_fn(batch_of_scenarios, k=16):
#     """
#     A custom collate function that implements scenario sub-sampling.
    
#     Args:
#         batch_of_scenarios: A list where each item is a list of samples from one scenario.
#         k: The number of random timesteps to sample from each scenario.
#     """
#     all_states = []
#     all_actions = []
    
#     # Iterate through the scenarios that the DataLoader has given us in this batch
#     for scenario_samples in batch_of_scenarios:
#         num_samples_in_scenario = len(scenario_samples)
        
#         # Choose k random indices from this scenario
#         # Use replacement=True in case a scenario has fewer than k samples
#         sample_indices = np.random.choice(
#             num_samples_in_scenario, 
#             size=k, 
#             replace=(num_samples_in_scenario < k)
#         )
        
#         # Collect the states and actions for the chosen indices
#         for i in sample_indices:
#             all_states.append(scenario_samples[i]['state'])
#             all_actions.append(scenario_samples[i]['action'])
            
#     # Now, collate the collected states and actions into final batch tensors
#     # Collate states (which are dictionaries)
#     final_states_dict = {
#         key: torch.stack([torch.from_numpy(s[key]) for s in all_states])
#         for key in all_states[0].keys()
#     }
    
#     # Collate actions
#     final_actions = torch.stack(all_actions)
    
#     return final_states_dict, final_actions


# class ScenarioDataset(Dataset):
#     """
#     A dataset where each item is an entire scenario (a list of all valid
#     timesteps from a single file).
#     """
#     def __init__(self, data_dir):
#         print(f"Indexing scenario files in: {data_dir}")
#         self.scenario_paths = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
#         if not self.scenario_paths:
#             raise FileNotFoundError(f"No scenario .pt files found.")
#         print(f"Found {len(self.scenario_paths)} scenarios.")

#     def __len__(self):
#         return len(self.scenario_paths)

#     def __getitem__(self, idx):
#         # Load and return the entire list of samples for one scenario
#         return torch.load(self.scenario_paths[idx], weights_only=False)
    
class IterableBCDataset(torch.utils.data.IterableDataset):
    """
    An iterable dataset designed for the "scenario-per-file" format.
    It streams data efficiently from disk by loading one full scenario at a time.

    In each epoch, it shuffles the list of scenarios and then yields every
    single timestep from every scenario, guaranteeing that all data is used
    while minimizing I/O operations.
    """
    def __init__(self, data_dir, worker_info=None):
        super().__init__()
        self.data_dir = data_dir
        self.scenario_paths = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
        
        # This is for multi-worker data loading. Each worker will get a
        # different subset of the scenario files to process.
        if worker_info is None:
            self.start = 0
            self.end = len(self.scenario_paths)
        else:
            per_worker = int(np.ceil(len(self.scenario_paths) / float(worker_info.num_workers)))
            self.start = worker_info.id * per_worker
            self.end = min(self.start + per_worker, len(self.scenario_paths))
            
            print(f"Worker {worker_info.id} processing scenarios {self.start} to {self.end} (total: {len(self.scenario_paths)})")

    def __iter__(self):
        # Get the list of files for this specific worker
        worker_scenario_paths = self.scenario_paths[self.start:self.end]
        
        # Shuffle the scenarios for this worker at the beginning of each epoch
        random.shuffle(worker_scenario_paths)
        
        # Iterate through the shuffled scenarios
        for scenario_path in worker_scenario_paths:
            # Load one full scenario into memory (one disk read)
            scenario_samples = torch.load(scenario_path, weights_only=False)
            
            # Yield every sample from that scenario
            for sample in scenario_samples:
                yield sample['state'], sample['action']

def structured_collate_fn(batch):
    """
    A custom collate function to handle batches of structured state dictionaries.
    """
    # Batch is a list of (state_dict, action_tensor) tuples
    states_list = [item[0] for item in batch]
    actions_list = [item[1] for item in batch]
    
    # Collate the states dictionary
    final_states_dict = {
        key: torch.stack([torch.from_numpy(s[key]) for s in states_list])
        for key in states_list[0].keys()
    }
    
    # Collate the actions
    final_actions = torch.stack(actions_list)
    
    return final_states_dict, final_actions

def worker_init_fn(worker_id):
    """
    Ensures each worker in the DataLoader gets a unique slice of the data.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    
    # Re-initialize the dataset for this specific worker
    dataset.__init__(dataset.data_dir, worker_info=worker_info)
           
# --- Main Training Logic ---
def train(config: TrainingConfig):
    print(f"Starting Structured BC training on device: {config.device}")
    
    os.makedirs(os.path.dirname(config.bcs_model_path), exist_ok=True)
    
    # --- Datasets and DataLoaders ---
    # train_dataset = ScenarioDataset(config.bcs_preprocess_train_dir)
    # val_dataset = ScenarioDataset(config.bcs_preprocess_val_dir)

    # Use functools.partial to create a collate function with our desired k
    # You can make k a parameter in your TrainingConfig
    # k_samples_per_scenario = 16
    # collate_fn = partial(scenario_subsampling_collate_fn, k=k_samples_per_scenario)

    # The batch_size for the DataLoader now refers to the number of SCENARIOS, not samples
    # To get a final batch of ~512, we need batch_size = 512 / k
    # scenario_batch_size = config.batch_size // k_samples_per_scenario

    # train_loader = DataLoader(
    #     train_dataset, 
    #     batch_size=scenario_batch_size, 
    #     shuffle=True, 
    #     num_workers=config.num_workers,
    #     pin_memory=True,
    #     persistent_workers=True if config.num_workers > 0 else False,
    #     collate_fn=collate_fn # Use our custom collate function
    # )

    # val_loader = DataLoader(
    #     val_dataset, batch_size=scenario_batch_size, 
    #     shuffle=False, 
    #     num_workers=config.num_workers,
    #     pin_memory=True,
    #     persistent_workers=True if config.num_workers > 0 else False,
    #     collate_fn=collate_fn # Use our custom collate function
    # )
    # --- Datasets and DataLoaders (UPDATED) ---
    train_dataset = IterableBCDataset(config.bcs_preprocess_train_dir)
    val_dataset = IterableBCDataset(config.bcs_preprocess_val_dir)

    # For IterableDataset, shuffling is done within the dataset's __iter__ method.
    # The DataLoader's `shuffle` argument MUST be False.
    # The batch size now refers to the final number of samples.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=False, # MUST be False for IterableDataset
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        collate_fn=structured_collate_fn, # Use our simple structured collator
        worker_init_fn=worker_init_fn, # Add this

    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        collate_fn=structured_collate_fn,
        worker_init_fn=worker_init_fn, # Add this

    )
    # --- Model, Optimizer, Loss ---
    # Instantiate our new StructuredBCModel
    model = StructuredBCModel(config).to(config.device)
    
    # Load previous checkpoint if it exists
    if os.path.exists(config.bcs_model_path):
        print(f"Loading existing model from {config.bcs_model_path}")
        model.load_state_dict(torch.load(config.bcs_model_path, map_location=config.device))
    else:
        print("No existing model found, starting from scratch.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5) # Added small weight decay
    
    weights_path = os.path.join(os.path.dirname(config.bcs_model_path), 'action_weights_stage_2_6.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Action weights not found at {weights_path}. Please run compute_action_weights.py first.")
    loss_fn = WeightedMSELoss(weights_path).to(config.device)
    
    best_val_loss = float('inf')
    
    # --- Main Training Loop ---
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for states_dict, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]"):
            if states_dict is None: continue # Skip empty batches
            
            states_dict = {k: v.to(config.device, non_blocking=True) for k, v in states_dict.items()}
            actions = actions.to(config.device, non_blocking=True)
            
            optimizer.zero_grad()
            pred_actions = model(states_dict)
            loss = loss_fn(pred_actions, actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for states_dict, actions in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]"):
                if states_dict is None: continue # Skip empty batches
                
                states_dict = {k: v.to(config.device, non_blocking=True) for k, v in states_dict.items()}
                actions = actions.to(config.device, non_blocking=True)
                pred_actions = model(states_dict)
                loss = loss_fn(pred_actions, actions)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{config.num_epochs} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.bcs_model_path)
            print(f"  -> New best model saved to {config.bcs_model_path} (Val Loss: {best_val_loss:.6f})")

    print("\n--- Structured BC Training with Weighted Loss Complete ---")

if __name__ == "__main__":
    # Ensure your utils.py has the correct config paths
    # e.g., bcs_model_path: str = os.path.expanduser('~/WaymoOfflineAgent/models/bc_structured_policy.pth')
    config = TrainingConfig()
    train(config)