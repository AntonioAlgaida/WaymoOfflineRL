# ======================================================================================
# train.py (Stage 2.8 - Structured ML BC)
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
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import os
import glob
from tqdm import tqdm
from functools import partial
import random
from datetime import datetime
import time
import random

from torch.utils.tensorboard import SummaryWriter

from src.shared.utils import TrainingConfig, WeightedMSELoss 
from src.stage_2_8_structured_mlp_bc.networks import StructuredMLP_BC_Model # Import our new structured model

def structured_collate_fn(batch):
    """
    A simple collate function, as the data is now already in tensor format.
    """
    states_list = [item[0] for item in batch]
    actions_list = [item[1] for item in batch]
    
    # The expensive torch.from_numpy() call is GONE from here.
    final_states_dict = {
        key: torch.stack([s[key] for s in states_list])
        for key in states_list[0].keys()
    }
    
    final_actions = torch.stack(actions_list)
    
    return final_states_dict, final_actions

class BCDataset(Dataset):
    """
    A map-style dataset for "scenario-per-file" data.

    This dataset is ideal for Behavioral Cloning. It creates a master index
    of every single valid timestep in the entire dataset, allowing the DataLoader
    to perform true random shuffling for statistically efficient training.
    It uses a simple cache to mitigate I/O for recently accessed scenarios.
    """
    def __init__(self, data_dir: str):
        super().__init__()
        print(f"Initializing BC Dataset from: {data_dir}")
        scenario_paths = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
        if not scenario_paths:
            raise FileNotFoundError(f"No scenario .pt files found in {data_dir}.")

        self.master_index = []
        print("Building master index of all valid timesteps...")

        # For debugging, we can limit the number of scenarios indexed
        scenario_paths = scenario_paths[:30000]  # Uncomment to limit for testing
        for scenario_path in tqdm(scenario_paths, desc="Indexing scenarios"):
            # Load each file once during init to find out how many samples it contains
            try:
                num_samples_in_scenario = len(torch.load(scenario_path, weights_only=False))
                for i in range(num_samples_in_scenario):
                    self.master_index.append((scenario_path, i))
            except Exception:
                print(f"Warning: Could not load or index file {scenario_path}. Skipping.")

        # A simple cache for the last loaded scenario file to speed up access
        self.cache = {'path': None, 'data': None}
        
        print(f"Indexing complete. Found {len(self.master_index)} total valid samples.")

    def __len__(self):
        return len(self.master_index)

    def __getitem__(self, idx):
        scenario_path, inner_index = self.master_index[idx]
        
        # 1. Check the cache
        if scenario_path == self.cache['path']:
            processed_scenario_data = self.cache['data']
        else:
            # 2. Load from disk
            raw_scenario_data = torch.load(scenario_path, weights_only=False)
            
            # --- THIS IS THE NEW OPTIMIZATION ---
            # 3. Pre-convert the ENTIRE scenario to tensors ONCE
            processed_scenario_data = []
            for sample in raw_scenario_data:
                state_dict_tensor = {
                    k: torch.from_numpy(v) for k, v in sample['state'].items()
                }
                processed_scenario_data.append({
                    'state': state_dict_tensor,
                    'action': sample['action'] # Already a tensor
                })
            
            # 4. Store the processed data in the cache
            self.cache['path'] = scenario_path
            self.cache['data'] = processed_scenario_data
        
        # 5. Get the specific sample (now it's a dict of tensors)
        sample = processed_scenario_data[inner_index]
        
        return sample['state'], sample['action']

# --- The Final, High-Performance "Stochastic Epoch" Iterable Dataset ---
class StochasticEpochBCDataset(IterableDataset):
    """
    The definitive, high-performance iterable dataset.
    - Uses a "stochastic epoch" strategy for speed and diversity.
    - Loads data in scenario-sized chunks to minimize disk I/O.
    - Pre-converts each chunk to Tensors to minimize CPU work in the main loop.
    """
    def __init__(self, data_dir: str, k_samples_per_scenario: int, worker_info=None):
        super().__init__()
        self.data_dir = data_dir
        self.k = k_samples_per_scenario
        self.scenario_paths = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
        
        if not self.scenario_paths:
            raise FileNotFoundError(f"No scenario .pt files found in {data_dir}.")
        
        print(f"Initializing Stochastic Epoch BC Dataset with {len(self.scenario_paths)} scenarios.")
            
        if worker_info is None:
            self.start = 0
            self.end = len(self.scenario_paths)
        else:
            per_worker = int(np.ceil(len(self.scenario_paths) / float(worker_info.num_workers)))
            self.start = worker_info.id * per_worker
            self.end = min(self.start + per_worker, len(self.scenario_paths))

    def __iter__(self):
        worker_scenario_paths = self.scenario_paths[self.start:self.end]
        random.shuffle(worker_scenario_paths)
        
        for scenario_path in worker_scenario_paths:
            # 1. Load one full scenario from disk (list of dicts with NumPy arrays)
            raw_scenario_data = torch.load(scenario_path, weights_only=False)
            
            # --- THIS IS THE CRITICAL OPTIMIZATION ---
            # 2. Pre-convert the entire scenario to Tensors at once.
            processed_scenario_data = []
            for sample in raw_scenario_data:
                state_dict_tensor = {
                    k: torch.from_numpy(v) for k, v in sample['state'].items()
                }
                processed_scenario_data.append({
                    'state': state_dict_tensor,
                    'action': sample['action'] # Already a tensor
                })
            
            num_samples_in_scenario = len(processed_scenario_data)
            if num_samples_in_scenario == 0:
                continue

            # 3. Randomly select k indices from this scenario
            sample_indices = np.random.choice(
                num_samples_in_scenario, 
                size=self.k, 
                replace=(num_samples_in_scenario < self.k)
            )
            
            # 4. Yield the pre-tensorized samples
            for i in sample_indices:
                sample = processed_scenario_data[i]
                yield sample['state'], sample['action']
                
# --- Main Training Logic ---
def train(config: TrainingConfig):
    print(f"Starting Structured BC training on device: {config.device}")
    # --- 1. Setup TensorBoard ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"BCS_v2_{timestamp}_lr{config.learning_rate}_bs{config.batch_size}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    os.makedirs(os.path.dirname(config.bcs_model_path), exist_ok=True)
    
    # --- Datasets and DataLoaders (UPDATED) ---
    # train_dataset = BCDataset(config.cql_preprocess_dir)
    val_dataset = BCDataset(config.cql_preprocess_val_dir)
    k_train = 2 # Sample 2 random timesteps from each training scenario per epoch
    train_dataset = StochasticEpochBCDataset(config.cql_preprocess_dir, k_samples_per_scenario=k_train)
    
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        # Re-initialize the dataset for this specific worker to get its slice
        dataset.__init__(dataset.data_dir, dataset.k, worker_info=worker_info)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False, # Shuffle must be False
        num_workers=config.num_workers, worker_init_fn=worker_init_fn,
        collate_fn=structured_collate_fn, pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,

    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        collate_fn=structured_collate_fn,
    )
    # --- Model, Optimizer, Loss ---
    # Instantiate our new StructuredBCModel
    model = StructuredMLP_BC_Model(config).to(config.device)
    
    # Load previous checkpoint if it exists
    if os.path.exists(config.bcs_model_path_v2):
        print(f"Loading existing model from {config.bcs_model_path_v2}")
        model.load_state_dict(torch.load(config.bcs_model_path_v2, map_location=config.device))
    else:
        print("No existing model found, starting from scratch.")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5) # Added small weight decay
    
    weights_path = os.path.join(os.path.dirname(config.bcs_model_path_v2), 'action_weights_stage_2_6.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Action weights not found at {weights_path}. Please run compute_action_weights.py first.")
    loss_fn = WeightedMSELoss(weights_path).to(config.device)
    
    best_val_loss = float('inf')
    
    # --- Main Training Loop ---
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0
        for states_dict, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]"):
            if states_dict is None:
                print("Warning: Encountered empty batch, skipping.")
                continue # Skip empty batches
            
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
        all_pred_actions = []

        with torch.no_grad():
            for states_dict, actions in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]"):
                if states_dict is None: continue # Skip empty batches
                
                states_dict = {k: v.to(config.device, non_blocking=True) for k, v in states_dict.items()}
                actions = actions.to(config.device, non_blocking=True)
                pred_actions = model(states_dict)
                loss = loss_fn(pred_actions, actions)
                val_loss += loss.item()
                val_batches += 1
                all_pred_actions.append(pred_actions.cpu())


        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        
        # Log Performance Metrics
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        
        # Log Action Distribution Metrics
        if all_pred_actions:
            all_pred_actions = torch.cat(all_pred_actions, dim=0).numpy()
            pred_accel = all_pred_actions[:, 0]
            pred_steer = all_pred_actions[:, 1]
            writer.add_scalar('Action/Pred_Accel_Mean', np.mean(pred_accel), epoch)
            writer.add_scalar('Action/Pred_Accel_Std', np.std(pred_accel), epoch)
            writer.add_scalar('Action/Pred_Steer_Mean', np.mean(pred_steer), epoch)
            writer.add_scalar('Action/Pred_Steer_Std', np.std(pred_steer), epoch)
            
            # Log Histograms for a richer view
            writer.add_histogram('Action/Pred_Accel_Hist', pred_accel, epoch)
            writer.add_histogram('Action/Pred_Steer_Hist', pred_steer, epoch)
        
        # Log System Performance
        epoch_time = time.time() - epoch_start_time
        # samples_per_second = len(train_dataset) / epoch_time
        writer.add_scalar('Performance/Epoch_Time_seconds', epoch_time, epoch)
        # writer.add_scalar('Performance/Samples_per_Second', samples_per_second, epoch)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.bcs_model_path_v2)
            print(f"  -> New best model saved to {config.bcs_model_path_v2} (Val Loss: {best_val_loss:.6f})")

    print("\n--- Structured BC Training with Weighted Loss Complete ---")

if __name__ == "__main__":
    config = TrainingConfig()
    train(config)