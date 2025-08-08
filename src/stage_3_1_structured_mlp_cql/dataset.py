# ======================================================================================
# dataset.py (Stage 3.1 - RL Dataset)
# ======================================================================================
import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import os
import glob
from tqdm import tqdm
import random

from src.shared.utils import TrainingConfig
from src.stage_3_1_structured_mlp_cql.reward_function import compute_reward

# class RLDataset(Dataset):
#     def __init__(self, data_dir: str, config: TrainingConfig):
#         self.config = config
#         scenario_paths = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
#         self.master_index = []
#         # Reduce the number of scenarios to 1000 for faster testing
#         scenario_paths = scenario_paths[:100000]
#         for scenario_path in tqdm(scenario_paths, desc="Indexing RL transitions"):
#             num_samples = len(torch.load(scenario_path, weights_only=False))
#             for i in range(num_samples - 1):
#                 self.master_index.append((scenario_path, i))
#         self.cache = {'path': None, 'data': None}

#     def __len__(self):
#         return len(self.master_index)

#     def __getitem__(self, idx: int) -> tuple:
#         scenario_path, inner_index = self.master_index[idx]
        
#         if scenario_path in self.cache:
#             scenario_data = self.cache[scenario_path]
#         else:
#             scenario_data = torch.load(scenario_path, weights_only=False)
#             self.cache = {scenario_path: scenario_data}
            
#         current_sample = scenario_data[inner_index]
#         next_sample = scenario_data[inner_index + 1]
        
#         # --- 1. Extract the raw data ---
#         state_dict_np = current_sample['state']    # This is a dict of NumPy arrays
#         action = current_sample['action']          # This is a Torch tensor
#         next_state_dict_np = next_sample['state']  # This is a dict of NumPy arrays
        
#         # --- 2. Compute Reward ---
#         # The state_dict is already in the correct NumPy format. Pass it directly.
#         reward = compute_reward(state_dict_np)
        
#         # --- 3. Determine `done` flag ---
#         done = 1.0 if current_sample['timestep'] == 88 else 0.0
        
#         # --- 4. Convert state dicts to tensors for the collate_fn ---
#         # The collate_fn expects all tensors, so we must convert the state dicts here.
#         state_dict_torch = {k: torch.from_numpy(v) for k, v in state_dict_np.items()}
#         next_state_dict_torch = {k: torch.from_numpy(v) for k, v in next_state_dict_np.items()}
        
#         # --- 5. Return the complete tuple of TENSORS ---
#         return (state_dict_torch, action, torch.tensor(reward, dtype=torch.float32), 
#                 next_state_dict_torch, torch.tensor(done, dtype=torch.float32))


class StochasticEpochRLDataset(IterableDataset):
    """
    An iterable dataset for Offline RL that implements the "stochastic epoch" strategy.
    
    In each epoch, it shuffles all scenarios and yields a small, random subset of k
    valid transitions from each one. This maximizes diversity and dramatically
    speeds up epoch times on very large datasets.
    """
    def __init__(self, data_dir: str, config: TrainingConfig, k_samples_per_scenario: int, worker_info=None):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.k = k_samples_per_scenario
        
        self.scenario_paths = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
        if not self.scenario_paths:
            raise FileNotFoundError(f"No scenario .pt files found in {data_dir}.")
            
        # For multi-worker data loading
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
            try:
                scenario_data = torch.load(scenario_path, weights_only=False)
                num_transitions = len(scenario_data) - 1
                if num_transitions <= 0: continue

                start_indices = np.random.choice(
                    num_transitions, size=self.k, replace=(num_transitions < self.k)
                )
                
                for i in start_indices:
                    current_sample = scenario_data[i]
                    next_sample = scenario_data[i+1]
                    
                    state_dict_np = current_sample['state']
                    action_tensor = current_sample['action']
                    next_state_dict_np = next_sample['state']
                    
                    # Compute reward using the NumPy dictionaries
                    reward = compute_reward(state_dict_np)
                    
                    done = 1.0 if next_sample['timestep'] == 89 else 0.0
                    
                    # --- FIX: Yield NumPy dicts. Let collate_fn handle tensor conversion. ---
                    yield (state_dict_np, action_tensor, torch.tensor(reward, dtype=torch.float32),
                           next_state_dict_np, torch.tensor(done, dtype=torch.float32))

            except Exception as e:
                print(f"Error processing scenario {scenario_path}: {e}")
                # print(f"Error processing scenario {scenario_path}. Skipping...")
                continue # Silently skip corrupted files