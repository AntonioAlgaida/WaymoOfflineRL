# ======================================================================================
# dataset.py (Stage 3 - Offline RL)
#
# Description:
#   Defines the PyTorch Dataset for Offline Reinforcement Learning.
#   This class reads the "scenario-per-file" structured data, computes rewards
#   for each transition, and assembles the full (s, a, r, s', d) tuples
#   required by the CQL algorithm.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from tqdm import tqdm

# Import our project's core components
from src.shared.utils import TrainingConfig
from src.stage_3_offline_rl.reward_function import compute_reward

class RLDataset(Dataset):
    """
    PyTorch Dataset for loading transitions for Offline Reinforcement Learning.
    Each item is a tuple of (state, action, reward, next_state, done).
    It operates on a "scenario-per-file" dataset format.
    """
    def __init__(self, data_dir: str, config: TrainingConfig):
        self.config = config
        scenario_paths = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
        if not scenario_paths:
            raise FileNotFoundError(f"No scenario .pt files found in {data_dir}. Please run the RL preprocessor.")
            
        self.master_index = []
        
        # Use only 10% of the scenarios for debugging purposes
        scenario_paths = scenario_paths[:len(scenario_paths) // 50]
        print(f"Initializing RLDataset: Building master index from {len(scenario_paths)} scenarios...")

        # This one-time scan builds a map of all valid transitions in the dataset.
        for scenario_path in tqdm(scenario_paths, desc="Indexing RL transitions"):
            # Load the list of samples for the entire scenario.
            # This is I/O intensive but only happens once at startup.
            scenario_samples = torch.load(scenario_path, weights_only=False)
            
            # A transition is valid if sample `i` and `i+1` are consecutive in time.
            for i in range(len(scenario_samples) - 1):
                current_ts = scenario_samples[i]['timestep']
                next_ts = scenario_samples[i+1]['timestep']
                
                # Check for a continuous, unbroken sequence
                if next_ts == current_ts + 1:
                    # Add a pointer to the start of this valid transition
                    self.master_index.append((scenario_path, i))

        # A simple cache for the most recently loaded scenario to reduce I/O
        self.cache = {}
        
        print(f"Indexing complete. Found {len(self.master_index)} total valid transitions.")

    def __len__(self):
        """Returns the total number of valid transitions."""
        return len(self.master_index)

    def __getitem__(self, idx: int) -> tuple:
        """
        Fetches a single (s, a, r, s', d) transition tuple.
        """
        # 1. Look up the file path and the index for the start of the transition
        scenario_path, inner_index = self.master_index[idx]
        
        # 2. Load the scenario data, using a cache to speed things up
        if scenario_path in self.cache:
            scenario_data = self.cache[scenario_path]
        else:
            scenario_data = torch.load(scenario_path, weights_only=False)
            # Update cache (simple last-item cache)
            self.cache = {scenario_path: scenario_data}
            
        # 3. Get the current and next samples from the loaded scenario
        current_sample = scenario_data[inner_index]
        next_sample = scenario_data[inner_index + 1]
        
        # 4. Extract state, action, and next_state
        state = current_sample['state']
        action = current_sample['action']
        next_state = next_sample['state']
        
        # 4.5 Extract the timestep for the current sample and the next sample
        current_timestep = current_sample['timestep']
        next_timestep = next_sample['timestep']
        # Raise an error if the timesteps are not consecutive
        if next_timestep != current_timestep + 1:
            raise ValueError(f"Non-consecutive timesteps found: {current_timestep} -> {next_timestep} in {scenario_path} at index {inner_index}.")
        
        
        # 5. Compute Reward using our external reward function
        # The reward is for being in `state` and taking `action`.
        reward = compute_reward(state)
        
        # 6. Determine the `done` flag
        # The episode is "done" if the *next* state is the last possible timestep (89 + 1 = 90).
        # Since our preprocessor generates 90 steps (0-89), the last timestep is 89.
        # A transition from 88->89 means the next state is terminal.
        done = 1.0 if current_sample['timestep'] == 88 else 0.0
        
        # 7. Return the complete tuple
        return (state, action, torch.tensor(reward, dtype=torch.float32), 
                next_state, torch.tensor(done, dtype=torch.float32))