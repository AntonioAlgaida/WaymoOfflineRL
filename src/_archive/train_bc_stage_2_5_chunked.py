# ======================================================================================
# train_bc_stage_2_5.py
#
# Description:
#   This script trains the Kinematic Behavioral Cloning (BC-K) agent.
#   It uses a fast PyTorch Dataset to load preprocessed state and kinematic
#   action pairs (acceleration, steering) and trains an MLP model to map
#   states to these physically plausible actions.
#
# Author: Antonio Guillen-Perez
# Date: 2025-07-31
# ======================================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm

# Import our new, centralized configuration and utilities
from utils import TrainingConfig, WeightedMSELoss 

# --- 1. The Data Pipeline (Fast Version) ---
class BCDataset(Dataset):
    """
    A dataset that loads data from large chunk files to improve I/O performance.
    It can operate in two modes:
    1. Preload all data into RAM (fastest if dataset fits).
    2. Load chunks on-the-fly (slower but handles huge datasets).
    """
    def __init__(self, data_dir, preload_to_ram=True):
        print(f"Initializing chunked dataset from: {data_dir}")
        self.chunk_paths = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
        if not self.chunk_paths:
            raise FileNotFoundError(f"No chunk .pt files found in {data_dir}.")
        
        self.preload_to_ram = preload_to_ram
        self.samples = []
        
        if self.preload_to_ram:
            print(f"Preloading all {len(self.chunk_paths)} chunks into RAM...")
            for chunk_path in tqdm(self.chunk_paths, desc="Loading chunks"):
                self.samples.extend(torch.load(chunk_path, weights_only=False))
            print(f"Preloading complete. Total samples: {len(self.samples)}")
        else:
            # In on-the-fly mode, we need to know how many samples are in each chunk
            # This is slower to initialize but saves RAM.
            self.chunk_lengths = [len(torch.load(p, weights_only=False)) for p in self.chunk_paths]
            self.cumulative_lengths = np.cumsum(self.chunk_lengths)
            self.total_samples = self.cumulative_lengths[-1]
            self.loaded_chunk_cache = {} # Cache for the most recently loaded chunk

    def __len__(self):
        return len(self.samples) if self.preload_to_ram else self.total_samples

    def __getitem__(self, idx):
        if self.preload_to_ram:
            sample = self.samples[idx]
            return sample['state'], sample['action']
        else:
            # On-the-fly loading logic
            # Find which chunk the index belongs to
            chunk_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
            
            # Check cache first
            if chunk_idx in self.loaded_chunk_cache:
                chunk_data = self.loaded_chunk_cache[chunk_idx]
            else: # Load new chunk
                chunk_data = torch.load(self.chunk_paths[chunk_idx], weights_only=False)
                self.loaded_chunk_cache = {chunk_idx: chunk_data} # Simple cache

            # Find the index within the chunk
            start_idx = self.cumulative_lengths[chunk_idx-1] if chunk_idx > 0 else 0
            local_idx = idx - start_idx
            
            sample = chunk_data[local_idx]
            return sample['state'], sample['action']

# --- 2. The Model ---
class MLP(nn.Module):
    """
    The neural network policy model. Its architecture is identical to the
    previous stage, but it will learn a different mapping.
    """
    def __init__(self, input_dim=279, output_dim=2, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# --- 3. The Training Logic ---
def train(config):
    print(f"Starting Kinematic BC training on device: {config.device}")
    
    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(config.bck_model_path), exist_ok=True)
    
    # Datasets and DataLoaders
    # Point to the new kinematic preprocessed data directories from our config
    train_dataset = BCDataset(config.bck_preprocess_train_dir)
    val_dataset = BCDataset(config.bck_preprocess_val_dir)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    
    # Model, Optimizer, Loss Function
    model = MLP().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4) 
    
    # Define the path to our pre-computed weights
    weights_path = os.path.join(os.path.dirname(config.bck_model_path), 'action_weights.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Action weights not found at {weights_path}. Please run compute_action_weights.py first.")
        
    # Instantiate and move our custom loss function to the GPU
    loss_fn = WeightedMSELoss(weights_path).to(config.device)
    
    best_val_loss = float('inf')
    
    # Main Training Loop
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        # Use non_blocking=True for a potential small speedup in data transfer
        for states, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]"):
            states, actions = states.to(config.device, non_blocking=True), actions.to(config.device, non_blocking=True)
            
            optimizer.zero_grad()
            pred_actions = model(states)
            loss = loss_fn(pred_actions, actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, actions in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]"):
                states, actions = states.to(config.device, non_blocking=True), actions.to(config.device, non_blocking=True)
                pred_actions = model(states)
                loss = loss_fn(pred_actions, actions)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Checkpointing: Save the model if it has the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save to the new model path from our config
            torch.save(model.state_dict(), config.bck_model_path)
            print(f"  -> New best model saved to {config.bck_model_path} (Val Loss: {best_val_loss:.6f})")

    print("\n--- Kinematic Training Complete ---")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final model saved at: {config.bck_model_path}")

# --- 4. Entry Point ---
if __name__ == "__main__":
    # Load the single, centralized config
    config = TrainingConfig()
    train(config)