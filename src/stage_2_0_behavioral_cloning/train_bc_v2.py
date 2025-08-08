# ======================================================================================
# train_bc.py
#
# Description:
#   This script trains a Behavioral Cloning (BC) agent for autonomous driving.
#   It uses a custom PyTorch Dataset to load and process data from the .npz files
#   created by parser.py, trains a simple MLP model, and saves the best-performing
#   model based on validation loss.
#   Use: conda activate waymax-rl
#
# Author: Antonio Guillen-Perez
# Date: 2025-07-29
# ======================================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from dataclasses import dataclass
from tqdm import tqdm
import argparse

# --- 1. Configuration ---
@dataclass
class TrainingConfig:
    # Paths
    # Data directories for preprocessing the training and validation data.
    train_data_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/processed_training')
    val_data_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/processed_validation')
    # Preprocessed directories for BC training and validation data.
    preprocessed_train_data_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/bc_preprocessed_training')
    preprocessed_val_data_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/bc_preprocessed_validation')
    
    model_save_path: str = os.path.expanduser('~/WaymoOfflineAgent/models/bc_policy_full.pth')
    
    # Training Hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 512
    num_epochs: int = 150
    num_workers: int = 12 # Number of CPU workers for DataLoader
    
    # Feature Engineering
    num_closest_agents: int = 15
    num_closest_map_points: int = 50
    num_closest_crosswalk_points: int = 10
    max_dist: float = 99.0 # Default distance for unseen rules
    
    # Execution
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. The Data Pipeline ---
class BCDataset(Dataset):
    """
    A very fast dataset that loads pre-computed state and action tensors.
    All the heavy feature engineering has already been done offline.
    """
    def __init__(self, data_dir, config):
        self.config = config
        print(f"Scanning for preprocessed .pt files in: {data_dir}")
        self.file_paths = glob.glob(os.path.join(data_dir, '*.pt'))
        if not self.file_paths:
            raise FileNotFoundError(f"No preprocessed .pt files found in {data_dir}. Did you run preprocess_for_bc.py?")
        print(f"Found {len(self.file_paths)} preprocessed samples.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # The entire logic is now just loading a single pre-saved file.
        # This is extremely fast.
        data = torch.load(self.file_paths[idx], weights_only=False)
        
        # if the shape of data['state'] is not (279,), raise an error
        # if data['state'].shape != (279,):
            # raise ValueError(f"Expected state shape (279,), got {data['state'].shape} in file {self.file_paths[idx]}")
        return data['state'], data['action']

# --- 3. The Model ---
class MLP(nn.Module):
    def __init__(self, input_dim=279, output_dim=2, hidden_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# --- 4. The Training Logic ---
def train(config):
    print(f"Starting training on device: {config.device}")
    
    # Setup
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    
    # Datasets and DataLoaders
    train_dataset = BCDataset(config.preprocessed_train_data_dir, config)
    val_dataset = BCDataset(config.preprocessed_val_data_dir, config)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, # Use multiple CPU cores
        pin_memory=True,                # Speed up CPU-to-GPU transfer
        persistent_workers=True,         # Keep workers alive between epochs
        pin_memory_device=config.device
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        pin_memory_device=config.device

    )
    
    # Model, Optimizer, Loss Function
    model = MLP().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')
    
    # Main Training Loop
    for epoch in range(config.num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for states, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]"):
            # print(f"Processing batch of size {states.size(0)}")
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
                states, actions = states.to(config.device), actions.to(config.device)
                pred_actions = model(states)
                loss = loss_fn(pred_actions, actions)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  -> New best model saved to {config.model_save_path} (Val Loss: {best_val_loss:.6f})")

    print("\n--- Training Complete ---")
    print(f"Best validation loss: {best_val_loss:.6f}")

# --- 5. Entry Point ---
if __name__ == "__main__":
    # For now, we use the default config. Argparse can be added later.
    config = TrainingConfig()
    train(config)