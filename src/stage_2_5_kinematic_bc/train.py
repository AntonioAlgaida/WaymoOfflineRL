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
# ======================================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm

# Import our new, centralized configuration and utilities
from src.shared.utils import TrainingConfig, WeightedMSELoss 

# --- 1. The Data Pipeline (Fast Version) ---
class BCDataset(Dataset):
    """
    A very fast dataset that loads pre-computed state and action tensors.
    """
    def __init__(self, data_dir):
        print(f"Scanning for preprocessed .pt files in: {data_dir}")
        self.file_paths = glob.glob(os.path.join(data_dir, '*.pt'))
        if not self.file_paths:
            raise FileNotFoundError(f"No preprocessed .pt files found in {data_dir}. Did you run preprocess_bc_stage_2_5.py?")
        print(f"Found {len(self.file_paths)} preprocessed samples.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the pre-saved dictionary containing state and action tensors
        # Use weights_only=False because we are loading a dict, not just weights.
        # This is safe as we created the files ourselves.
        data = torch.load(self.file_paths[idx], weights_only=False)
        return data['state'], data['action']

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