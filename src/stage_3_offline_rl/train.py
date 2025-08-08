# ======================================================================================
# train.py (Stage 3 - Offline RL - With TensorBoard Logging)
#
# Description:
#   The main training script for the Conservative Q-Learning (CQL) agent, now with 
#   integrated TensorBoard logging to visualize training metrics and monitor run health.
#   This script orchestrates the entire training process:
#   1. Initializes the RLDataset to load (s, a, r, s', d) tuples.
#   2. Initializes the Actor and Critic networks.
#   3. Initializes the CQLTrainer with the networks and hyperparameters.
#   4. Runs the main training loop, calling the trainer at each step.
#   5. Logs progress and saves model checkpoints.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

# Import our project's core components for Stage 3
from src.shared.utils import TrainingConfig
from src.stage_3_offline_rl.dataset import RLDataset
from src.stage_3_offline_rl.cql_trainer import CQLTrainer

def train(config: TrainingConfig):
    print(f"--- Starting Stage 3: CQL Offline RL Training ---")
    
    # --- 1. Setup ---
    # --- NEW: Create a unique directory for this training run's logs and models ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"cql_{timestamp}"
    log_dir = os.path.join("runs", run_name)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize the TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # --- Load Normalization Stats in the Main Script ---
    # print("Loading state normalization statistics...")
    # stats_path = os.path.join(os.path.dirname(config.cql_model_path), 'state_stats.pt')
    # if not os.path.exists(stats_path):
        # raise FileNotFoundError(f"State stats not found at {stats_path}. Please run compute_state_stats.py first.")
    
    # stats = torch.load(stats_path, weights_only=True)
    # state_mean = stats['mean'].to(config.device)
    # state_std = stats['std'].to(config.device)
    # print("...Statistics loaded and moved to GPU.")

    # --- 2. Datasets and DataLoaders ---
    train_dataset = RLDataset(config.cql_preprocess_dir, config)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
        # We need a custom collate_fn for the dictionary structure
        collate_fn=lambda batch: torch.utils.data.default_collate(batch)
    )
    
    # --- 3. Initialize the CQL Trainer ---
    trainer = CQLTrainer(config=config, device=config.device)


    # --- 4. Main Training Loop ---
    print(f"Starting training for {config.num_epochs} epochs...")
    global_step = 0
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")):
            # batch = [item.to(config.device, non_blocking=True) for item in batch]
            # Unpack the un-normalized batch
            state_dict, action, reward, next_state_dict, done = batch
            
            reward_signal = reward # Get the reward tensor
            
            # Move the components to the GPU
            # For dictionaries, we must iterate through their values
            state_dict = {k: v.to(config.device, non_blocking=True) for k, v in state_dict.items()}
            action = action.to(config.device, non_blocking=True)
            reward = reward.to(config.device, non_blocking=True)
            next_state_dict = {k: v.to(config.device, non_blocking=True) for k, v in next_state_dict.items()}
            done = done.to(config.device, non_blocking=True)

            # --- NEW: Apply Normalization on the GPU ---
            # state = (state - state_mean) / state_std
            # next_state = (next_state - state_mean) / state_std

            # Re-assemble the batch for the trainer
            final_batch = (state_dict, action, reward, next_state_dict, done)
            
            # Perform a single training step
            losses = trainer.train_step(final_batch)
            
            # --- 5. Logging with TensorBoard ---
            # Log all the important metrics at each training step
            if (i + 1) % 100 == 0: # Log every 100 steps for finer detail
                writer.add_scalar('Loss/Critic_Loss', losses['critic_loss'], global_step)
                writer.add_scalar('Loss/Actor_Loss', losses['actor_loss'], global_step)
                writer.add_scalar('Loss/Alpha_Loss', losses['alpha_loss'], global_step)
                writer.add_scalar('Parameters/Alpha', losses['alpha'], global_step)
                writer.add_scalar('CQL/CQL_Term_Q1', losses['cql_loss_q1'], global_step)
                
                # Log the reward signal for monitoring
                writer.add_scalar('Reward/Mean', reward_signal.mean().item(), global_step)
                writer.add_scalar('Reward/StdDev', reward_signal.std().item(), global_step)
                writer.add_scalar('Reward/Max', reward_signal.max().item(), global_step)
                writer.add_scalar('Reward/Min', reward_signal.min().item(), global_step)
                    
            global_step += 1
        
        # --- 6. Checkpointing ---
        # Save the model at the end of each epoch
        checkpoint_path = os.path.join(model_dir, f"cql_policy_epoch_{epoch+1}.pth")
        trainer.save(checkpoint_path)
        tqdm.write(f"  -> Model checkpoint saved to {checkpoint_path}")

    writer.close()
    print("\n--- CQL Training Complete ---")
    print(f"Final model and logs saved in: {log_dir}")

# --- Entry Point ---
if __name__ == "__main__":
    config = TrainingConfig()
    train(config)