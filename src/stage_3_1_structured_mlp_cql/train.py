# ======================================================================================
# train.py (Stage 3.1 - Structured MLP CQL)
#
# Description:
#   The main training script for the structured MLP CQL agent.
#   This script orchestrates the entire training process, including:
#   - Loading the RL-specific, scenario-per-file dataset.
#   - Applying state normalization on the GPU.
#   - Calling the CQL trainer at each step.
#   - Logging a comprehensive set of metrics to TensorBoard.
#   - Saving model checkpoints.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import time

from torch.utils.tensorboard import SummaryWriter

# Import our project's core components
from src.shared.utils import TrainingConfig
from src.stage_3_1_structured_mlp_cql.dataset import StochasticEpochRLDataset
from src.stage_3_1_structured_mlp_cql.cql_trainer import CQLTrainer

def rl_structured_collate_fn(batch):
    states_list = [item[0] for item in batch]
    actions_list = [item[1] for item in batch]
    rewards_list = [item[2] for item in batch]
    next_states_list = [item[3] for item in batch]
    dones_list = [item[4] for item in batch]
    
    batched_states_dict = {
        key: torch.stack([torch.from_numpy(s[key]) for s in states_list])
        for key in states_list[0].keys()
    }
    batched_next_states_dict = {
        key: torch.stack([torch.from_numpy(s[key]) for s in next_states_list])
        for key in next_states_list[0].keys()
    }
    
    batched_actions = torch.stack(actions_list)
    batched_rewards = torch.stack(rewards_list)
    batched_dones = torch.stack(dones_list)
    
    return batched_states_dict, batched_actions, batched_rewards, batched_next_states_dict, batched_dones

def train(config: TrainingConfig):
    print(f"--- Starting Stage 3.1: Structured MLP CQL Training ---")
    
    # --- 1. Setup Logging and Directories ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"CQL_MLP_{timestamp}_lr{config.learning_rate}_gamma{config.gamma}_cqlalpha{config.cql_alpha}"
    log_dir = os.path.join("runs", run_name)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    # Save the config for this run for reproducibility
    with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
        f.write(str(config))
    
    # --- 2. Load State Normalization Statistics ---
    # --- Load Normalization Stats (Corrected) ---
    print("Loading structured state normalization statistics...")
    stats_path = config.state_stats_structured_path
    stats = torch.load(stats_path, weights_only=True)

    # stats['mean'] is already the dictionary we want. We just iterate over it.
    state_mean_dict = {k: v.to(config.device) for k, v in stats['mean'].items()}
    state_std_dict = {k: v.to(config.device) for k, v in stats['std'].items()}
    
    # print("...Statistics loaded and moved to GPU.")
    # The worker_init_fn needs to be adapted for the new dataset args
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.__init__(dataset.data_dir, dataset.config, dataset.k, worker_info=worker_info)
    # --- 3. Datasets and DataLoaders ---
    # Use the RL-specific dataset that builds transitions
    k_train_transitions = 2 # A new control parameter
    train_dataset = StochasticEpochRLDataset(
        config.cql_preprocess_dir, 
        config, 
        k_samples_per_scenario=k_train_transitions
    )
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True if config.num_workers > 0 else False,
        collate_fn=rl_structured_collate_fn # This collator is now more complex
    )
    
    # --- 4. Initialize the CQL Trainer ---
    trainer = CQLTrainer(config=config, device=config.device)

    # --- 5. Main Training Loop ---
    print(f"Starting training for {config.num_epochs} epochs...")
    global_step = 0
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")):
            state_dict, action, reward, next_state_dict, done = batch
            
            # Move data to the GPU
            state_dict = {k: v.to(config.device, non_blocking=True) for k, v in state_dict.items()}
            action = action.to(config.device, non_blocking=True)
            reward = reward.to(config.device, non_blocking=True)
            next_state_dict = {k: v.to(config.device, non_blocking=True) for k, v in next_state_dict.items()}
            done = done.to(config.device, non_blocking=True)

            # Apply Normalization to state dictionaries
            state_dict = {k: (v - state_mean_dict[k]) / state_std_dict[k] for k, v in state_dict.items()}
            next_state_dict = {k: (v - state_mean_dict[k]) / state_std_dict[k] for k, v in next_state_dict.items()}
            
            normalized_batch = (state_dict, action, reward, next_state_dict, done)
            
            # Perform a single training step
            losses = trainer.train_step(normalized_batch)
            
            # --- Logging to TensorBoard ---
            if (i + 1) % 50 == 0:
                # Log all the important metrics for the paper
                writer.add_scalar('Loss/Critic_Loss', losses['critic_loss'], global_step)
                writer.add_scalar('Loss/Actor_Loss', losses['actor_loss'], global_step)
                writer.add_scalar('Loss/Alpha_Loss', losses['alpha_loss'], global_step)
                writer.add_scalar('Parameters/Alpha', losses['alpha'], global_step)
                writer.add_scalar('CQL/CQL_Term_Q1', losses['cql_loss_q1'], global_step)
                
                # Log reward signal for monitoring
                writer.add_scalar('Reward/Mean', reward.mean().item(), global_step)
                writer.add_scalar('Reward/StdDev', reward.std().item(), global_step)
                writer.add_scalar('Reward/Max', reward.max().item(), global_step)
                writer.add_scalar('Reward/Min', reward.min().item(), global_step)

                # Log Q-values to understand the scale of the value function
                # We can add Q-value logging inside the trainer and return it
                # writer.add_scalar('Q_Values/Mean_Q', losses['mean_q'], global_step)
                    
            global_step += 1
        
        # --- Checkpointing ---
        checkpoint_path = os.path.join(model_dir, f"cql_mlp_policy_epoch_{epoch+1}.pth")
        trainer.save(checkpoint_path)
        tqdm.write(f"  -> Model checkpoint saved to {checkpoint_path}")

    writer.close()
    print("\n--- CQL Training Complete ---")
    print(f"Final model and logs saved in: {log_dir}")

# --- Entry Point ---
if __name__ == "__main__":
    # Ensure all necessary paths and hyperparameters are in your TrainingConfig
    config = TrainingConfig()
    # Let's use the hyperparameters that we found to be more stable
    config.learning_rate = 3e-5
    config.gamma = 0.95
    config.cql_alpha = 10.0
    train(config)