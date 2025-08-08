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
    train_data_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/processed_training')
    val_data_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/processed_validation')
    model_save_path: str = os.path.expanduser('~/WaymoOfflineAgent/models/bc_policy_full.pth')
    
    # Training Hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 512
    num_epochs: int = 50
    num_workers: int = 8
    
    # Feature Engineering
    num_closest_agents: int = 15
    num_closest_map_points: int = 50
    max_dist: float = 99.0 # Default distance for unseen rules
    
    # Execution
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. The Data Pipeline ---
class BCDataset(Dataset):
    def __init__(self, data_dir, config):
        self.config = config
        self.master_index = []
        print(f"Scanning directory: {data_dir}")
        for file_path in tqdm(glob.glob(os.path.join(data_dir, '*.npz')), desc="Indexing files"):
            # Each scenario has 91 steps, so 90 possible transitions (from t=0..89 to t=1..90)
            for i in range(90):
                self.master_index.append((file_path, i))

    def __len__(self):
        return len(self.master_index)

    def _get_ego_centric_transform(self, x, y, heading):
        c, s = np.cos(heading), np.sin(heading)
        rot_matrix = np.array([[c, s], [-s, c]])
        return rot_matrix, np.array([x, y])

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def __getitem__(self, idx):
        file_path, time_step_index = self.master_index[idx]
        data = np.load(file_path, allow_pickle=True)
        
        # --- A. Establish Frame of Reference & Get Action ---
        sdc_index = int(data['sdc_track_index'])
        ego_state = data['all_trajectories'][sdc_index, time_step_index]
        ego_x, ego_y, ego_heading = ego_state[0], ego_state[1], ego_state[6]
        rot_matrix, ego_pos = self._get_ego_centric_transform(ego_x, ego_y, ego_heading)

        next_ego_state = data['all_trajectories'][sdc_index, time_step_index + 1]
        action_global = np.array([next_ego_state[0] - ego_x, next_ego_state[1] - ego_y])
        action_ego = np.dot(rot_matrix, action_global)

        # --- B. Engineer State Vector ---
        
        # 1. Ego Features (3 features)
        ego_vel_x, ego_vel_y = ego_state[7], ego_state[8]
        current_speed = np.linalg.norm([ego_vel_x, ego_vel_y])
        
        if time_step_index > 0:
            prev_ego_state = data['all_trajectories'][sdc_index, time_step_index - 1]
            prev_speed = np.linalg.norm([prev_ego_state[7], prev_ego_state[8]]) # state.velocity_x and state.velocity_y
            acceleration = (current_speed - prev_speed) / 0.1
            yaw_rate = self._normalize_angle(ego_heading - prev_ego_state[6]) / 0.1
        else: # First step, no previous state
            acceleration = 0.0
            yaw_rate = 0.0
        ego_features = np.array([current_speed, acceleration, yaw_rate])

        # 2. Agent Features (15 agents * 10 features = 150 features)
        agent_features_list = []
        valid_agents_states = []
        for i in range(len(data['object_ids'])):
            if i != sdc_index and data['all_trajectories'][i, time_step_index, 9] > 0:
                agent_state = data['all_trajectories'][i, time_step_index]
                dist = np.linalg.norm(agent_state[:2] - ego_pos)  # [(Center x, y) - (ego x, y)]
                valid_agents_states.append((dist, agent_state, data['object_types'][i]))
        
        try:
            valid_agents_states.sort(key=lambda item: item[0])
        except Exception as e:
            print(f"Error sorting agents for index {idx} in file {file_path}: {e}")
            print(f"Valid agents states: {valid_agents_states}")
            raise ValueError(f"Failed to sort agents for index {idx} in file {file_path}")
        
        for _, agent_state, obj_type in valid_agents_states[:self.config.num_closest_agents]:
            relative_pos = np.dot(rot_matrix, agent_state[:2] - ego_pos)
            relative_vel = np.dot(rot_matrix, agent_state[7:9])
            relative_heading = self._normalize_angle(agent_state[6] - ego_heading)
            
            type_vec = np.zeros(3) # Vehicle, Pedestrian, Cyclist
            if obj_type in [1, 2, 3]: type_vec[obj_type - 1] = 1.0

            agent_features = np.concatenate([relative_pos, relative_vel, [relative_heading, agent_state[3], agent_state[4]], type_vec])
            agent_features_list.append(agent_features)
            
        num_found = len(agent_features_list)
        if num_found < self.config.num_closest_agents:
            padding = np.zeros((self.config.num_closest_agents - num_found, 10))
            agent_features_list.extend(padding)
        
        try:
            all_agent_features = np.concatenate(agent_features_list).flatten()
        except ValueError as e:
            print(f"Error concatenating agent features: {e}")
            # Raise an error
            raise ValueError(f"Failed to concatenate agent features for index {idx} in file {file_path}")
        
        # 3. Roadgraph Features (50 points * 2 features = 100 features)
        map_data = data['map_data'].item()
        lane_points = list(map_data['lane_polylines'].values())
        if lane_points:
            all_lane_points = np.vstack(lane_points)[:, :2]
            distances = np.linalg.norm(all_lane_points - ego_pos, axis=1)
            closest_indices = np.argsort(distances)[:self.config.num_closest_map_points]
            closest_points = all_lane_points[closest_indices]
            
            relative_map_points = np.dot(rot_matrix, (closest_points - ego_pos).T).T
            
            num_found_points = len(relative_map_points)
            if num_found_points < self.config.num_closest_map_points:
                padding = np.zeros((self.config.num_closest_map_points - num_found_points, 2))
                relative_map_points = np.vstack([relative_map_points, padding])
        else:
            relative_map_points = np.zeros((self.config.num_closest_map_points, 2))
        all_map_features = relative_map_points.flatten()

        # 4 & 5. Rule-Based Features (7 features)
        # (Simplified implementation for now, full graph traversal can be complex)
        dist_to_stop_sign, is_stop_controlled = self.config.max_dist, 0.0
        dist_to_crosswalk = self.config.max_dist
        tl_state_vec = np.array([0, 0, 0, 1.0]) # Default to unknown
        
        # This simple version just finds the nearest of each type, not necessarily "on path"
        if map_data['stopsign_positions']:
            stop_sign_positions = np.array([d['position'][:2] for d in map_data['stopsign_positions'].values()])
            dist_to_stop_sign = np.min(np.linalg.norm(stop_sign_positions - ego_pos, axis=1))
        
        if map_data['crosswalk_polygons']:
            crosswalk_points = np.vstack([p[:, :2] for p in map_data['crosswalk_polygons'].values()])
            dist_to_crosswalk = np.min(np.linalg.norm(crosswalk_points - ego_pos, axis=1))

        all_rules_features = np.array([dist_to_stop_sign, is_stop_controlled, dist_to_crosswalk])
        all_rules_features = np.concatenate([all_rules_features, tl_state_vec])

        # Final state vector
        state_vector = np.concatenate([ego_features, all_agent_features, all_map_features, all_rules_features])
        
        return torch.tensor(state_vector, dtype=torch.float32), torch.tensor(action_ego, dtype=torch.float32)

# --- 3. The Model ---
class MLP(nn.Module):
    def __init__(self, input_dim=260, output_dim=2, hidden_dim=512):
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
    train_dataset = BCDataset(config.train_data_dir, config)
    val_dataset = BCDataset(config.val_data_dir, config)
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
            states, actions = states.to(config.device), actions.to(config.device)
            
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