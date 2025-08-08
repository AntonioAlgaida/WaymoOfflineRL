# ======================================================================================
# preprocess.py (Stage 2.6 - Scenario-per-File & Structured)
#
# Description:
#   This script preprocesses the .npz data into a structured dictionary format
#   and saves the results into one .pt file per scenario. This provides a balance
#   of I/O performance and simplicity, and is suitable for both BC and RL datasets.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import multiprocessing
import jax
from jax import numpy as jnp
import uuid 

# Import from our project and Waymax
from src.shared.utils import TrainingConfig, normalize_angle
from waymax import datatypes
from waymax.dynamics import bicycle_model

# --- NEW: Define Action Filtering Thresholds ---
MAX_ACCELERATION = 8.0  # m/s^2, very strong acceleration
MIN_ACCELERATION = -10.0 # m/s^2, emergency braking
MAX_STEERING_ANGLE = 0.8 # radians, ~45 degrees, a very sharp turn


def process_single_scenario_file(task_tuple):
    """
    Worker function. Processes a single .npz file (one scenario) and saves
    all its valid timesteps into a single corresponding .pt file.
    """
    file_path, config, output_dir = task_tuple
    
    try:
        # --- The entire feature engineering logic is here ---
        # This part is identical to your previous version.
        # It processes one .npz file and generates a list of 90 samples.
        data = np.load(file_path, allow_pickle=True)
        sdc_index = int(data['sdc_track_index'])
        
        num_agents = data['object_ids'].shape[0]
        timestamps_micros = jnp.tile(data['timestamps_seconds'][None, :], (num_agents, 1)) * 1e6

        full_trajectory = datatypes.Trajectory(
            x=jnp.array(data['all_trajectories'][:, :, 0]),
            y=jnp.array(data['all_trajectories'][:, :, 1]),
            z=jnp.array(data['all_trajectories'][:, :, 2]),
            length=jnp.array(data['all_trajectories'][:, :, 3]),
            width=jnp.array(data['all_trajectories'][:, :, 4]),
            height=jnp.array(data['all_trajectories'][:, :, 5]),
            yaw=jnp.array(data['all_trajectories'][:, :, 6]),
            vel_x=jnp.array(data['all_trajectories'][:, :, 7]),
            vel_y=jnp.array(data['all_trajectories'][:, :, 8]),
            valid=jnp.array(data['all_trajectories'][:, :, 9], dtype=bool),
            timestamp_micros=jnp.array(timestamps_micros, dtype=jnp.int64),
        )
        
        # --- 2. Use `compute_inverse` to get all actions at once ---
        # We need to JIT-compile this function for performance
        jit_compute_inverse = jax.jit(bicycle_model.compute_inverse)
        
        sdc_index = int(data['sdc_track_index'])
        
        # --- Pre-computation for the entire file (remains the same) ---
        sdc_trajectory = data['all_trajectories'][sdc_index]
        sdc_positions_all_steps = sdc_trajectory[:, :2]
        sdc_headings_all_steps = sdc_trajectory[:, 6]
        sdc_velocities_all_steps = sdc_trajectory[:, 7:9]

        # --- 2. Vectorized Nearest Neighbor Search for Map Features ---
        map_data = data['map_data'].item()
        # (This entire block is unchanged from the previous optimized version)
        lane_polylines = list(map_data['lane_polylines'].values())
        if lane_polylines:
            all_lane_points = np.vstack(lane_polylines)[:, :2]
            dist_matrix_lanes = np.linalg.norm(sdc_positions_all_steps[:, np.newaxis, :] - all_lane_points[np.newaxis, :, :], axis=2)
            closest_lane_indices_all_steps = np.argsort(dist_matrix_lanes, axis=1)[:, :config.num_closest_map_points]
        else:
            all_lane_points = np.array([])
            closest_lane_indices_all_steps = np.zeros((91, config.num_closest_map_points), dtype=int)
        
        crosswalk_polygons = list(map_data['crosswalk_polygons'].values())
        if crosswalk_polygons:
            all_crosswalk_points = np.vstack(crosswalk_polygons)[:, :2]
            dist_matrix_cw = np.linalg.norm(sdc_positions_all_steps[:, np.newaxis, :] - all_crosswalk_points[np.newaxis, :, :], axis=2)
            closest_crosswalk_indices_all_steps = np.argsort(dist_matrix_cw, axis=1)[:, :config.num_closest_crosswalk_points]
        else:
            all_crosswalk_points = np.array([])
            closest_crosswalk_indices_all_steps = np.zeros((91, config.num_closest_crosswalk_points), dtype=int)
            
        if map_data['stopsign_positions']:
            stop_sign_positions = np.array([d['position'][:2] for d in map_data['stopsign_positions'].values()])
            dist_matrix_ss = np.linalg.norm(sdc_positions_all_steps[:, np.newaxis, :] - stop_sign_positions[np.newaxis, :, :], axis=2)
            dist_to_closest_stop_sign_all_steps = np.min(dist_matrix_ss, axis=1)
        else:
            dist_to_closest_stop_sign_all_steps = np.full((91,), config.max_dist, dtype=np.float32)

        # --- 3. Vectorized Agent Feature Search ---
        all_agent_trajectories = data['all_trajectories']
        agent_positions_all_steps_t = np.transpose(all_agent_trajectories[:, :, :2], (1, 0, 2))
        dist_matrix_agents = np.linalg.norm(sdc_positions_all_steps[:, np.newaxis, :] - agent_positions_all_steps_t, axis=2)
        dist_matrix_agents[:, sdc_index] = np.inf
        closest_agent_indices_all_steps = np.argsort(dist_matrix_agents, axis=1)
        
        # This list will hold the 0-90 valid samples for THIS scenario
        scenario_samples = []
        # --- 4. Main Loop ---
        for ts in range(90):
            # --- ACTION CALCULATION (THE KEY CHANGE FOR STAGE 2.5 with InvertibleBicycleModel dynamics) ---
            all_agents_actions = jit_compute_inverse(traj=full_trajectory, timestep=ts, dt=0.1)
            sdc_action = all_agents_actions.data[sdc_index] # Shape (2,) -> [accel, steer]
            # Extract the expert action from the model's output
            model_expert_acceleration, model_expert_steering = sdc_action[0], sdc_action[1]

            # 1. Calculate Expert Acceleration from change in speed
            current_speed = np.linalg.norm(sdc_velocities_all_steps[ts])

            # --- THIS IS THE NEW FILTERING LOGIC ---
            if not (MIN_ACCELERATION <= model_expert_acceleration <= MAX_ACCELERATION and
                    -MAX_STEERING_ANGLE <= model_expert_steering <= MAX_STEERING_ANGLE):
                # If the action is outside our reasonable limits, skip this sample.
                # print(f"Skipping timestep {ts} in file {file_path} due to action limits: "
                    #   f"acceleration={model_expert_acceleration}, steering={model_expert_steering}")
                continue
            
            # --- STATE VECTOR CONSTRUCTION (Unchanged) ---
            # (The logic for ego_features, all_agent_features, etc. is identical to the previous script)
            ego_pos = sdc_positions_all_steps[ts]
            ego_heading = sdc_headings_all_steps[ts]
            
            c, s = np.cos(ego_heading), np.sin(ego_heading)
            rot_matrix = np.array([[c, s], [-s, c]])

            acceleration = (current_speed - np.linalg.norm(sdc_velocities_all_steps[ts-1])) / 0.1 if ts > 0 else 0.0
            yaw_rate_state = normalize_angle(ego_heading - sdc_headings_all_steps[ts-1]) / 0.1 if ts > 0 else 0.0
            ego_features = np.array([current_speed, acceleration, yaw_rate_state])

            # --- Agent Features (Now uses pre-computed indices) ---
            all_agent_features_padded = np.zeros((config.num_closest_agents, 10), dtype=np.float32)
            
            # Get the pre-sorted indices of the closest agents for this timestep
            indices_of_closest_agents = closest_agent_indices_all_steps[ts]
            
            num_agents_added = 0
            for agent_idx in indices_of_closest_agents:
                # Stop if we've added enough agents
                if num_agents_added >= config.num_closest_agents:
                    break
                
                # Check if this agent is valid at this timestep
                agent_state = all_agent_trajectories[agent_idx, ts]
                if agent_state[9] > 0: # Check the 'valid' flag
                    obj_type = data['object_types'][agent_idx]
                    relative_pos = np.dot(rot_matrix, agent_state[:2] - ego_pos)
                    relative_vel = np.dot(rot_matrix, agent_state[7:9])
                    relative_heading = normalize_angle(agent_state[6] - ego_heading)
                    type_vec = np.zeros(3); 
                    if obj_type in [1, 2, 3]: type_vec[obj_type - 1] = 1.0
                    
                    features = np.concatenate([relative_pos, relative_vel, [relative_heading, agent_state[3], agent_state[4]], type_vec])
                    all_agent_features_padded[num_agents_added, :] = features
                    num_agents_added += 1
                    
            # --- Map and Rule Features (Unchanged, already fast) ---
            relative_lane_points = np.zeros((config.num_closest_map_points, 2), dtype=np.float32)
            if all_lane_points.size > 0:
                closest_points_global = all_lane_points[closest_lane_indices_all_steps[ts]]
                transformed_points = np.dot(rot_matrix, (closest_points_global - ego_pos).T).T
                relative_lane_points[:transformed_points.shape[0], :] = transformed_points
            
            relative_crosswalk_points = np.zeros((config.num_closest_crosswalk_points, 2), dtype=np.float32)
            if all_crosswalk_points.size > 0:
                closest_cw_points_global = all_crosswalk_points[closest_crosswalk_indices_all_steps[ts]]
                transformed_points = np.dot(rot_matrix, (closest_cw_points_global - ego_pos).T).T
                relative_crosswalk_points[:transformed_points.shape[0], :] = transformed_points
            
            all_map_features = np.concatenate([relative_lane_points.flatten(), relative_crosswalk_points.flatten()])
            dist_to_stop_sign = dist_to_closest_stop_sign_all_steps[ts]
            is_stop_controlled = 0.0
            tl_state_vec = np.array([0, 0, 0, 1.0])
            all_rules_features = np.concatenate([np.array([dist_to_stop_sign, is_stop_controlled]), tl_state_vec])
            
            # --- Assemble the final structured state and action ---
            state_dict = {
                'ego': ego_features.astype(np.float32),
                'agents': all_agent_features_padded, # Already float32
                'lanes': relative_lane_points,       # Already float32
                'crosswalks': relative_crosswalk_points, # Already float32
                'rules': all_rules_features.astype(np.float32)
            }
            action_kinematic = np.array([model_expert_acceleration, model_expert_steering], dtype=np.float32)
            
            scenario_samples.append({
                'state': state_dict,
                'action': torch.from_numpy(action_kinematic),
                'timestep': ts # Add the timestep index
            })
                        
        # --- 4. Save the single output file for this scenario ---
        if scenario_samples: # Only save if there are any valid samples
            scenario_id = os.path.basename(file_path).replace('.npz', '')
            output_filename = os.path.join(output_dir, f"{scenario_id}.pt")
            torch.save(scenario_samples, output_filename, pickle_protocol=4)
            
    except Exception as e:
        # If any error occurs, we log it and skip this file
        print(f"Error processing file {file_path}: {e}")
        print(f"Error processing file {file_path}. Skipping this file.")


def main():
    config = TrainingConfig()
    num_workers = max(1, multiprocessing.cpu_count()//2)
    print(f"Using {num_workers} CPU cores for parallel processing.")
    
    for data_split in ['train', 'val']:
        input_dir = config.train_data_dir if data_split == 'train' else config.val_data_dir
        # Use the correct, distinct output directories
        output_dir = config.bcs_preprocess_train_dir if data_split == 'train' else config.bcs_preprocess_val_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n--- Processing '{data_split}' split (Scenario-per-File) ---")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        
        print(f"Cleaning old files in {output_dir}...")
        for f in glob.glob(os.path.join(output_dir, '*.pt')): os.remove(f)

        files_to_process = glob.glob(os.path.join(input_dir, '*.npz'))
        if not files_to_process: continue
            
        # The list of tasks is now just the list of files
        tasks = [(file_path, config, output_dir) for file_path in files_to_process]
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            # The progress bar now tracks the number of .npz files processed
            list(tqdm(pool.imap_unordered(process_single_scenario_file, tasks), total=len(tasks), desc=f"Processing {data_split} scenarios"))

    print("\n--- Scenario-per-File Preprocessing Complete ---")

if __name__ == "__main__":
    main()