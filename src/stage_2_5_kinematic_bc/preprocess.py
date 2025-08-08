# ======================================================================================
# preprocess_bc_stage_2_5_filtered.py
#
# Description:
#   This version implements a data filtering strategy. It calculates the
#   kinematic action and then discards any state-action pairs where the
#   action is outside of a predefined, physically plausible range. This
#   creates a cleaner, less noisy dataset for training.
#
# Author: Antonio Guillen-Perez
# Date: 2025-08-01
# ======================================================================================

import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import multiprocessing
from src.shared.utils import TrainingConfig, normalize_angle # Make sure utils has normalize_angle

# --- NEW: Define Action Filtering Thresholds ---
MAX_ACCELERATION = 8.0  # m/s^2, very strong acceleration
MIN_ACCELERATION = -10.0 # m/s^2, emergency braking
MAX_STEERING_ANGLE = 0.8 # radians, ~45 degrees, a very sharp turn


def process_file(file_path_tuple):
    """
    Optimized worker function to process all transitions in a single .npz file.
    It calculates and saves a kinematic action (acceleration, steering).
    """
    file_path, config, output_dir = file_path_tuple
    try:
        data = np.load(file_path, allow_pickle=True)
        sdc_index = int(data['sdc_track_index'])
        
        # --- 1. Pre-compute all SDC trajectory data ---
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

        # --- 4. Main Loop ---
        for ts in range(90):
            # --- ACTION CALCULATION (THE KEY CHANGE FOR STAGE 2.5 with InvertibleBicycleModel dynamics) ---
            # 1. Calculate Expert Acceleration from change in speed
            current_speed = np.linalg.norm(sdc_velocities_all_steps[ts])
            next_speed = np.linalg.norm(sdc_velocities_all_steps[ts + 1])
            expert_acceleration = (next_speed - current_speed) / 0.1 # dt = 0.1s

            # 2. Calculate Expert Steering Angle from change in yaw
            current_yaw = sdc_headings_all_steps[ts]
            next_yaw = sdc_headings_all_steps[ts + 1]
            yaw_rate = normalize_angle(next_yaw - current_yaw) / 0.1
            
            # 3. Calculate Expert Steering Angle using kinematic model
            # Assuming a standard car with a wheelbase of 2.8 meters
            WHEELBASE = 2.8 # Wheelbase in meters for a standard car
            # Add a small epsilon to speed to avoid division by zero when static
            current_speed_safe = current_speed if current_speed > 1e-4 else 1e-4
            # tan(steering_angle) = (yaw_rate * wheelbase) / speed
            expert_steering = np.arctan2((yaw_rate * WHEELBASE), current_speed_safe)
            
            action_kinematic = np.array([expert_acceleration, expert_steering], dtype=np.float32)

            # --- THIS IS THE NEW FILTERING LOGIC ---
            if not (MIN_ACCELERATION <= expert_acceleration <= MAX_ACCELERATION and
                    -MAX_STEERING_ANGLE <= expert_steering <= MAX_STEERING_ANGLE):
                # If the action is outside our reasonable limits, skip this sample.
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
                    
            all_agent_features = all_agent_features_padded.flatten()

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
            
            # --- Final Assembly and Saving ---
            state_vector = np.concatenate([ego_features, all_agent_features, all_map_features, all_rules_features])
            
            # If the shape of state_vector is not (279,), raise an error
            if state_vector.shape != (279,):
                raise ValueError(f"Expected state vector shape (279,), got {state_vector.shape} in file {file_path} at timestep {ts}")
            
            # --- SAVE THE FINAL TENSORS ---
            output_data = {
                'state': torch.tensor(state_vector, dtype=torch.float32),
                'action': torch.tensor(action_kinematic, dtype=torch.float32)
            }
            scenario_id = os.path.basename(file_path).replace('.npz', '')
            output_filename = os.path.join(output_dir, f"{scenario_id}_ts{ts}.pt")
            torch.save(output_data, output_filename, pickle_protocol=4) # Use protocol 4 for compatibility

    except Exception as e:
        # For debugging, it's useful to see which files fail
        print(f"WARNING: Skipping file {file_path} due to error: {e}")
        raise e

def main():
    config = TrainingConfig()
    
    for data_split in ['train', 'val']:
        input_dir = config.train_data_dir if data_split == 'train' else config.val_data_dir
        output_dir = config.bck_preprocess_train_dir if data_split == 'train' else config.bck_preprocess_val_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n--- Preprocessing for Stage 2.5 ('{data_split}' split) ---")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        files_to_process = glob.glob(os.path.join(input_dir, '*.npz'))
        tasks = [(fp, config, output_dir) for fp in files_to_process]

        with multiprocessing.Pool(processes=config.num_workers) as pool:
            list(tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc=f"Preprocessing {data_split} files"))

    print("\n--- Kinematic Preprocessing Complete ---")

if __name__ == "__main__":
    main()