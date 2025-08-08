# ======================================================================================
# preprocess_for_bc.py
#
# Description:
#   Performs a one-time, offline preprocessing of the parsed .npz data to generate
#   final training samples (state-action pairs) for the Behavioral Cloning agent.
#   This script does all the heavy feature engineering *before* training,
#   which dramatically speeds up the training loop.
#
# Author: Antonio Guillen-Perez
# Date: 2025-07-29
# ======================================================================================

import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import multiprocessing

# Use the same config, but we only need the data paths and feature engineering params
from train_bc import TrainingConfig # Assumes train_bc.py is in the same folder


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
    
def process_file(file_path_tuple):
    """
    Worker function to process all transitions in a single .npz file.
    """
    file_path, config, output_dir = file_path_tuple
    data = np.load(file_path, allow_pickle=True)
    
    # Re-use the feature engineering logic, slightly adapted
    for time_step_index in range(90):
        try:
            # --- A. Establish Frame of Reference & Get Action ---
            sdc_index = int(data['sdc_track_index'])
            ego_state = data['all_trajectories'][sdc_index, time_step_index]
            ego_x, ego_y, ego_heading = ego_state[0], ego_state[1], ego_state[6]
            
            c, s = np.cos(ego_heading), np.sin(ego_heading)
            rot_matrix = np.array([[c, s], [-s, c]])
            ego_pos = np.array([ego_x, ego_y])

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
                yaw_rate = normalize_angle(ego_heading - prev_ego_state[6]) / 0.1
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
                print(f"Error sorting agents for index in file {file_path}: {e}")
                print(f"Valid agents states: {valid_agents_states}")
                raise ValueError(f"Failed to sort agents for index in file {file_path}")
            
            for _, agent_state, obj_type in valid_agents_states[:config.num_closest_agents]:
                relative_pos = np.dot(rot_matrix, agent_state[:2] - ego_pos)
                relative_vel = np.dot(rot_matrix, agent_state[7:9])
                relative_heading = normalize_angle(agent_state[6] - ego_heading)
                
                type_vec = np.zeros(3) # Vehicle, Pedestrian, Cyclist
                if obj_type in [1, 2, 3]: type_vec[obj_type - 1] = 1.0

                agent_features = np.concatenate([relative_pos, relative_vel, [relative_heading, agent_state[3], agent_state[4]], type_vec])
                agent_features_list.append(agent_features)
                
            num_found = len(agent_features_list)
            if num_found < config.num_closest_agents:
                padding = np.zeros((config.num_closest_agents - num_found, 10))
                agent_features_list.extend(padding)
            
            try:
                all_agent_features = np.concatenate(agent_features_list).flatten()
            except ValueError as e:
                print(f"Error concatenating agent features: {e}")
                # Raise an error
                raise ValueError(f"Failed to concatenate agent features for index in file {file_path}")
            
            # 3. Roadgraph Features (50 points * 2 features = 100 features)
            map_data = data['map_data'].item()
            lane_points = list(map_data['lane_polylines'].values())
            if lane_points:
                all_lane_points = np.vstack(lane_points)[:, :2]
                distances = np.linalg.norm(all_lane_points - ego_pos, axis=1)
                closest_indices = np.argsort(distances)[:config.num_closest_map_points]
                closest_points = all_lane_points[closest_indices]
                
                relative_map_points = np.dot(rot_matrix, (closest_points - ego_pos).T).T
                
                num_found_points = len(relative_map_points)
                if num_found_points < config.num_closest_map_points:
                    padding = np.zeros((config.num_closest_map_points - num_found_points, 2))
                    relative_map_points = np.vstack([relative_map_points, padding])
            else:
                relative_map_points = np.zeros((config.num_closest_map_points, 2))
            all_map_features = relative_map_points.flatten()

            # 4 & 5. Rule-Based Features (7 features)
            # (Simplified implementation for now, full graph traversal can be complex)
            dist_to_stop_sign, is_stop_controlled = config.max_dist, 0.0
            dist_to_crosswalk = config.max_dist
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
            
            # --- C. Save the final tensors ---
            output_data = {
                'state': torch.tensor(state_vector, dtype=torch.float32),
                'action': torch.tensor(action_ego, dtype=torch.float32)
            }
            
            scenario_id = os.path.basename(file_path).replace('.npz', '')
            output_filename = os.path.join(output_dir, f"{scenario_id}_ts{time_step_index}.pt")
            torch.save(output_data, output_filename)
        
        except Exception as e:
            # Skip this transition if something goes wrong (e.g., missing data)
            print(f"Skipping ts {time_step_index} in {file_path} due to error: {e}")
            continue


def main():
    config = TrainingConfig()
    
    for data_split in ['train', 'val']:
        input_dir = config.train_data_dir if data_split == 'train' else config.val_data_dir
        output_dir = input_dir.replace('processed', 'bc_preprocessed')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n--- Preprocessing for '{data_split}' split ---")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        files_to_process = glob.glob(os.path.join(input_dir, '*.npz'))
        
        # Prepare arguments for multiprocessing
        tasks = [(fp, config, output_dir) for fp in files_to_process]

        # Use multiprocessing to speed up the preprocessing
        with multiprocessing.Pool(processes=config.num_workers) as pool:
            list(tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc=f"Preprocessing {data_split} files"))

    print("\n--- Preprocessing Complete ---")

if __name__ == "__main__":
    main()