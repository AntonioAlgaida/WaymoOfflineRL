# ======================================================================================
# preprocess_for_bc.py (Optimized Version)
#
# Description:
#   This version is heavily optimized to reduce redundant calculations. It processes
#   each .npz file as a whole, using vectorized NumPy operations to find nearest
#   neighbors for all timesteps at once, dramatically speeding up preprocessing.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import multiprocessing
import time

# Use the same config, but we only need the data paths and feature engineering params
from train_bc import TrainingConfig

def _normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def process_file(file_path_tuple):
    """
    Optimized worker function to process all transitions in a single .npz file.
    It uses vectorized NumPy operations to find nearest neighbors for all
    timesteps at once, dramatically speeding up preprocessing.
    """
    file_path, config, output_dir = file_path_tuple
    try:
        data = np.load(file_path, allow_pickle=True)
        sdc_index = int(data['sdc_track_index'])
        
        # --- 1. Pre-compute all SDC positions and transformations ---
        sdc_trajectory = data['all_trajectories'][sdc_index]  # Shape: (91, 10)
        sdc_positions_all_steps = sdc_trajectory[:, :2] # All 91 (x,y) positions, shape: (91, 2)
        sdc_headings_all_steps = sdc_trajectory[:, 6]   # All 91 headings, shape: (91,)

        # --- 2. Vectorized Nearest Neighbor Search for ALL Map Features ---
        map_data = data['map_data'].item()
        
        # --- Lanes ---
        lane_polylines = list(map_data['lane_polylines'].values())
        if lane_polylines:
            all_lane_points = np.vstack(lane_polylines)[:, :2]  # All lane points, shape: (num_lane_points, 2) Only first two columns (x, y)
            
            # This computes a (91, num_lane_points) distance matrix in one shot.
            dist_matrix_lanes = np.linalg.norm(sdc_positions_all_steps[:, np.newaxis, :] - all_lane_points[np.newaxis, :, :], axis=2)  # Shape: (91, num_lane_points)
            
            # This sorts each row (timestep) and gets the indices of the closest points.
            closest_lane_indices_all_steps = np.argsort(dist_matrix_lanes, axis=1)[:, :config.num_closest_map_points]  # Shape: (91, num_closest_map_points)
        else:
            all_lane_points = np.array([]) # Empty array for later checks
            closest_lane_indices_all_steps = np.zeros((91, config.num_closest_map_points), dtype=int)

        # --- Crosswalks ---
        crosswalk_polygons = list(map_data['crosswalk_polygons'].values())
        if crosswalk_polygons:
            all_crosswalk_points = np.vstack(crosswalk_polygons)[:, :2]
            dist_matrix_cw = np.linalg.norm(sdc_positions_all_steps[:, np.newaxis, :] - all_crosswalk_points[np.newaxis, :, :], axis=2)
            closest_crosswalk_indices_all_steps = np.argsort(dist_matrix_cw, axis=1)[:, :config.num_closest_crosswalk_points]
        else:
            all_crosswalk_points = np.array([])
            closest_crosswalk_indices_all_steps = np.zeros((91, config.num_closest_crosswalk_points), dtype=int)
            
        # --- Stop Signs ---
        if map_data['stopsign_positions']:
            stop_sign_positions = np.array([d['position'][:2] for d in map_data['stopsign_positions'].values()])
            dist_matrix_ss = np.linalg.norm(sdc_positions_all_steps[:, np.newaxis, :] - stop_sign_positions[np.newaxis, :, :], axis=2)
            # For each timestep, find the minimum distance to any stop sign.
            dist_to_closest_stop_sign_all_steps = np.min(dist_matrix_ss, axis=1)
        else:
            dist_to_closest_stop_sign_all_steps = np.full((91,), config.max_dist, dtype=np.float32)


        # --- 3. Main Loop (Now extremely fast) ---
        for ts in range(90):
            # Establish current frame of reference
            ego_pos = sdc_positions_all_steps[ts]
            ego_heading = sdc_headings_all_steps[ts]
            c, s = np.cos(ego_heading), np.sin(ego_heading)
            rot_matrix = np.array([[c, s], [-s, c]])

            # Action
            action_global = sdc_positions_all_steps[ts + 1] - ego_pos
            action_ego = np.dot(rot_matrix, action_global)

            # 1. Ego Features
            current_speed = np.linalg.norm(sdc_trajectory[ts, 7:9])
            if ts > 0:
                prev_speed = np.linalg.norm(sdc_trajectory[ts - 1, 7:9])
                acceleration = (current_speed - prev_speed) / 0.1
                yaw_rate = _normalize_angle(ego_heading - sdc_headings_all_steps[ts - 1]) / 0.1
            else:
                acceleration, yaw_rate = 0.0, 0.0
            ego_features = np.array([current_speed, acceleration, yaw_rate])

            # 2. Agent Features (This part is already reasonably fast per-step)
            all_agent_trajectories = data['all_trajectories']
            valid_agents_states = []
            
            # Find all valid agents at the current timestep
            for i in range(all_agent_trajectories.shape[0]):
                if i != sdc_index and all_agent_trajectories[i, ts, 9] > 0:
                    agent_state = all_agent_trajectories[i, ts]
                    dist = np.linalg.norm(agent_state[:2] - ego_pos)
                    valid_agents_states.append((dist, agent_state, data['object_types'][i]))
            
            # Sort them by distance to find the closest ones
            valid_agents_states.sort(key=lambda item: item[0])

            # Pre-allocate a fixed-size array for the features. This is the key.
            # It guarantees the final shape will be correct.
            all_agent_features_padded = np.zeros((config.num_closest_agents, 10), dtype=np.float32)

            # Determine how many of the closest agents we will actually use
            num_agents_to_add = min(len(valid_agents_states), config.num_closest_agents)

            # Loop through only the closest agents and fill the pre-allocated array
            for i in range(num_agents_to_add):
                _, agent_state, obj_type = valid_agents_states[i]
                
                # Perform transformations
                relative_pos = np.dot(rot_matrix, agent_state[:2] - ego_pos)
                relative_vel = np.dot(rot_matrix, agent_state[7:9])
                relative_heading = _normalize_angle(agent_state[6] - ego_heading)
                
                # Create one-hot vector for type
                type_vec = np.zeros(3, dtype=np.float32)
                if obj_type in [1, 2, 3]: # Vehicle, Pedestrian, Cyclist
                    type_vec[obj_type - 1] = 1.0
                
                # Construct the feature vector for this single agent
                features = np.concatenate([
                    relative_pos, 
                    relative_vel, 
                    [relative_heading, agent_state[3], agent_state[4]], # heading, length, width
                    type_vec
                ])
                
                # Place the 10 features for this agent directly into the correct row of our padded array
                all_agent_features_padded[i, :] = features

            # Flatten the final, correctly-sized array. This will ALWAYS have shape (150,)
            all_agent_features = all_agent_features_padded.flatten()

            # 3. Map Features (using pre-computed indices)
            # --- Lanes ---
            # Pre-allocate a fixed-size array for the final features.
            relative_lane_points = np.zeros((config.num_closest_map_points, 2), dtype=np.float32)

            if all_lane_points.size > 0:
                # Get the global coordinates of the closest points for this timestep
                closest_points_global = all_lane_points[closest_lane_indices_all_steps[ts]]
                
                # Determine how many points we actually found (can be less than the max if the map is small)
                num_points_found = closest_points_global.shape[0]
                
                # Transform them to the ego-centric frame
                transformed_points = np.dot(rot_matrix, (closest_points_global - ego_pos).T).T
                
                # Fill the pre-allocated array with the transformed points
                relative_lane_points[:num_points_found, :] = transformed_points
                
            # --- Crosswalks ---
            # Pre-allocate another fixed-size array.
            relative_crosswalk_points = np.zeros((config.num_closest_crosswalk_points, 2), dtype=np.float32)

            if all_crosswalk_points.size > 0:
                # Get the global coordinates of the closest points for this timestep
                closest_cw_points_global = all_crosswalk_points[closest_crosswalk_indices_all_steps[ts]]
                
                # Determine how many points we actually found
                num_points_found = closest_cw_points_global.shape[0]
                
                # Transform them
                transformed_points = np.dot(rot_matrix, (closest_cw_points_global - ego_pos).T).T
                
                # Fill the pre-allocated array
                relative_crosswalk_points[:num_points_found, :] = transformed_points
                
            # Now, flattening these arrays will ALWAYS result in the correct size.
            all_map_features = np.concatenate([
                relative_lane_points.flatten(), 
                relative_crosswalk_points.flatten()
            ])

            # 4. Rule-Based Features (using pre-computed distances)
            dist_to_stop_sign = dist_to_closest_stop_sign_all_steps[ts]
            is_stop_controlled = 0.0 # Placeholder for more complex logic
            tl_state_vec = np.array([0, 0, 0, 1.0]) # Placeholder

            all_rules_features = np.concatenate([np.array([dist_to_stop_sign, is_stop_controlled]), tl_state_vec])

            # Final state vector concatenation
            state_vector = np.concatenate([ego_features, all_agent_features, all_map_features, all_rules_features])
            
            # If the shape of state_vector is not (279,), raise an error
            if state_vector.shape != (279,):
                raise ValueError(f"Expected state vector shape (279,), got {state_vector.shape} in file {file_path} at timestep {ts}")
            
            # --- C. Save the final tensors ---
            output_data = {'state': torch.tensor(state_vector, dtype=torch.float32), 'action': torch.tensor(action_ego, dtype=torch.float32)}
            scenario_id = os.path.basename(file_path).replace('.npz', '')
            output_filename = os.path.join(output_dir, f"{scenario_id}_ts{ts}.pt")
            torch.save(output_data, output_filename)

    except Exception as e:
        # Silently skip corrupted files or files that cause errors during processing
        # For debugging, you might want to print the error:
        print(f"CRITICAL ERROR processing file {file_path}: {e}")
        pass


# ... (The main function remains the same as the previous version) ...
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
        tasks = [(fp, config, output_dir) for fp in files_to_process]

        with multiprocessing.Pool(processes=config.num_workers) as pool:
            list(tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc=f"Preprocessing {data_split} files"))

    print("\n--- Preprocessing Complete ---")

if __name__ == "__main__":
    main()