# ======================================================================================
# preprocess_for_bc.py (Fully Optimized Version 3.0)
#
# Description:
#   This definitive version vectorizes BOTH map feature and agent feature searches.
#   All expensive nearest-neighbor calculations are performed once per file using
#   highly optimized NumPy operations, eliminating the last major performance bottleneck.
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

from utils import TrainingConfig # Assumes train_bc.py is in the same folder
from utils import normalize_angle


def process_file(file_path_tuple):
    """
    Fully optimized worker function. All nearest-neighbor searches are vectorized.
    """
    file_path, config, output_dir = file_path_tuple
    try:
        data = np.load(file_path, allow_pickle=True)
        sdc_index = int(data['sdc_track_index'])
        
        # --- 1. Pre-compute SDC trajectory ---
        sdc_trajectory = data['all_trajectories'][sdc_index]
        sdc_positions_all_steps = sdc_trajectory[:, :2]
        sdc_headings_all_steps = sdc_trajectory[:, 6]

        # --- 2. Vectorized Map Feature Search (already fast) ---
        map_data = data['map_data'].item()
        # ... (Your correct and fast map logic remains here) ...
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

        # --- 3. NEW: Vectorized Agent Feature Search ---
        all_agent_trajectories = data['all_trajectories']
        num_agents = all_agent_trajectories.shape[0]
        
        # Get all agent positions at all timesteps: shape (num_agents, 91, 2)
        agent_positions_all_steps = all_agent_trajectories[:, :, :2]
        
        # Transpose to (91, num_agents, 2) to align with SDC positions for broadcasting
        agent_positions_all_steps_t = np.transpose(agent_positions_all_steps, (1, 0, 2))

        # Calculate distance matrix: SDC at each step vs all other agents at that step
        # Shape: (91, num_agents)
        dist_matrix_agents = np.linalg.norm(sdc_positions_all_steps[:, np.newaxis, :] - agent_positions_all_steps_t, axis=2)
        
        # Set distance to self (SDC) to infinity so it's never chosen as a neighbor
        dist_matrix_agents[:, sdc_index] = np.inf

        # Sort once to get the closest agent indices for every timestep
        # Shape: (91, num_agents)
        closest_agent_indices_all_steps = np.argsort(dist_matrix_agents, axis=1)

        # --- 4. Main Loop (Now fully optimized) ---
        for ts in range(90):
            # ... (Action and Ego Feature logic is unchanged and fast) ...
            ego_pos = sdc_positions_all_steps[ts]
            ego_heading = sdc_headings_all_steps[ts]
            c, s = np.cos(ego_heading), np.sin(ego_heading)
            rot_matrix = np.array([[c, s], [-s, c]])
            action_global = sdc_positions_all_steps[ts + 1] - ego_pos
            action_ego = np.dot(rot_matrix, action_global)
            current_speed = np.linalg.norm(sdc_trajectory[ts, 7:9])
            if ts > 0:
                prev_speed = np.linalg.norm(sdc_trajectory[ts - 1, 7:9])
                acceleration = (current_speed - prev_speed) / 0.1
                yaw_rate = normalize_angle(ego_heading - sdc_headings_all_steps[ts - 1]) / 0.1
            else:
                acceleration, yaw_rate = 0.0, 0.0
            ego_features = np.array([current_speed, acceleration, yaw_rate])

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
            
            # --- C. Save the final tensors ---
            output_data = {'state': torch.tensor(state_vector, dtype=torch.float32), 'action': torch.tensor(action_ego, dtype=torch.float32)}
            scenario_id = os.path.basename(file_path).replace('.npz', '')
            output_filename = os.path.join(output_dir, f"{scenario_id}_ts{ts}.pt")
            torch.save(output_data, output_filename, pickle_protocol=4) # Use protocol 4 for compatibility

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