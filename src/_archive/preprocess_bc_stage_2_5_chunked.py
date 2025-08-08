# ======================================================================================
# preprocess_bc_stage_2_5_chunked.py
#
# Description:
#   This version saves the preprocessed data into large "chunk" files instead of
#   millions of small files. Each chunk contains many state-action pairs. This
#   dramatically improves I/O performance during training.
#
# Author: Antonio Guillen-Perez
# ======================================================================================
import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import multiprocessing
from utils import TrainingConfig, normalize_angle

# NEW: Define how many samples go into each chunk file
SAMPLES_PER_CHUNK = 20000

def process_file_and_collect(file_path_tuple):
    """
    Worker function. Processes a single .npz file and returns a list
    of all valid (state, action) pairs found within it.
    """
    file_path, config = file_path_tuple
    collected_samples = []
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
            
            WHEELBASE = 2.8 # Wheelbase in meters for a standard car
            # Add a small epsilon to speed to avoid division by zero when static
            current_speed_safe = current_speed if current_speed > 1e-4 else 1e-4
            # tan(steering_angle) = (yaw_rate * wheelbase) / speed
            expert_steering = np.arctan2((yaw_rate * WHEELBASE), current_speed_safe)
            
            action_kinematic = np.array([expert_acceleration, expert_steering], dtype=np.float32)

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
            
            # Instead of saving, append to a list
            state_tensor = torch.tensor(state_vector, dtype=torch.float32)
            action_tensor = torch.tensor(action_kinematic, dtype=torch.float32)
            collected_samples.append({'state': state_tensor, 'action': action_tensor})

    except Exception as e:
        # For debugging, it's useful to see which files fail
        print(f"WARNING: Skipping file {file_path} due to error: {e}")
        raise e

        
    return collected_samples

def main():
    config = TrainingConfig()
    num_workers = multiprocessing.cpu_count() - 2
    
    for data_split in ['train', 'val']:
        input_dir = config.train_data_dir if data_split == 'train' else config.val_data_dir
        # --- NEW OUTPUT DIRECTORIES FOR CHUNKS ---
        output_dir = input_dir.replace('processed', 'bc_kinematic_chunked')
        os.makedirs(output_dir, exist_ok=True)
        # Clean the directory before starting
        for f in glob.glob(os.path.join(output_dir, '*.pt')): os.remove(f)

        print(f"\n--- Preprocessing '{data_split}' split into chunks ---")
        
        files_to_process = glob.glob(os.path.join(input_dir, '*.npz'))
        tasks = [(fp, config) for fp in files_to_process]

        all_samples = []
        with multiprocessing.Pool(processes=num_workers) as pool:
            # The pool returns a list of lists of samples
            results = list(tqdm(pool.imap_unordered(process_file_and_collect, tasks), total=len(tasks), desc=f"Processing .npz files"))
        
        # Flatten the list of lists into a single list of all samples
        print("Flattening results...")
        for res in results:
            all_samples.extend(res)
            
        print(f"Total samples collected: {len(all_samples)}. Now saving into chunks...")

        # Save the collected samples into chunked files
        chunk_num = 0
        for i in tqdm(range(0, len(all_samples), SAMPLES_PER_CHUNK), desc="Saving chunks"):
            chunk = all_samples[i:i + SAMPLES_PER_CHUNK]
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_num}.pt")
            torch.save(chunk, chunk_path)
            chunk_num += 1

    print("\n--- Chunked Preprocessing Complete ---")

if __name__ == "__main__":
    main()