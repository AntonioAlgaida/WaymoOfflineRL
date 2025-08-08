# ======================================================================================
# preprocess.py (Stage 3 - RL Dataset Creator)
#
# Description:
#   This script creates the dataset for Offline RL. It saves data in a
#   "scenario-per-file" format, where each .pt file contains the complete,
#   ordered sequence of timesteps for a single scenario. This is crucial for
#   building (s, a, s') transitions.
#   NOTE: This script does NOT filter actions, as that would break the sequence.
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

MAX_ACCELERATION = 8.0  # m/s^2, very strong acceleration
MIN_ACCELERATION = -10.0 # m/s^2, emergency braking
MAX_STEERING_ANGLE = 0.8 # radians, ~45 degrees, a very sharp turn


def find_current_lane_id(ego_pos, ego_heading, map_data):
    """
    Finds the ID of the lane the ego vehicle is most likely in.
    """
    if not map_data['lane_polylines']:
        return None

    best_lane_id = None
    min_dist = float('inf')

    for lane_id, polyline in map_data['lane_polylines'].items():
        # Find the closest point on this lane's polyline to the ego vehicle
        distances = np.linalg.norm(polyline[:, :2] - ego_pos, axis=1)
        closest_point_idx = np.argmin(distances)
        dist = distances[closest_point_idx]

        # Check if this is the closest lane found so far
        if dist < min_dist:
            # --- Check for heading alignment ---
            # Get the direction of the lane at the closest point
            if closest_point_idx < len(polyline) - 1:
                lane_dir = polyline[closest_point_idx + 1, :2] - polyline[closest_point_idx, :2]
            else: # At the end of the polyline
                lane_dir = polyline[closest_point_idx, :2] - polyline[closest_point_idx - 1, :2]
            
            lane_heading = np.arctan2(lane_dir[1], lane_dir[0])
            ego_dir_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])
            
            # Check if the dot product is positive (i.e., angle is < 90 degrees)
            # This ensures we don't snap to a lane going the opposite direction
            if np.dot(lane_dir, ego_dir_vec) > 0:
                min_dist = dist
                best_lane_id = lane_id

    # Only return a lane ID if we are reasonably close to it (e.g., within 5 meters)
    if min_dist < 5.0:
        return best_lane_id
    
    return None

def process_single_scenario_file(task_tuple):
    """
    Worker function. Processes one .npz file and saves one corresponding .pt
    file containing all 90 structured samples for that scenario.
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
        
        # --- NEW: Create a quick lookup map for traffic light data ---
        # Shape of tl_states_raw: (91, num_tl_lanes, 4)
        tl_states_raw = data['traffic_light_states']
        # Create a dictionary: {timestep -> {lane_id -> state_enum}}
        tl_lookup = {}
        for ts in range(tl_states_raw.shape[0]):
            ts_map = {}
            for i in range(tl_states_raw.shape[1]):
                lane_id = int(tl_states_raw[ts, i, 0])
                state_enum = int(tl_states_raw[ts, i, 1])
                if lane_id > 0: # Lane ID 0 is invalid
                    ts_map[lane_id] = state_enum
            tl_lookup[ts] = ts_map
        
        # NEW: Stop sign lookup set for all controlled lanes
        stop_controlled_lanes = set()
        if map_data['stopsign_positions']:
            for stop_sign_data in map_data['stopsign_positions'].values():
                stop_controlled_lanes.update(stop_sign_data['controls_lanes'])
        
        # This list will hold the 0-90 valid samples for THIS scenario
        scenario_samples = []
        # --- 4. Main Loop ---
        for ts in range(90):
            # --- ACTION CALCULATION (THE KEY CHANGE FOR STAGE 2.5 with InvertibleBicycleModel dynamics) ---
            all_agents_actions = jit_compute_inverse(traj=full_trajectory, timestep=ts, dt=0.1)
            sdc_action = all_agents_actions.data[sdc_index] # Shape (2,) -> [accel, steer]
            # Extract the expert action from the model's output
            model_expert_acceleration, model_expert_steering = sdc_action[0], sdc_action[1]

            # Instead of filtering/skipping, we clip the actions to a plausible range.
            # This preserves the sequence while cleaning the data.
            model_expert_acceleration = np.clip(model_expert_acceleration, MIN_ACCELERATION, MAX_ACCELERATION)
            model_expert_steering = np.clip(model_expert_steering, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
            
            # 1. Calculate Expert Acceleration from change in speed
            current_speed = np.linalg.norm(sdc_velocities_all_steps[ts])
            
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
                
                # ADD THIS EXPLICIT CHECK
                if agent_idx == sdc_index:
                    continue # Never process the SDC, even if we reach it
    
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
            
            # all_map_features = np.concatenate([relative_lane_points.flatten(), relative_crosswalk_points.flatten()])
            dist_to_stop_sign = dist_to_closest_stop_sign_all_steps[ts]
            
            # --- NEW: Goal and Route Feature Calculation ---
    
            # 1. Goal Features
            final_destination = sdc_positions_all_steps[-1] # The last point in the 9-sec trajectory
            distance_to_goal = np.linalg.norm(final_destination - ego_pos)
            
            # Direction to goal in ego-centric frame
            dir_to_goal_global = final_destination - ego_pos
            dir_to_goal_ego = np.dot(rot_matrix, dir_to_goal_global)
            # Normalize the direction vector
            dir_norm = np.linalg.norm(dir_to_goal_ego)
            if dir_norm > 1e-4:
                dir_to_goal_ego /= dir_norm
            
            # 2. Route Features (Future Waypoints)
            # Sample 10 future waypoints at 0.5s intervals (5 timesteps)
            future_indices = np.arange(ts + 5, ts + 51, 5)
            # Ensure indices are within the 91-step boundary
            future_indices = np.clip(future_indices, 0, 90)
            
            future_waypoints_global = sdc_positions_all_steps[future_indices]
            
            # Convert all 10 waypoints to the ego-centric frame
            relative_future_waypoints = np.dot(rot_matrix, (future_waypoints_global - ego_pos).T).T
            
            # 3. Traffic Light States
                # 1. Find the SDC's current lane
            current_lane_id = find_current_lane_id(ego_pos, ego_heading, map_data)
            
                # 2. Look up the traffic light state for this lane at this timestep
            tl_state = 0 # Default to LANE_STATE_UNKNOWN
            if current_lane_id and current_lane_id in tl_lookup[ts]:
                tl_state = tl_lookup[ts][current_lane_id]
                
                # 3. Convert the state enum to a one-hot vector
            # From Waymo Proto: UNKNOWN=0, STOP=4, CAUTION=5, GO=6 (simplified)
            tl_state_vec = np.zeros(4, dtype=np.float32) # [G, Y, R, U]
            if tl_state in [3, 6]: # LANE_STATE_ARROW_GO or LANE_STATE_GO
                tl_state_vec[0] = 1.0
            elif tl_state in [2, 5, 8]: # Any CAUTION state
                tl_state_vec[1] = 1.0
            elif tl_state in [1, 4, 7]: # Any STOP state
                tl_state_vec[2] = 1.0
            else: # UNKNOWN
                tl_state_vec[3] = 1.0
            
            # Stop sign state
            dist_to_stop_sign = dist_to_closest_stop_sign_all_steps[ts]
            # NEW: Real is_stop_controlled logic
            is_stop_controlled = 1.0 if current_lane_id in stop_controlled_lanes else 0.0

            # --- Assemble the final structured state and action ---
            state_dict = {
                'ego': ego_features.astype(np.float32),  # Shape (3,)
                'agents': all_agent_features_padded, # Already float32, shape (num_agents, 10)
                'lanes': relative_lane_points,       # Already float32, shape (num_closest_map_points, 2)
                'crosswalks': relative_crosswalk_points, # Already float32, shape (num_closest_crosswalk_points, 2)
                'route': relative_future_waypoints.astype(np.float32), # Shape (10, 2)

                'rules': np.concatenate([
                    np.array([distance_to_goal, dir_to_goal_ego[0], dir_to_goal_ego[1], dist_to_stop_sign, is_stop_controlled]),
                    tl_state_vec
                ]).astype(np.float32) # Shape (9,)
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
        output_dir = config.cql_preprocess_dir if data_split == 'train' else config.cql_preprocess_val_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n--- Processing '{data_split}' split (Scenario-per-File) ---")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        
        
        if data_split == 'train':
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