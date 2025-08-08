# ======================================================================================
# evaluate.py (Stage 2.8 - Structured MLP BC)
#
# Description:
#   Loads the trained Structured MLP Behavioral Cloning (BC-S v2) policy and
#   evaluates it in a closed-loop simulation using the Waymax environment.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import torch
import numpy as np
import jax
from jax import numpy as jnp
import os
from tqdm import tqdm
import argparse
import mediapy

# Import our project-specific modules
from src.shared.utils import TrainingConfig, construct_state_from_npz, normalize_angle
from src.stage_2_8_structured_mlp_bc.networks import StructuredMLP_BC_Model

# Import Waymax components
from waymax import datatypes
from waymax import visualization
from waymax import env as _env
from waymax import dynamics
from waymax import config as _config

# In src/stage_2_8_structured_mlp_bc/evaluate.py

# --- NEW HELPER FOR LIVE EVALUATION ---
def find_current_lane_id_live(ego_pos, ego_heading, roadgraph_points):
    """
    Finds the ID of the lane the ego vehicle is most likely in from live
    Waymax RoadgraphPoints.
    """
    WAYMAX_LANE_TYPE = 1 # Corresponds to LANE_FREEWAY, a common proxy
    
    # Filter for lane points only
    lane_mask = (roadgraph_points.types == WAYMAX_LANE_TYPE)
    if not lane_mask.any():
        return None

    lane_points_xyz = jnp.stack([roadgraph_points.x, roadgraph_points.y, roadgraph_points.z], axis=-1)
    lane_ids = roadgraph_points.ids
    
    # Get only the valid lane points and their IDs
    valid_lane_points = lane_points_xyz[lane_mask]
    valid_lane_ids = lane_ids[lane_mask]
    
    # This is a simplified nearest-neighbor search. For evaluation, this is fast enough.
    # A more optimized version would use a k-d tree.
    distances = jnp.linalg.norm(valid_lane_points[:, :2] - ego_pos, axis=1)
    closest_point_overall_idx = jnp.argmin(distances)
    
    # We need to find points belonging to the same lane as the closest point
    closest_lane_id = valid_lane_ids[closest_point_overall_idx]
    points_on_same_lane_mask = (valid_lane_ids == closest_lane_id)
    points_on_same_lane = valid_lane_points[points_on_same_lane_mask]
    
    # Re-calculate closest point index *within this specific lane*
    distances_on_lane = jnp.linalg.norm(points_on_same_lane[:, :2] - ego_pos, axis=1)
    closest_point_on_lane_idx = jnp.argmin(distances_on_lane)
    
    # Now we can safely get the lane direction
    if closest_point_on_lane_idx < len(points_on_same_lane) - 1:
        lane_dir = points_on_same_lane[closest_point_on_lane_idx + 1, :2] - points_on_same_lane[closest_point_on_lane_idx, :2]
    else:
        lane_dir = points_on_same_lane[closest_point_on_lane_idx, :2] - points_on_same_lane[closest_point_on_lane_idx - 1, :2]

    ego_dir_vec = jnp.array([jnp.cos(ego_heading), jnp.sin(ego_heading)])
    
    # Check for heading alignment
    if jnp.dot(lane_dir, ego_dir_vec) > 0 and jnp.min(distances) < 5.0:
        return closest_lane_id.item()

    return None

def state_to_feature_dict(state: datatypes.SimulatorState, config: TrainingConfig) -> dict:
    """
    Converts a live Waymax SimulatorState object into the structured dictionary
    of features that our model was trained on. This is a direct translation
    of the logic in the final preprocessing script.
    """
    ts = state.timestep.item()
    sdc_index = jnp.argmax(state.object_metadata.is_sdc).item()
    
    # --- 1. Ego Features ---
    ego_pos = jnp.array([state.sim_trajectory.x[sdc_index, ts], state.sim_trajectory.y[sdc_index, ts]])
    ego_heading = state.sim_trajectory.yaw[sdc_index, ts].item()
    current_vel_xy = jnp.array([state.sim_trajectory.vel_x[sdc_index, ts], state.sim_trajectory.vel_y[sdc_index, ts]])
    current_speed = jnp.linalg.norm(current_vel_xy).item()
    if ts > 0:
        prev_vel_xy = jnp.array([state.sim_trajectory.vel_x[sdc_index, ts-1], state.sim_trajectory.vel_y[sdc_index, ts-1]])
        prev_speed = jnp.linalg.norm(prev_vel_xy).item()
        prev_heading = state.sim_trajectory.yaw[sdc_index, ts - 1].item()
        acceleration = (current_speed - prev_speed) / 0.1
        yaw_rate = normalize_angle(ego_heading - prev_heading) / 0.1
    else:
        acceleration, yaw_rate = 0.0, 0.0
    ego_features = np.array([current_speed, acceleration, yaw_rate], dtype=np.float32)

    # --- 2. Agent Features ---
    # rot_matrix = np.array([[np.cos(ego_heading), np.sin(ego_heading)], [-np.sin(ego_heading), np.cos(ego_heading)]])
    # all_agents_x = state.sim_trajectory.x[:, ts]
    # all_agents_y = state.sim_trajectory.y[:, ts]
    # all_agents_valid = state.sim_trajectory.valid[:, ts]
    # valid_agents_info = []
    # for i in range(state.num_objects):
    #     if all_agents_valid[i] and i != sdc_index:
    #         agent_pos = jnp.array([all_agents_x[i], all_agents_y[i]])
    #         dist = jnp.linalg.norm(agent_pos - ego_pos).item()
    #         valid_agents_info.append((dist, i))
    # valid_agents_info.sort(key=lambda item: item[0])
    # agent_features_padded = np.zeros((config.num_closest_agents, 10), dtype=np.float32)
    # num_agents_to_add = min(len(valid_agents_info), config.num_closest_agents)
    # for i in range(num_agents_to_add):
    #     _, agent_idx = valid_agents_info[i]
    #     agent_pos_global = jnp.array([state.sim_trajectory.x[agent_idx, ts], state.sim_trajectory.y[agent_idx, ts]])
    #     agent_vel_global = jnp.array([state.sim_trajectory.vel_x[agent_idx, ts], state.sim_trajectory.vel_y[agent_idx, ts]])
    #     agent_heading_global = state.sim_trajectory.yaw[agent_idx, ts].item()
    #     agent_length = state.sim_trajectory.length[agent_idx, ts].item()
    #     agent_width = state.sim_trajectory.width[agent_idx, ts].item()
    #     obj_type = state.object_metadata.object_types[agent_idx].item()
    #     relative_pos = np.dot(rot_matrix, agent_pos_global - ego_pos)
    #     relative_vel = np.dot(rot_matrix, agent_vel_global)
    #     relative_heading = normalize_angle(agent_heading_global - ego_heading)
    #     type_vec = np.zeros(3, dtype=np.float32)
    #     if obj_type in [1, 2, 3]: type_vec[obj_type - 1] = 1.0
    #     features = np.concatenate([relative_pos, relative_vel, [relative_heading, agent_length, agent_width], type_vec])
    #     agent_features_padded[i, :] = features
    
    # --- 2. Agent Features ---
    # ##############################################################################
    # ### WARNING: TEMPORARY BUG-FOR-BUG COMPATIBLE LOGIC ###
    # ##############################################################################
    # The following logic is intentionally made to be bug-for-bug compatible with
    # the `preprocess.py` script used to generate the training data.
    #
    # THE BUG: The preprocessing script uses a "sort-then-filter" approach: it sorts
    # ALL agents by distance (including invalid ones) and then iterates through this
    # sorted list, picking the first N *valid* agents it finds.
    #
    # THE CORRECT LOGIC (commented out below) is to "filter-then-sort": first, find
    # all valid agents, and then sort that smaller, valid-only list by distance.
    #
    # This temporary code ensures that the live evaluation sees the exact same agent
    # feature distribution as the model was trained on. This block should be
    # reverted to the correct logic once the dataset is re-preprocessed.
    # ##############################################################################

    rot_matrix = np.array([[np.cos(ego_heading), np.sin(ego_heading)], [-np.sin(ego_heading), np.cos(ego_heading)]])
    
    # 1. Get positions of all agents and calculate distance to all of them
    all_agents_pos_global = jnp.stack([state.sim_trajectory.x[:, ts], state.sim_trajectory.y[:, ts]], axis=-1)
    dist_matrix_agents = jnp.linalg.norm(all_agents_pos_global - ego_pos, axis=1)
    
    # 2. Invalidate the SDC's own distance by setting it to infinity
    dist_matrix_agents = dist_matrix_agents.at[sdc_index].set(jnp.inf)
    
    # 3. Sort ALL agent indices by distance. This creates the flawed ordering.
    closest_agent_indices = jnp.argsort(dist_matrix_agents)

    # 4. Initialize the padded array and loop through the flawed sorted list
    agent_features_padded = np.zeros((config.num_closest_agents, 10), dtype=np.float32)
    num_agents_added = 0
    for agent_idx_item in closest_agent_indices:
        agent_idx = agent_idx_item.item()

        # Stop if we've filled our list of N agents
        if num_agents_added >= config.num_closest_agents:
            break
        
        # Explicitly skip the SDC itself (this part is a fix to the *other* bug we found)
        # if agent_idx == sdc_index:
            # continue

        # Now, check if this agent is valid *after* it has been selected by the sort
        if state.sim_trajectory.valid[agent_idx, ts]:
            # If valid, extract its data and compute features
            agent_pos_global = jnp.array([state.sim_trajectory.x[agent_idx, ts], state.sim_trajectory.y[agent_idx, ts]])
            agent_vel_global = jnp.array([state.sim_trajectory.vel_x[agent_idx, ts], state.sim_trajectory.vel_y[agent_idx, ts]])
            agent_heading_global = state.sim_trajectory.yaw[agent_idx, ts].item()
            agent_length = state.sim_trajectory.length[agent_idx, ts].item()
            agent_width = state.sim_trajectory.width[agent_idx, ts].item()
            obj_type = state.object_metadata.object_types[agent_idx].item()
            
            relative_pos = np.dot(rot_matrix, agent_pos_global - ego_pos)
            relative_vel = np.dot(rot_matrix, agent_vel_global)
            relative_heading = normalize_angle(agent_heading_global - ego_heading)
            type_vec = np.zeros(3, dtype=np.float32)
            if obj_type in [1, 2, 3]: type_vec[obj_type - 1] = 1.0
            
            features = np.concatenate([relative_pos, relative_vel, [relative_heading, agent_length, agent_width], type_vec])
            
            # Add the features to the padded array
            agent_features_padded[num_agents_added, :] = features
            num_agents_added += 1
            
    # ##############################################################################
    # ### END OF TEMPORARY LOGIC ###
    # ##############################################################################
    # --- 3. Map & Rule Features (FULL IMPLEMENTATION) ---
    
    # First, extract the full point cloud of map features from the state
    rg_points = state.roadgraph_points
    
    # Define the Waymax Internal Enums we need to identify point types
    WAYMAX_LANE_TYPES = {1, 2, 3}
    WAYMAX_CROSSWALK_TYPE = 18
    WAYMAX_STOP_SIGN_TYPE = 17
    
    # --- Lanes ---
    relative_lane_points = np.zeros((config.num_closest_map_points, 2), dtype=np.float32)
    lane_mask = jnp.isin(rg_points.types, jnp.array(list(WAYMAX_LANE_TYPES)))
    all_lane_points = jnp.stack([rg_points.x[lane_mask], rg_points.y[lane_mask]], axis=-1)

    if all_lane_points.shape[0] > 0:
        distances = jnp.linalg.norm(all_lane_points - ego_pos, axis=1)
        # Use np.asarray to handle JAX arrays safely
        closest_indices = np.argsort(np.asarray(distances))[:config.num_closest_map_points]
        closest_points_global = all_lane_points[closest_indices]
        transformed_points = np.dot(rot_matrix, (np.asarray(closest_points_global) - np.asarray(ego_pos)).T).T
        relative_lane_points[:transformed_points.shape[0], :] = transformed_points

    # --- Crosswalks ---
    relative_crosswalk_points = np.zeros((config.num_closest_crosswalk_points, 2), dtype=np.float32)
    cw_mask = (rg_points.types == WAYMAX_CROSSWALK_TYPE)
    all_crosswalk_points = jnp.stack([rg_points.x[cw_mask], rg_points.y[cw_mask]], axis=-1)

    if all_crosswalk_points.shape[0] > 0:
        distances = jnp.linalg.norm(all_crosswalk_points - ego_pos, axis=1)
        closest_indices = np.argsort(np.asarray(distances))[:config.num_closest_crosswalk_points]
        closest_points_global = all_crosswalk_points[closest_indices]
        transformed_points = np.dot(rot_matrix, (np.asarray(closest_points_global) - np.asarray(ego_pos)).T).T
        relative_crosswalk_points[:transformed_points.shape[0], :] = transformed_points

    # --- Rules (Stop Signs & Traffic Lights) ---
    dist_to_stop_sign = config.max_dist
    stop_sign_mask = (rg_points.types == WAYMAX_STOP_SIGN_TYPE)
    all_stop_sign_points = jnp.stack([rg_points.x[stop_sign_mask], rg_points.y[stop_sign_mask]], axis=-1)
    
    if all_stop_sign_points.shape[0] > 0:
        distances = jnp.linalg.norm(all_stop_sign_points - ego_pos, axis=1)
        dist_to_stop_sign = jnp.min(distances).item()

    is_stop_controlled = 0.0 # Placeholder for more complex logic
    
    # --- 4. Route and Rules Features ---
    # Goal Features
    final_ego_x = state.log_trajectory.x[sdc_index, -1]
    final_ego_y = state.log_trajectory.y[sdc_index, -1]
    
    final_destination = [final_ego_x, final_ego_y]
    ego_pos_np = np.array(ego_pos)
    final_destination = np.array(final_destination)
    distance_to_goal = np.linalg.norm(np.array(final_destination) - ego_pos_np)
    dir_to_goal_global = np.array(final_destination) - ego_pos_np
    dir_to_goal_ego = np.dot(rot_matrix, dir_to_goal_global)
    dir_norm = np.linalg.norm(dir_to_goal_ego)
    if dir_norm > 1e-4: dir_to_goal_ego /= dir_norm
    
    # Route Features (Future Waypoints)
    future_indices = np.clip(np.arange(ts + 5, ts + 51, 5), 0, 90)
    future_waypoints_global = state.log_trajectory.xy[sdc_index, future_indices]
    relative_future_waypoints = np.dot(rot_matrix, (np.array(future_waypoints_global) - ego_pos_np).T).T

    # Stop Sign Features
    dist_to_stop_sign = config.max_dist
    ss_mask = (rg_points.types == 17)
    if ss_mask.any():
        all_stop_sign_points = np.stack([np.array(rg_points.x[ss_mask]), np.array(rg_points.y[ss_mask])], axis=-1)
        distances = np.linalg.norm(all_stop_sign_points - ego_pos_np, axis=1)
        dist_to_stop_sign = np.min(distances)
    
    # Traffic Light and is_stop_controlled Features (Simplified for eval)
    # A full implementation would require rebuilding the lane connectivity and TL lookup
    ego_pos_jax = jnp.array(ego_pos)    
    current_lane_id = find_current_lane_id_live(ego_pos_jax, ego_heading, state.roadgraph_points)

    # 2. Look up the traffic light state for this lane directly from the state object
    tl_state_enum = 0 # Default to LANE_STATE_UNKNOWN
    if current_lane_id and state.log_traffic_light is not None:
        # Find which traffic light controls our current lane
        # state.log_traffic_light.lane_ids has shape (num_lights, num_lanes_controlled)
        
        # We search for our lane_id in the mapping
        light_controls_our_lane_mask = (state.log_traffic_light.lane_ids == current_lane_id).any(axis=-1)
        
        if light_controls_our_lane_mask.any():
            # Get the index of the first traffic light that controls our lane
            light_idx = jnp.argmax(light_controls_our_lane_mask).item()
            
            # Get the state of that specific light at the current timestep
            tl_state_enum = state.log_traffic_light.state[light_idx, ts].item()

    # 3. Convert the state enum to a one-hot vector (same as before)
    tl_state_vec = np.zeros(4, dtype=np.float32) # [G, Y, R, U]
    if tl_state_enum in [3, 6]: # GO states
        tl_state_vec[0] = 1.0
    elif tl_state_enum in [2, 5, 8]: # CAUTION states
        tl_state_vec[1] = 1.0
    elif tl_state_enum in [1, 4, 7]: # STOP states
        tl_state_vec[2] = 1.0
    else: # UNKNOWN
        tl_state_vec[3] = 1.0
        
    is_stop_controlled = 0.0

            
    # --- 5. Assemble Final Dictionary ---
    state_dict = {
        'ego': ego_features.astype(np.float32),
        'agents': agent_features_padded,
        'lanes': relative_lane_points,
        'crosswalks': relative_crosswalk_points,
        'route': relative_future_waypoints.astype(np.float32),
        'rules': np.concatenate([
            np.array([distance_to_goal, dir_to_goal_ego[0], dir_to_goal_ego[1], dist_to_stop_sign, is_stop_controlled]),
            tl_state_vec
        ]).astype(np.float32)
    }
    return state_dict


def main(args):
    print("--- Starting Stage 2.8: Structured MLP BC Policy Evaluation ---")
    config = TrainingConfig()

    # --- 1. Load Scenario and Environment ---
    print(f"Loading scenario: {args.scenario_id}")
    filepath = os.path.join(config.val_data_dir, f"{args.scenario_id}.npz")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scenario file not found: {filepath}.")
    
    data = np.load(filepath, allow_pickle=True)
    initial_state = construct_state_from_npz(data)
    
    # We ALSO load our preprocessed .pt file for this scenario.
    # This contains the exact state_dicts our model was trained on.
    pt_path = os.path.join(config.cql_preprocess_val_dir, f"{args.scenario_id}.pt")
    preprocessed_samples = torch.load(pt_path, weights_only=False)
    
    env_config = _config.EnvironmentConfig(
        max_num_objects=initial_state.num_objects,
        controlled_object=_config.ObjectType.SDC,
        metrics=_config.MetricsConfig(),
        rewards=_config.LinearCombinationRewardConfig(rewards={}),
    )
    dynamics_model = dynamics.InvertibleBicycleModel()
    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=env_config,
    )

    # --- 2. Load Trained Structured MLP Model ---
    model_path = config.bcs_model_path_v2 # Use the correct path from config
    print(f"Loading trained model from: {model_path}")
    model = StructuredMLP_BC_Model(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    model.eval()

    # --- 3. The Simulation Loop ---
    print("Starting closed-loop simulation...")
    current_state = env.reset(initial_state)
    rollout_states = [current_state]
    sdc_index = jnp.argmax(current_state.object_metadata.is_sdc).item()
    jit_step = jax.jit(env.step)

    for ts in tqdm(range(current_state.remaining_timesteps), desc="Simulating"):
        # a. Featurize state into a dictionary
        # feature_dict_np = preprocessed_samples[ts]['state']
        feature_dict = state_to_feature_dict(current_state, config)
# 
        # b. Infer Action (PyTorch)
        with torch.no_grad():
            state_dict_tensor = {
                k: torch.from_numpy(v).to(config.device).unsqueeze(0) 
                for k, v in feature_dict.items()
            }
            action_kinematic_tensor = model(state_dict_tensor)
            action_kinematic = action_kinematic_tensor.squeeze(0).cpu().numpy()

        # c. Step the Environment (Waymax)
        all_agent_actions = jnp.zeros((current_state.num_objects, 2))
        final_action_data = all_agent_actions.at[sdc_index].set(action_kinematic)
        action_valid_mask = jnp.zeros((current_state.num_objects, 1), dtype=bool).at[sdc_index].set(True)
        waymax_action = datatypes.Action(data=final_action_data, valid=action_valid_mask)
        
        current_state = jit_step(current_state, waymax_action)
        rollout_states.append(current_state)

    print("...Simulation complete.")

    # --- 4. Visualization ---
    print("Generating rollout video...")
    imgs = []
    for state in tqdm(rollout_states, desc="Rendering frames"):
        imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))

    output_path = os.path.join(args.output_dir, f"{args.scenario_id}_bcs_mlp_v2_rollout.mp4")
    mediapy.write_video(output_path, imgs, fps=10)
    print(f"--- ✅ Structured MLP BC Evaluation Complete. Video saved to: {output_path} ---")
    
        
        
    # # The simulation is correctly initialized to start at timestep 10
    # current_state = env.reset(initial_state) 
    
    # # 1. Generate features LIVE for the current state (which is at ts=10)
    # print("\n--- Generating LIVE features for ts=10 ---")
    # live_feature_dict = state_to_feature_dict(current_state, config)

    # # 2. Load the pre-calculated features for THE SAME timestep (ts=10)
    # print("--- Loading PREPROCESSED features for ts=10 ---")
    # # The list is 0-indexed, so index 10 corresponds to ts=10
    # preprocessed_feature_dict = preprocessed_samples[10]['state'] 
    
    # # 3. Now, compare them!
    # print("\n--- Comparing Dictionaries ---")
    # for key in live_feature_dict:
    #     live_val = live_feature_dict[key]
    #     preproc_val = preprocessed_feature_dict[key]
        
    #     # Use np.allclose for robust floating point comparison
    #     are_equal = np.allclose(live_val, preproc_val, atol=1e-5)
        
    #     if not are_equal:
    #         print(f"❌ MISMATCH found in key: '{key}'")
    #         # print("  Live value:\n", live_val)
    #         # print("  Preprocessed value:\n", preproc_val)
    #         print("  Difference:\n", live_val - preproc_val)
    #     else:
    #         print(f"✅ Match for key: '{key}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Structured MLP BC policy (Stage 2.8).")
    parser.add_argument("-s","--scenario_id", type=str,
                        default="bb08a56a2a870417", 
                        help="The scenario ID to evaluate.")
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.expanduser('~/WaymoOfflineAgent/outputs/evaluations/stage_2_8_structured_mlp_bc'),
        help="Directory to save the output video."
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)