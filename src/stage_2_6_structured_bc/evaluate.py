# ======================================================================================
# evaluate.py (Stage 2.6 - Structured BC)
#
# Description:
#   Loads the trained Structured Behavioral Cloning (BC-S) policy and evaluates
#   it in a closed-loop simulation using the Waymax environment with the
#   InvertibleBicycleModel.
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
from src.stage_2_6_structured_bc.networks import StructuredBCModel # Import our new structured model

# Import Waymax components
from waymax import datatypes
from waymax import visualization
from waymax import env as _env
from waymax import dynamics
from waymax import config as _config

# --- Live Feature Engineering for Structured Model ---
def state_to_feature_dict(state: datatypes.SimulatorState, config: TrainingConfig) -> dict:
    """
    Converts a live Waymax SimulatorState object into the structured dictionary
    of features that our models were trained on.
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
    rot_matrix = np.array([[np.cos(ego_heading), np.sin(ego_heading)], [-np.sin(ego_heading), np.cos(ego_heading)]])
    all_agents_x = state.sim_trajectory.x[:, ts]
    all_agents_y = state.sim_trajectory.y[:, ts]
    all_agents_valid = state.sim_trajectory.valid[:, ts]
    valid_agents_info = []
    for i in range(state.num_objects):
        if all_agents_valid[i] and i != sdc_index:
            agent_pos = jnp.array([all_agents_x[i], all_agents_y[i]])
            dist = jnp.linalg.norm(agent_pos - ego_pos).item()
            valid_agents_info.append((dist, i))
    valid_agents_info.sort(key=lambda item: item[0])
    agent_features_padded = np.zeros((config.num_closest_agents, 10), dtype=np.float32)
    num_agents_to_add = min(len(valid_agents_info), config.num_closest_agents)
    for i in range(num_agents_to_add):
        _, agent_idx = valid_agents_info[i]
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
        agent_features_padded[i, :] = features
    
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
    
    # In a full implementation, you would add logic here to check the
    # state.log_traffic_light data against the SDC's current lane.
    # For now, the placeholder is acceptable as we haven't built that yet.
    tl_state_vec = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) # Default to unknown
    
    all_rules_features = np.concatenate([np.array([dist_to_stop_sign, is_stop_controlled]), tl_state_vec]).astype(np.float32)

    # --- 4. Assemble Final Dictionary ---
    state_dict = {
        'ego': ego_features,
        'agents': agent_features_padded,
        'lanes': relative_lane_points,
        'crosswalks': relative_crosswalk_points,
        'rules': all_rules_features
    }
    return state_dict

def main(args):
    print("--- Starting Stage 2.6: Structured BC Policy Evaluation ---")
    config = TrainingConfig()

    # --- 1. Load Scenario and Environment ---
    print(f"Loading scenario: {args.scenario_id}")
    filepath = os.path.join(config.val_data_dir, f"{args.scenario_id}.npz")
    data = np.load(filepath, allow_pickle=True)
    initial_state = construct_state_from_npz(data)
    
    env_config = _config.EnvironmentConfig(
        max_num_objects=initial_state.num_objects,
        controlled_object=_config.ObjectType.SDC,
        metrics=_config.MetricsConfig(),
        rewards=_config.LinearCombinationRewardConfig(rewards={}),
    )
    dynamics_model = dynamics.InvertibleBicycleModel()
    env = _env.MultiAgentEnvironment(dynamics_model=dynamics_model, config=env_config)

    # --- 2. Load Trained Structured Model ---
    print(f"Loading trained model from: {config.bcs_model_path}")
    model = StructuredBCModel(config)
    model.load_state_dict(torch.load(config.bcs_model_path, map_location=config.device))
    model.to(config.device)
    model.eval()

    # --- 3. The Simulation Loop ---
    print("Starting closed-loop simulation...")
    current_state = env.reset(initial_state)
    rollout_states = [current_state]
    sdc_index = jnp.argmax(current_state.object_metadata.is_sdc).item()
    jit_step = jax.jit(env.step)

    for _ in tqdm(range(current_state.remaining_timesteps), desc="Simulating"):
        # a. Featurize state into a dictionary
        feature_dict = state_to_feature_dict(current_state, config)
        
        # b. Infer Action (PyTorch)
        with torch.no_grad():
            # Convert each numpy array in the dict to a batched tensor on the GPU
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

    output_path = os.path.join(args.output_dir, f"{args.scenario_id}_bcs_rollout.mp4")
    mediapy.write_video(output_path, imgs, fps=10)
    print(f"--- ✅ Structured BC Evaluation Complete. Video saved to: {output_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Structured BC policy (Stage 2.6).")
    parser.add_argument("scenario_id", type=str, help="The scenario ID to evaluate.")
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.expanduser('~/WaymoOfflineAgent/outputs/evaluations/stage_2_6_structured_bc'),
        help="Directory to save the output video."
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # This script depends on the shared utils file
    # We will need to make sure the state_to_feature_dict is fully implemented
    # and moved to utils.py for the final version.
    main(args)