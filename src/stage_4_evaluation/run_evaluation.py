# ======================================================================================
# run_evaluation.py (Stage 4)
#
# Description:
#   The master evaluation script for the project. This script can load either a
#   Behavioral Cloning (BC) or a Conservative Q-Learning (CQL) policy and run
#   it in a closed-loop simulation across a directory of validation scenarios.
#
#   It computes a standard set of metrics for each rollout (collision, off-road,
#   goal completion) and aggregates the results into a final JSON report.
#
#   This script unifies the evaluation logic for all models and ensures that
#   feature generation is consistent for fair comparison.
#
# Author: Antonio Guillen-Perez (and Gemini, your AI technical partner)
# ======================================================================================

import torch
import numpy as np
import jax
from jax import numpy as jnp
import os
import glob
import json
from tqdm import tqdm
import argparse

# --- Project-Specific Imports ---
# Shared utilities
from src.shared.utils import TrainingConfig, construct_state_from_npz, normalize_angle
# Models from different stages
from src.stage_2_8_structured_mlp_bc.networks import StructuredMLP_BC_Model
from src.stage_3_1_structured_mlp_cql.networks import Actor as CQL_Actor
# Evaluation metrics
import src.stage_4_evaluation.metrics as metrics

# --- Waymax Imports ---
from waymax import datatypes
from waymax import env as _env
from waymax import dynamics
from waymax import config as _config


# ======================================================================================
#  Canonical Feature Generation (Bug-for-Bug Compatible Version)
# ======================================================================================

def find_current_lane_id_live(ego_pos, ego_heading, roadgraph_points, map_data_dict):
    """
    Finds the ID of the lane the ego vehicle is most likely in from live
    Waymax RoadgraphPoints, using the original map_data_dict for polylines.
    """
    if not map_data_dict['lane_polylines']:
        return None

    best_lane_id = None
    min_dist = float('inf')

    # Convert JAX array to NumPy for this calculation
    ego_pos_np = np.array(ego_pos)

    for lane_id, polyline in map_data_dict['lane_polylines'].items():
        distances = np.linalg.norm(polyline[:, :2] - ego_pos_np, axis=1)
        closest_point_idx = np.argmin(distances)
        dist = distances[closest_point_idx]

        if dist < min_dist:
            if closest_point_idx < len(polyline) - 1:
                lane_dir = polyline[closest_point_idx + 1, :2] - polyline[closest_point_idx, :2]
            else:
                lane_dir = polyline[closest_point_idx, :2] - polyline[closest_point_idx - 1, :2]
            
            ego_dir_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])
            if np.dot(lane_dir, ego_dir_vec) > 0:
                min_dist = dist
                best_lane_id = lane_id

    if min_dist < 5.0:
        return best_lane_id
    return None

def state_to_feature_dict(state: datatypes.SimulatorState, config: TrainingConfig, map_data_dict) -> dict:
    """
    Converts a live Waymax SimulatorState into the structured dictionary of features.
    This is the UNIFIED function for all models. It contains the bug-for-bug compatible
    logic to match the pre-existing training dataset.
    """
    ts = state.timestep.item()
    sdc_index = jnp.argmax(state.object_metadata.is_sdc).item()
    ego_pos = jnp.array([state.sim_trajectory.x[sdc_index, ts], state.sim_trajectory.y[sdc_index, ts]])
    ego_heading = state.sim_trajectory.yaw[sdc_index, ts].item()
    rot_matrix = np.array([[np.cos(ego_heading), np.sin(ego_heading)], [-np.sin(ego_heading), np.cos(ego_heading)]])
    ego_pos_np = np.array(ego_pos)
    # --- 1. Ego Features ---
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

    # --- 2. Agent Features (Bug-for-bug compatible logic) ---
    all_agents_pos_global = jnp.stack([state.sim_trajectory.x[:, ts], state.sim_trajectory.y[:, ts]], axis=-1)
    dist_matrix_agents = jnp.linalg.norm(all_agents_pos_global - ego_pos, axis=1)
    dist_matrix_agents = dist_matrix_agents.at[sdc_index].set(jnp.inf)
    closest_agent_indices = jnp.argsort(dist_matrix_agents)
    agent_features_padded = np.zeros((config.num_closest_agents, 10), dtype=np.float32)
    num_agents_added = 0
    for agent_idx_item in closest_agent_indices:
        agent_idx = agent_idx_item.item()
        if num_agents_added >= config.num_closest_agents: break
        if state.sim_trajectory.valid[agent_idx, ts]:
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
            agent_features_padded[num_agents_added, :] = features
            num_agents_added += 1

    # --- 3. Map Features ---
    # Reconstruct the point clouds from the original map_data dictionary
    lane_polylines = list(map_data_dict['lane_polylines'].values())
    relative_lane_points = np.zeros((config.num_closest_map_points, 2), dtype=np.float32)
    if lane_polylines:
        all_lane_points = np.vstack(lane_polylines)[:, :2]
        distances = np.linalg.norm(all_lane_points - ego_pos_np, axis=1)
        closest_indices = np.argsort(distances)[:config.num_closest_map_points]
        closest_points_global = all_lane_points[closest_indices]
        transformed_points = np.dot(rot_matrix, (closest_points_global - ego_pos_np).T).T
        relative_lane_points[:transformed_points.shape[0], :] = transformed_points

    crosswalk_polygons = list(map_data_dict['crosswalk_polygons'].values())
    relative_crosswalk_points = np.zeros((config.num_closest_crosswalk_points, 2), dtype=np.float32)
    if crosswalk_polygons:
        all_crosswalk_points = np.vstack(crosswalk_polygons)[:, :2]
        distances = np.linalg.norm(all_crosswalk_points - ego_pos_np, axis=1)
        closest_indices = np.argsort(distances)[:config.num_closest_crosswalk_points]
        closest_points_global = all_crosswalk_points[closest_indices]
        transformed_points = np.dot(rot_matrix, (closest_points_global - ego_pos_np).T).T
        relative_crosswalk_points[:transformed_points.shape[0], :] = transformed_points

    # --- 4. Route and Rules Features ---
    # Goal Features (using log_trajectory as the ground truth plan)
    final_destination = state.log_trajectory.xy[sdc_index, -1]
    distance_to_goal = np.linalg.norm(np.array(final_destination) - ego_pos_np)
    dir_to_goal_global = np.array(final_destination) - ego_pos_np
    dir_to_goal_ego = np.dot(rot_matrix, dir_to_goal_global)
    dir_norm = np.linalg.norm(dir_to_goal_ego)
    if dir_norm > 1e-4: dir_to_goal_ego /= dir_norm
    
    # Route Features (Future Waypoints from log)
    future_indices = np.clip(np.arange(ts + 5, ts + 51, 5), 0, 90)
    future_waypoints_global = state.log_trajectory.xy[sdc_index, future_indices]
    relative_future_waypoints = np.dot(rot_matrix, (np.array(future_waypoints_global) - ego_pos_np).T).T

    # Stop Sign and Traffic Light Features
    dist_to_stop_sign = config.max_dist
    if map_data_dict['stopsign_positions']:
        stop_sign_positions = np.array([d['position'][:2] for d in map_data_dict['stopsign_positions'].values()])
        dist_to_stop_sign = np.min(np.linalg.norm(stop_sign_positions - ego_pos_np, axis=1))

    current_lane_id = find_current_lane_id_live(ego_pos, ego_heading, state.roadgraph_points, map_data_dict)
    
    tl_state_enum = 0
    if current_lane_id and state.log_traffic_light is not None:
        light_controls_our_lane_mask = (state.log_traffic_light.lane_ids == current_lane_id).any(axis=-1)
        if light_controls_our_lane_mask.any():
            light_idx = jnp.argmax(light_controls_our_lane_mask).item()
            tl_state_enum = state.log_traffic_light.state[light_idx, ts].item()

    tl_state_vec = np.zeros(4, dtype=np.float32)
    if tl_state_enum in [3, 6]: tl_state_vec[0] = 1.0
    elif tl_state_enum in [2, 5, 8]: tl_state_vec[1] = 1.0
    elif tl_state_enum in [1, 4, 7]: tl_state_vec[2] = 1.0
    else: tl_state_vec[3] = 1.0
    
    stop_controlled_lanes = set()
    if map_data_dict['stopsign_positions']:
        for stop_sign_data in map_data_dict['stopsign_positions'].values():
            stop_controlled_lanes.update(stop_sign_data['controls_lanes'])
    is_stop_controlled = 1.0 if current_lane_id in stop_controlled_lanes else 0.0

    # --- 5. Assemble Final Dictionary ---
    state_dict = {
        'ego': ego_features,
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

# ======================================================================================
#  Main Evaluation Logic
# ======================================================================================

def run_single_rollout(env, initial_state, model, model_type, config, map_data_dict):
    """Runs one full closed-loop simulation for a given initialized environment and model."""
    current_state = env.reset(initial_state)
    sdc_index = jnp.argmax(current_state.object_metadata.is_sdc).item()
    jit_step = jax.jit(env.step)
    
    for i in range(current_state.remaining_timesteps):
        # print(f"Step {i+1}/{current_state.remaining_timesteps} - Timestep {current_state.timestep.item()}")
        # 1. Featurize current state
        feature_dict = state_to_feature_dict(current_state, config, map_data_dict)
        state_dict_tensor = {k: torch.from_numpy(v).to(config.device).unsqueeze(0) for k, v in feature_dict.items()}

        # 2. Get action from the loaded model
        with torch.no_grad():
            if model_type == 'bcs':
                action_tensor = model(state_dict_tensor)
            elif model_type == 'cql':
                mean, _ = model(state_dict_tensor)
                y_t = torch.tanh(mean)
                action_tensor = y_t * model.action_scale + model.action_bias
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            action_kinematic = action_tensor.squeeze(0).cpu().numpy()

        # 3. Step the environment
        all_agent_actions = jnp.zeros((current_state.num_objects, 2))
        final_action_data = all_agent_actions.at[sdc_index].set(action_kinematic)
        action_valid_mask = jnp.zeros((current_state.num_objects, 1), dtype=bool).at[sdc_index].set(True)
        waymax_action = datatypes.Action(data=final_action_data, valid=action_valid_mask)
        current_state = jit_step(current_state, waymax_action)
    
    return current_state

def main(args):
    print(f"--- Starting Evaluation for Model Type: {args.model_type.upper()} ---")
    config = TrainingConfig()

    # --- 1. Load Model ---
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    if args.model_type == 'bcs':
        model = StructuredMLP_BC_Model(config)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=config.device))
    elif args.model_type == 'cql':
        model = CQL_Actor(config)
        checkpoint = torch.load(args.checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['actor_state_dict'])
    else:
        raise ValueError(f"Invalid model_type specified: {args.model_type}")
    
    model.to(config.device)
    model.eval()

    # --- 2. Load Scenarios ---
    scenario_files = sorted(glob.glob(os.path.join(args.scenario_dir, '*.npz')))
    if args.num_scenarios > 0:
        scenario_files = scenario_files[:args.num_scenarios]
    print(f"Found {len(scenario_files)} scenarios to evaluate.")
    
    # --- 3. Main Evaluation Loop ---
    all_scenario_results = []
    dynamics_model = dynamics.InvertibleBicycleModel()
    
    for scenario_path in tqdm(scenario_files, desc="Evaluating Scenarios"):
        scenario_id = os.path.basename(scenario_path).replace('.npz', '')
        try:
            # a. Initialize environment for this scenario
            data = np.load(scenario_path, allow_pickle=True)
            map_data_dict = data['map_data'].item()

            initial_state = construct_state_from_npz(data)
            
            env_config = _config.EnvironmentConfig(
                max_num_objects=initial_state.num_objects,
                controlled_object=_config.ObjectType.SDC,
                metrics=_config.MetricsConfig(),
                rewards=_config.LinearCombinationRewardConfig(rewards={}),
            )
            env = _env.MultiAgentEnvironment(
            dynamics_model=dynamics_model,
            config=env_config,
        )

            # b. Run the full simulation
            final_state = run_single_rollout(env, initial_state, model, args.model_type, config, map_data_dict) # Pass it here

            sdc_index = jnp.argmax(final_state.object_metadata.is_sdc).item()

            # c. Compute metrics on the completed rollout
            collided = metrics.check_collision(final_state, sdc_index)
            off_road = metrics.check_off_road(final_state, sdc_index)
            goal_completed = metrics.check_goal_completion(final_state, sdc_index)
            
            # A scenario is a "success" if it completes the goal without collision or going off-road
            is_success = goal_completed and not collided and not off_road

            all_scenario_results.append({
                'scenario_id': scenario_id,
                'collided': collided,
                'off_road': off_road,
                'goal_completed': goal_completed,
                'is_success': is_success
            })
        
        except Exception as e:
            print(f"\nERROR: Failed to process scenario {scenario_id}. Reason: {e}")
            all_scenario_results.append({
                'scenario_id': scenario_id, 'error': str(e)
            })

    # --- 4. Aggregate and Save Results ---
    num_evaluated = len(all_scenario_results)
    total_successes = sum(1 for r in all_scenario_results if r.get('is_success', False))
    total_collisions = sum(1 for r in all_scenario_results if r.get('collided', False))
    total_off_road = sum(1 for r in all_scenario_results if r.get('off_road', False))
    total_goal_completed = sum(1 for r in all_scenario_results if r.get('goal_completed', False))
    total_errors = sum(1 for r in all_scenario_results if 'error' in r)

    summary = {
        'model_type': args.model_type,
        'checkpoint_path': args.checkpoint_path,
        'num_scenarios_evaluated': num_evaluated,
        'aggregation': {
            'success_rate': (total_successes / num_evaluated) * 100 if num_evaluated > 0 else 0,
            'collision_rate': (total_collisions / num_evaluated) * 100 if num_evaluated > 0 else 0,
            'off_road_rate': (total_off_road / num_evaluated) * 100 if num_evaluated > 0 else 0,
            'goal_completion_rate': (total_goal_completed / num_evaluated) * 100 if num_evaluated > 0 else 0,
            'error_rate': (total_errors / num_evaluated) * 100 if num_evaluated > 0 else 0
        },
        'per_scenario_results': all_scenario_results
    }

    print("\n--- Evaluation Summary ---")
    print(json.dumps(summary['aggregation'], indent=4))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✅ Full results saved to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master evaluation script for BC and CQL models.")
    parser.add_argument('--model_type', type=str, required=True, choices=['bcs', 'cql'],
                        help="Type of the model to evaluate: 'bcs' for MLP BC or 'cql' for CQL.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the trained model .pth checkpoint file.")
    parser.add_argument('--scenario_dir', type=str,
                        default=os.path.expanduser('~/WaymoOfflineAgent/data/processed_validation'),
                        help="Directory containing the .npz validation scenarios.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to save the final JSON results file.")
    parser.add_argument('--num_scenarios', type=int, default=-1,
                        help="Number of scenarios to evaluate. -1 for all.")
    
    args = parser.parse_args()
    main(args)
    
'''

python -m src.stage_4_evaluation.run_evaluation \
  --model_type bcs \
  --checkpoint_path ~/WaymoOfflineAgent/models/bc_structured_policy_v2.pth \
  --output_path ~/WaymoOfflineAgent/outputs/eval_results/bcs_v2_results.json \
  --num_scenarios 50 # Use -1 to run on all validation scenarios
  
python -m src.stage_4_evaluation.run_evaluation \
  --model_type cql \
  --checkpoint_path ~/WaymoOfflineAgent/runs/CQL_MLP_2025-08-07_07-50-00_lr3e-05_gamma0.95_cqlalpha10.0/models/cql_mlp_policy_epoch_47.pth \
  --output_path ~/WaymoOfflineAgent/outputs/eval_results/cql_results.json \
  --num_scenarios 50 # Use -1 to run on all validation scenarios

'''