# ======================================================================================
# evaluate_bc.py
#
# Description:
#   Loads a trained Behavioral Cloning policy and evaluates it in a closed-loop
#   simulation using the Waymax environment. The script performs the following steps:
#   1. Loads a scenario from a processed .npz file.
#   2. Loads the trained PyTorch model.
#   3. At each simulation step, converts the Waymax state to a feature vector.
#   4. Uses the model to predict an action.
#   5. Applies the action in the Waymax environment.
#   6. Saves a video of the resulting simulation rollout.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import torch
import numpy as np
import jax
from jax import numpy as jnp
import os
import glob
from tqdm import tqdm
import argparse

# Import our project-specific modules
from utils import MLP, TrainingConfig
from utils import construct_state_from_npz, state_to_feature_vector # We will create this file

# Import Waymax
from waymax import datatypes
from waymax import visualization
from waymax import env as _env
from waymax import dynamics
from waymax import config as _config
import mediapy # NEW: Add this import


def main(args):
    print("--- Starting BC Policy Evaluation ---")
    config = TrainingConfig() # Get feature config from our training script

    # --- 1. Load Scenario and Environment ---
    print(f"Loading scenario: {args.scenario_id}")
    filepath = os.path.join(config.val_data_dir.replace('bc_preprocessed', 'processed'), f"{args.scenario_id}.npz")
    data = np.load(filepath, allow_pickle=True)
    
    initial_state = construct_state_from_npz(data)
    
    env_config = _config.EnvironmentConfig(
        max_num_objects=initial_state.num_objects,
        controlled_object=_config.ObjectType.SDC,
        metrics=_config.MetricsConfig(),
        rewards=_config.LinearCombinationRewardConfig(rewards={}),
    )
    dynamics_model = dynamics.StateDynamics()
    env = _env.MultiAgentEnvironment(dynamics_model=dynamics_model, config=env_config)

    # --- 2. Load Trained PyTorch Model ---
    print(f"Loading trained model from: {config.model_save_path}")
    model = MLP(input_dim=279) # Use the correct input dim
    model.load_state_dict(torch.load(config.model_save_path, map_location=config.device, weights_only=False))
    model.to(config.device)
    model.eval() # Set model to evaluation mode

    # --- 3. The Simulation Loop ---
    print("Starting closed-loop simulation...")
    current_state = env.reset(initial_state)
    rollout_states = [current_state]
    
    sdc_index = jnp.argmax(current_state.object_metadata.is_sdc).item()

    for _ in tqdm(range(current_state.remaining_timesteps), desc="Simulating"):
        # a. Featurize
        feature_vector = state_to_feature_vector(current_state, config)
        
        # b. Infer Action (PyTorch)
        with torch.no_grad():
            state_tensor = torch.from_numpy(feature_vector).to(config.device).float().unsqueeze(0) # Add batch dim
            action_ego_tensor = model(state_tensor)
            action_ego = action_ego_tensor.squeeze(0).cpu().numpy() # Remove batch dim and move to CPU

        # c. De-transform Action: Convert model's ego-centric displacement to Waymax's absolute 5D action
        print("Converting action to Waymax format...")
        print(f"Predicted ego-centric action: {action_ego}")
        
        # Add a random noise to the action_ego
        action_ego += np.random.normal(0, 0.1, size=action_ego.shape)  # Add small noise
        
        # Get the SDC's CURRENT absolute state from the simulation trajectory
        sdc_current_sim_x = current_state.sim_trajectory.x[sdc_index, current_state.timestep]
        sdc_current_sim_y = current_state.sim_trajectory.y[sdc_index, current_state.timestep]
        sdc_current_sim_yaw = current_state.sim_trajectory.yaw[sdc_index, current_state.timestep]
        sdc_current_sim_vel_x = current_state.sim_trajectory.vel_x[sdc_index, current_state.timestep]
        sdc_current_sim_vel_y = current_state.sim_trajectory.vel_y[sdc_index, current_state.timestep]

        # Convert predicted ego-centric displacement (action_ego) to global delta
        # This part remains the same as before, correctly calculates global dx, dy
        ego_heading = sdc_current_sim_yaw # Use current sim yaw for rotation
        c, s = jnp.cos(ego_heading), jnp.sin(ego_heading)
        inv_rot_matrix = jnp.array([[c, -s], [s, c]]) # Inverse rotation to go from ego to global
        delta_global_xy = jnp.dot(inv_rot_matrix, action_ego) # This is [global_dx, global_dy]

        # Calculate ABSOLUTE next position (x_next, y_next)
        x_next = sdc_current_sim_x + delta_global_xy[0]
        y_next = sdc_current_sim_y + delta_global_xy[1]
        
        # For yaw, vel_x, vel_y, use current values (BC model doesn't predict these directly)
        # A more advanced model might predict deltas for these too.
        yaw_next = sdc_current_sim_yaw
        vel_x_next = sdc_current_sim_vel_x
        vel_y_next = sdc_current_sim_vel_y

        # Construct the final 5D action data for StateDynamics
        # This must be [x_next, y_next, yaw_next, vel_x_next, vel_y_next]
        action_5d_data = jnp.array([x_next, y_next, yaw_next, vel_x_next, vel_y_next])

        # d. Step the Environment (Waymax)
        # Create a full action array for all agents, default to 0 for uncontrolled agents
        # The shape must be (num_objects, 5)
        all_agent_actions_data = jnp.zeros((current_state.num_objects, 5))
        
        # Place our SDC's 5D action data in the correct slot
        final_action_data = all_agent_actions_data.at[sdc_index].set(action_5d_data)

        # Create the corresponding boolean validity mask (True for SDC, False for others)
        action_valid_mask = jnp.zeros((current_state.num_objects, 1), dtype=bool)
        final_action_valid = action_valid_mask.at[sdc_index].set(True)

        # Waymax expects a specific Action dataclass with both data and valid fields.
        waymax_action = datatypes.Action(
            data=final_action_data,
            valid=final_action_valid
        )

        current_state = env.step(current_state, waymax_action)
        rollout_states.append(current_state)

    print("...Simulation complete.")

    # --- 4. Visualization ---
    print("Generating rollout video...")
    imgs = []
    for state in tqdm(rollout_states, desc="Rendering frames"):
        # This part is correct
        imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))

    output_path = os.path.join(args.output_dir, f"{args.scenario_id}_bc_rollout.mp4")

    # Use mediapy.write_video to save the list of image arrays as an MP4.
    print(f"Saving video to: {output_path}")
    mediapy.write_video(output_path, imgs, fps=10) # 10 frames per second, as per Waymo's data rate

    print(f"--- Evaluation Complete. Video saved to: {output_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Behavioral Cloning policy.")
    parser.add_argument(
                "--scenario_id",
                type=str,
                default="1a1a379c4e09cc59",  # Example scenario ID
                help="The scenario ID to run the evaluation on (e.g., '1a1a379c4e09cc59')."  
                )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser('~/WaymoOfflineAgent/outputs/evaluations'),
        help="Directory to save the output video."
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)